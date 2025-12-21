from __future__ import annotations

import math

import numpy as np

from ..actions import ActionIndex
from ..constants import PACK_SIZE
from ..env.env import EchelonEnv
from ..sim.los import has_los


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _pack_index(mech_id: str) -> int | None:
    _, sep, suffix = mech_id.rpartition("_")
    if not sep:
        return None
    try:
        idx = int(suffix)
    except ValueError:
        return None
    return int(idx) // int(PACK_SIZE)


class HeuristicPolicy:
    """
    Simple baseline controller (cheats by reading env.sim state).

    Behavior:
    - Turn to face nearest enemy.
    - Move toward them until "brawling" distance; then strafe.
    - Fire laser when in LOS and roughly facing target.
    - Vent when heat is high.
    - Squad Cohesion: Move to squad center if far away.
    - Anti-Stuck: Back off and strafe if blocked.
    """

    def __init__(self, desired_range: float = 5.5, approach_speed_scale: float = 0.5):
        self.desired_range = float(desired_range)
        self.approach_speed_scale = float(approach_speed_scale)
        # State tracking: {mid: {"last_pos": np.array, "stuck_timer": float, "maneuver": str, "maneuver_timer": float}}
        self.states: dict[str, dict] = {}

    def act(self, env: EchelonEnv, mech_id: str) -> np.ndarray:
        sim = env.sim
        world = env.world
        if sim is None or world is None:
            raise RuntimeError("env must be reset() before act()")

        mech = sim.mechs[mech_id]
        if not mech.alive:
            return np.zeros(env.ACTION_DIM, dtype=np.float32)

        # Initialize state
        if mech_id not in self.states:
            self.states[mech_id] = {
                "last_pos": mech.pos.copy(),
                "stuck_counter": 0,
                "maneuver": None, # "reverse", "strafe"
                "maneuver_timer": 0.0
            }
        
        state = self.states[mech_id]
        dt = env.config.dt_sim * env.config.decision_repeat
        
        # --- Anti-Stuck Logic ---
        # If currently performing a maneuver, execute it
        if state["maneuver"]:
            state["maneuver_timer"] -= dt
            if state["maneuver_timer"] <= 0:
                # Next stage or finish
                if state["maneuver"] == "reverse":
                    state["maneuver"] = "strafe"
                    state["maneuver_timer"] = 1.0 # Strafe for 1s
                    # Pick random strafe dir
                    state["strafe_dir"] = 1.0 if np.random.random() > 0.5 else -1.0
                else:
                    state["maneuver"] = None # Done
            
            # Execute Maneuver
            forward = -1.0 if state["maneuver"] == "reverse" else 0.0
            strafe = state["strafe_dir"] if state["maneuver"] == "strafe" else 0.0
            
            # Reset stuck counter so we don't trigger immediately again
            state["last_pos"] = mech.pos.copy()
            state["stuck_counter"] = 0
            
            a = np.zeros(env.ACTION_DIM, dtype=np.float32)
            a[ActionIndex.FORWARD] = forward
            a[ActionIndex.STRAFE] = strafe
            return a

        # Detect Stuck
        # Check distance moved since last check (every ~0.5s?)
        # Let's check every step. If moved < 0.1m and we WANTED to move.
        dist_moved = np.linalg.norm(mech.pos - state["last_pos"])
        state["last_pos"] = mech.pos.copy()
        
        # We only care if we are trying to move forward
        # But we don't know "tried" throttle from previous step here easily without saving it.
        # Assume we always try to move.
        if dist_moved < 0.2: # Stuck threshold
            state["stuck_counter"] += 1
        else:
            state["stuck_counter"] = max(0, state["stuck_counter"] - 1)
        
        if state["stuck_counter"] > 10: # Stuck for ~10 decision steps (approx 2-3s)
            # Trigger Unstick
            state["maneuver"] = "reverse"
            state["maneuver_timer"] = 1.0
            state["stuck_counter"] = 0
            return np.zeros(env.ACTION_DIM, dtype=np.float32) # Wait for next step to execute

        # --- Normal Logic ---
        best = None
        for other in sim.mechs.values():
            if not other.alive or other.team == mech.team:
                continue
            d = other.pos - mech.pos
            dist = float(np.linalg.norm(d))
            if best is None or dist < best[0]:
                best = (dist, other, d)

        if best is None:
            return np.zeros(env.ACTION_DIM, dtype=np.float32)

        dist, target, delta = best
        
        # Cohesion Logic: Find squad center
        squad_pos = np.zeros(3, dtype=np.float32)
        squad_count = 0
        for m in sim.mechs.values():
            if m.alive and m.team == mech.team:
                squad_pos += m.pos
                squad_count += 1
        
        move_target = target.pos # Default: Move to enemy
        
        if squad_count > 1:
            centroid = squad_pos / squad_count
            dist_to_squad = float(np.linalg.norm(centroid - mech.pos))
            
            # If far from squad and safe-ish, move to squad first
            if dist_to_squad > 15.0 and dist > 20.0:
                # Blend 70% to squad, 30% to enemy
                move_target = centroid * 0.7 + target.pos * 0.3
        
        # Re-calculate delta based on move_target
        move_delta = move_target - mech.pos
        
        desired_yaw = math.atan2(float(move_delta[1]), float(move_delta[0]))
        yaw_err = _wrap_pi(desired_yaw - float(mech.yaw))
        yaw_rate = float(np.clip(yaw_err / (math.pi / 3), -1.0, 1.0))

        # Local-frame movement (forward/strafe), keeping some distance.
        c = math.cos(mech.yaw)
        s = math.sin(mech.yaw)
        forward = np.asarray([c, s], dtype=np.float32)
        right = np.asarray([-s, c], dtype=np.float32)
        dir_xy = move_delta[:2].astype(np.float32, copy=False)
        n = float(np.linalg.norm(dir_xy))
        if n > 1e-6:
            dir_xy /= n

        forward_throttle = float(np.clip(np.dot(dir_xy, forward), -1.0, 1.0))
        strafe_throttle = float(np.clip(np.dot(dir_xy, right), -1.0, 1.0))

        if dist < self.desired_range:
            forward_throttle = -0.3
            strafe_throttle = float(np.clip(strafe_throttle * 1.0, -1.0, 1.0))
        else:
            strafe_throttle *= 0.25
        
        # Slow down approach so the baseline doesn't instantly brawl.
        forward_throttle *= self.approach_speed_scale
        strafe_throttle *= self.approach_speed_scale

        vertical = 0.0

        # Vent/firing logic.
        vent = 1.0 if mech.heat > 0.75 * mech.spec.heat_cap else 0.0
        fire_laser = 0.0
        fire_missile = 0.0
        fire_kinetic = 0.0
        paint = 0.0
        
        if vent <= 0.5:
            # Laser logic
            in_range = dist <= 8.0
            facing_ok = abs(yaw_err) <= math.radians(60.0)
            los_ok = has_los(world, mech.pos, target.pos)
            fire_laser = 1.0 if (in_range and facing_ok and los_ok) else 0.0
            
            # Missile logic (Heavy only effectively, but logic is generic)
            # Range 35, Arc 180 (so +/- 90 yaw err)
            missile_in_range = dist <= 35.0
            missile_facing = abs(yaw_err) <= math.radians(90.0)
            painted_lock = False
            if target.painted_remaining > 0.0 and target.last_painter_id is not None:
                painter = sim.mechs.get(target.last_painter_id)
                if painter is not None and painter.team == mech.team:
                    p_me = _pack_index(mech_id)
                    p_painter = _pack_index(painter.mech_id)
                    painted_lock = (p_me is not None) and (p_me == p_painter)
            can_fire_missile = missile_in_range and missile_facing and (los_ok or painted_lock)
            fire_missile = 1.0 if can_fire_missile else 0.0
            
            # Kinetic logic (Gauss/AC)
            # Gauss Range 60, AC Range 20
            # Simple check: if medium and < 20, or heavy and < 60
            kinetic_range = 60.0 if mech.spec.name == "heavy" else 20.0
            kinetic_in_range = dist <= kinetic_range
            kinetic_facing = abs(yaw_err) <= math.radians(30.0) # Need better aim
            # Dumbfire needs LOS roughly? Or ballistic arc over wall?
            # Heuristic just shoots if facing
            fire_kinetic = 1.0 if (kinetic_in_range and kinetic_facing) else 0.0

            # Paint logic (Light only)
            # Range 15, Arc 60
            paint_in_range = dist <= 15.0
            paint_facing = abs(yaw_err) <= math.radians(30.0)
            paint = 1.0 if (paint_in_range and paint_facing and los_ok) else 0.0

        a = np.zeros(env.ACTION_DIM, dtype=np.float32)
        a[ActionIndex.FORWARD] = forward_throttle
        a[ActionIndex.STRAFE] = strafe_throttle
        a[ActionIndex.VERTICAL] = vertical
        a[ActionIndex.YAW_RATE] = yaw_rate
        a[ActionIndex.FIRE_LASER] = fire_laser
        a[ActionIndex.VENT] = vent
        a[ActionIndex.FIRE_MISSILE] = fire_missile
        a[ActionIndex.PAINT] = paint
        a[ActionIndex.FIRE_KINETIC] = fire_kinetic
        return a
