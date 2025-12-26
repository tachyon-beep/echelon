from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..actions import ActionIndex
from ..constants import PACK_SIZE
from ..gen.objective import capture_zone_params
from ..sim.los import has_los

if TYPE_CHECKING:
    from ..env.env import EchelonEnv
    from ..sim.sim import Sim
    from ..sim.world import VoxelWorld


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
    - Optional handicap: suppress weapon outputs on some ticks.
    - Squad Cohesion: Move to squad center if far away.
    - Anti-Stuck: Back off and strafe if blocked.
    """

    def __init__(
        self,
        desired_range: float = 5.5,
        approach_speed_scale: float = 1.0,
        weapon_fire_prob: float = 0.5,
    ):
        self.desired_range = float(desired_range)
        self.approach_speed_scale = float(approach_speed_scale)
        self.weapon_fire_prob = float(weapon_fire_prob)
        # State tracking: {mid: {"last_pos": np.array, "stuck_timer": float, "maneuver": str, "maneuver_timer": float}}
        self.states: dict[str, dict] = {}

    def act(
        self, env: EchelonEnv, mech_id: str, sim: Sim | None = None, world: VoxelWorld | None = None
    ) -> np.ndarray:
        sim = sim or env.sim
        world = world or env.world
        if sim is None or world is None:
            raise RuntimeError("env or (sim and world) must be provided")

        mech = sim.mechs[mech_id]
        if not mech.alive:
            return np.zeros(env.ACTION_DIM, dtype=np.float32)

        zone_center: np.ndarray | None = None
        zone_radius = 0.0
        try:
            cx, cy, zr = capture_zone_params(world.meta, size_x=world.size_x, size_y=world.size_y)
            zone_center = np.array([cx, cy, float(mech.pos[2])], dtype=np.float32)
            zone_radius = float(zr)
        except Exception:
            zone_center = None
            zone_radius = 0.0

        # Initialize state
        if mech_id not in self.states:
            self.states[mech_id] = {
                "last_pos": mech.pos.copy(),
                "stuck_counter": 0,
                "maneuver": None,  # "reverse", "strafe"
                "maneuver_timer": 0.0,
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
                    state["maneuver_timer"] = 1.0  # Strafe for 1s
                    # Pick random strafe dir
                    state["strafe_dir"] = 1.0 if np.random.random() > 0.5 else -1.0
                else:
                    state["maneuver"] = None  # Done

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
        if dist_moved < 0.2:  # Stuck threshold
            state["stuck_counter"] += 1
        else:
            state["stuck_counter"] = max(0, state["stuck_counter"] - 1)

        if state["stuck_counter"] > 10:  # Stuck for ~10 decision steps (approx 2-3s)
            # Trigger Unstick
            state["maneuver"] = "reverse"
            state["maneuver_timer"] = 1.0
            state["stuck_counter"] = 0
            return np.zeros(env.ACTION_DIM, dtype=np.float32)  # Wait for next step to execute

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

        dist, target, _delta = best

        # All mechs prioritize the objective (zone); fall back to enemy if no zone.
        move_target = zone_center if zone_center is not None else target.pos

        # Cohesion Logic: Find squad center
        squad_pos = np.zeros(3, dtype=np.float32)
        squad_count = 0
        for m in sim.mechs.values():
            if m.alive and m.team == mech.team:
                squad_pos += m.pos
                squad_count += 1

        if squad_count > 1:
            centroid = squad_pos / squad_count
            dist_to_squad = float(np.linalg.norm(centroid - mech.pos))

            # Only cluster AFTER reaching the zone area - don't let slow mechs hold others back
            dist_to_zone = (
                float(np.linalg.norm(zone_center[:2] - mech.pos[:2])) if zone_center is not None else 999.0
            )

            # Role-based cohesion: different mechs have different clustering preferences
            # Heavies want to stay closer to squad (protected asset), scouts/lights spread out
            # This prevents the "tight ball of death" problem
            cohesion_threshold: float
            centroid_weight: float
            if mech.spec.name == "heavy":
                # Heavies stay closer - they're the protected fire support
                cohesion_threshold = 20.0
                centroid_weight = 0.4
            elif mech.spec.name in ("scout", "light"):
                # Scouts and lights spread out - they're flankers/recon
                cohesion_threshold = 35.0
                centroid_weight = 0.15
            else:
                # Medium/leader - moderate cohesion
                cohesion_threshold = 28.0
                centroid_weight = 0.25

            if dist_to_zone < zone_radius * 2 and dist_to_squad > cohesion_threshold and dist > 25.0:
                # Blend to squad with role-appropriate weight
                move_target = centroid * centroid_weight + move_target * (1.0 - centroid_weight)

        # Re-calculate delta based on move_target
        move_delta = move_target - mech.pos

        # Aim at the enemy, even when moving toward the objective.
        aim_target = target.pos
        aim_delta = aim_target - mech.pos
        desired_yaw = math.atan2(float(aim_delta[1]), float(aim_delta[0]))
        yaw_err = _wrap_pi(desired_yaw - float(mech.yaw))
        yaw_rate = float(np.clip(yaw_err / (math.pi / 3), -1.0, 1.0))

        # Local-frame movement (forward/strafe), keeping some distance.
        c = math.cos(mech.yaw)
        s = math.sin(mech.yaw)
        forward_vec = np.asarray([c, s], dtype=np.float32)
        right_vec = np.asarray([-s, c], dtype=np.float32)
        dir_xy = move_delta[:2].astype(np.float32, copy=False)
        n = float(np.linalg.norm(dir_xy))
        if n > 1e-6:
            dir_xy /= n

        forward_throttle = float(np.clip(np.dot(dir_xy, forward_vec), -1.0, 1.0))
        strafe_throttle = float(np.clip(np.dot(dir_xy, right_vec), -1.0, 1.0))

        in_zone = False
        if zone_center is not None and zone_radius > 0.0:
            in_zone = float(np.linalg.norm(mech.pos[:2] - zone_center[:2])) <= zone_radius

        # Back off if we're too close, except for heavies holding the zone.
        is_heavy_holding = mech.spec.name == "heavy" and in_zone
        if dist < self.desired_range and not is_heavy_holding:
            forward_throttle = -0.3
            strafe_throttle = float(np.clip(strafe_throttle * 1.0, -1.0, 1.0))
        else:
            strafe_throttle *= 0.25

        # Scale approach speed (1.0 = full speed to zone, then slow for combat)
        if in_zone:
            # Already in zone - slow down for combat maneuvers
            forward_throttle *= 0.5
            strafe_throttle *= 0.5
        else:
            # Not in zone - move at full speed to get there
            forward_throttle *= self.approach_speed_scale
            strafe_throttle *= self.approach_speed_scale

        vertical = 0.0

        # Vent/firing logic.
        vent = 1.0 if mech.heat > 0.75 * mech.spec.heat_cap else 0.0

        primary = 0.0
        secondary = 0.0
        tertiary = 0.0

        if vent <= 0.5:
            # Laser logic
            laser_in_range = dist <= 8.0
            laser_facing = abs(yaw_err) <= math.radians(60.0)
            los_ok = has_los(world, mech.pos, target.pos)

            # Missile logic (Heavy only effectively)
            # Paint lock = 50 range, LOS only = 12 range
            painted_lock = False
            if target.painted_remaining > 0.0 and target.last_painter_id is not None:
                painter = sim.mechs.get(target.last_painter_id)
                if painter is not None and painter.team == mech.team:
                    p_me = _pack_index(mech_id)
                    p_painter = _pack_index(painter.mech_id)
                    painted_lock = (p_me is not None) and (p_me == p_painter)
            missile_range = 50.0 if painted_lock else 12.0
            missile_in_range = dist <= missile_range
            missile_facing = abs(yaw_err) <= math.radians(90.0)
            can_fire_missile = missile_in_range and missile_facing and (los_ok or painted_lock)

            # Kinetic logic (Gauss=20, AC=12)
            kinetic_range = 20.0 if mech.spec.name == "heavy" else 12.0
            kinetic_in_range = dist <= kinetic_range
            kinetic_facing = abs(yaw_err) <= math.radians(30.0)

            # Paint logic (range=10)
            paint_in_range = dist <= 10.0
            paint_facing = abs(yaw_err) <= math.radians(30.0)

            if mech.spec.name == "scout":
                # Scout: PRIMARY=Paint, SECONDARY=EWAR
                primary = 1.0 if (paint_in_range and paint_facing and los_ok) else 0.0
                # EWAR logic below
            elif mech.spec.name == "light":
                # Light: PRIMARY=Flamer, SECONDARY=Laser, TERTIARY=Paint
                # Heuristic: flamer if close, laser if mid, paint if possible
                primary = 1.0 if (dist <= 4.5 and laser_facing and los_ok) else 0.0
                secondary = 1.0 if (laser_in_range and laser_facing and los_ok) else 0.0
                tertiary = 1.0 if (paint_in_range and paint_facing and los_ok) else 0.0
            elif mech.spec.name == "medium":
                # Medium: PRIMARY=Laser, TERTIARY=Autocannon, (TERTIARY also Paint)
                primary = 1.0 if (laser_in_range and laser_facing and los_ok) else 0.0
                tertiary = 1.0 if (kinetic_in_range and kinetic_facing) else 0.0
            elif mech.spec.name == "heavy":
                # Heavy: PRIMARY=Laser, SECONDARY=Missile, TERTIARY=Gauss
                primary = 1.0 if (laser_in_range and laser_facing and los_ok) else 0.0
                secondary = 1.0 if can_fire_missile else 0.0
                tertiary = 1.0 if (kinetic_in_range and kinetic_facing) else 0.0

        a = np.zeros(env.ACTION_DIM, dtype=np.float32)
        a[ActionIndex.FORWARD] = forward_throttle
        a[ActionIndex.STRAFE] = strafe_throttle
        a[ActionIndex.VERTICAL] = vertical
        a[ActionIndex.YAW_RATE] = yaw_rate
        a[ActionIndex.PRIMARY] = primary
        a[ActionIndex.VENT] = vent
        a[ActionIndex.SECONDARY] = secondary
        a[ActionIndex.TERTIARY] = tertiary
        a[ActionIndex.SPECIAL] = 0.0  # Smoke not used by heuristic yet

        # Target selection: focus our chosen target if it exists in the last-observed contact slots.
        slots = env._last_contact_slots.get(mech_id)
        if slots is not None:
            for i, oid in enumerate(slots[: env.CONTACT_SLOTS]):
                if oid == target.mech_id:
                    a[env.TARGET_START + int(i)] = 1.0
                    break

        # EWAR logic
        if mech.spec.name == "scout":
            if mech.heat < 0.7 * mech.spec.heat_cap:
                # Value (0, 0.5] = ECM
                a[ActionIndex.SECONDARY] = 0.25
            else:
                # Value > 0.5 = ECCM
                a[ActionIndex.SECONDARY] = 0.75

        # Handicap: fire less often (suppresses weapon outputs on a fraction of eligible ticks).
        if self.weapon_fire_prob < 1.0:
            wants_fire = (
                float(a[ActionIndex.PRIMARY]) > 0.0
                or float(a[ActionIndex.TERTIARY]) > 0.0
                or float(a[ActionIndex.SPECIAL]) > 0.0
                or (mech.spec.name != "scout" and float(a[ActionIndex.SECONDARY]) > 0.0)
            )
            if wants_fire and float(env.rng.random()) > self.weapon_fire_prob:
                a[ActionIndex.PRIMARY] = 0.0
                a[ActionIndex.TERTIARY] = 0.0
                a[ActionIndex.SPECIAL] = 0.0
                if mech.spec.name != "scout":
                    a[ActionIndex.SECONDARY] = 0.0

        # Observation controls: keep the contact table hostile-focused and sorted by closest.
        a[env.OBS_CTRL_START + 0] = 1.0  # closest
        a[env.OBS_CTRL_START + 1] = 0.0
        a[env.OBS_CTRL_START + 2] = 0.0
        a[env.OBS_CTRL_START + 3] = 1.0  # hostile-only
        return a
