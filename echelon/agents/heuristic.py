from __future__ import annotations

import math

import numpy as np

from ..env.env import EchelonEnv
from ..sim.los import has_los


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class HeuristicPolicy:
    """
    Simple baseline controller (cheats by reading env.sim state).

    Behavior:
    - Turn to face nearest enemy.
    - Move toward them until "brawling" distance; then strafe.
    - Fire laser when in LOS and roughly facing target.
    - Vent when heat is high.
    """

    def __init__(self, desired_range: float = 5.5):
        self.desired_range = float(desired_range)

    def act(self, env: EchelonEnv, mech_id: str) -> np.ndarray:
        sim = env.sim
        world = env.world
        if sim is None or world is None:
            raise RuntimeError("env must be reset() before act()")

        mech = sim.mechs[mech_id]
        if not mech.alive:
            return np.zeros(env.ACTION_DIM, dtype=np.float32)

        # Find nearest enemy (perfect info baseline).
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
        desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
        yaw_err = _wrap_pi(desired_yaw - float(mech.yaw))
        yaw_rate = float(np.clip(yaw_err / (math.pi / 3), -1.0, 1.0))

        # Local-frame movement (forward/strafe), keeping some distance.
        c = math.cos(mech.yaw)
        s = math.sin(mech.yaw)
        forward = np.asarray([c, s], dtype=np.float32)
        right = np.asarray([-s, c], dtype=np.float32)
        dir_xy = delta[:2].astype(np.float32, copy=False)
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

        vertical = 0.0

        # Vent/firing logic.
        vent = 1.0 if mech.heat > 0.75 * mech.spec.heat_cap else 0.0
        fire_laser = 0.0
        fire_missile = 0.0
        paint = 0.0
        
        if vent <= 0.5:
            # Laser logic
            in_range = dist <= 8.0
            facing_ok = abs(yaw_err) <= math.radians(60.0)
            los_ok = has_los(world, mech.pos, target.pos)
            fire_laser = 1.0 if (in_range and facing_ok and los_ok) else 0.0
            
            # Missile logic (Heavy only effectively, but logic is generic)
            # Range 25, Arc 180 (so +/- 90 yaw err)
            missile_in_range = dist <= 25.0
            missile_facing = abs(yaw_err) <= math.radians(90.0)
            is_painted = target.painted_remaining > 0.0
            can_fire_missile = missile_in_range and missile_facing and (los_ok or is_painted)
            fire_missile = 1.0 if can_fire_missile else 0.0
            
            # Paint logic (Light only)
            # Range 15, Arc 60
            paint_in_range = dist <= 15.0
            paint_facing = abs(yaw_err) <= math.radians(30.0)
            paint = 1.0 if (paint_in_range and paint_facing and los_ok) else 0.0

        return np.asarray(
            [forward_throttle, strafe_throttle, vertical, yaw_rate, fire_laser, vent, fire_missile, paint], dtype=np.float32
        )

