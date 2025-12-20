from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .los import has_los
from .mech import MechState
from .projectile import Projectile
from .world import VoxelWorld


@dataclass(frozen=True)
class WeaponSpec:
    name: str
    range_vox: float
    damage: float
    heat: float
    cooldown_s: float
    arc_deg: float


LASER = WeaponSpec(
    name="laser",
    range_vox=8.0,
    damage=14.0,
    heat=22.0,
    cooldown_s=0.6,
    arc_deg=120.0,
)

MISSILE = WeaponSpec(
    name="missile",
    range_vox=25.0,  # Long range
    damage=40.0,
    heat=45.0,
    cooldown_s=3.0,
    arc_deg=180.0,   # Wide lock angle
)

PAINTER = WeaponSpec(
    name="painter",
    range_vox=15.0,
    damage=0.0,
    heat=5.0,
    cooldown_s=1.0,
    arc_deg=60.0,
)


def _wrap_pi(angle: float) -> float:
    # Wrap to (-pi, pi]
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    return a


def _yaw_to_forward_right(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    forward = np.asarray([c, s, 0.0], dtype=np.float32)
    right = np.asarray([-s, c, 0.0], dtype=np.float32)
    return forward, right


def _angle_between_yaw(yaw: float, dir_xy: np.ndarray) -> float:
    # Returns absolute yaw delta in radians between facing and direction (XY plane).
    forward, _ = _yaw_to_forward_right(yaw)
    v = np.asarray([dir_xy[0], dir_xy[1], 0.0], dtype=np.float32)
    n = float(np.linalg.norm(v[:2]))
    if n <= 1e-6:
        return 0.0
    v[:2] /= n
    dot = float(np.clip(forward[0] * v[0] + forward[1] * v[1], -1.0, 1.0))
    return float(math.acos(dot))


class Sim:
    def __init__(self, world: VoxelWorld, dt_sim: float, rng: np.random.Generator):
        self.world = world
        self.dt = float(dt_sim)
        self.rng = rng
        self.time_s = 0.0
        self.tick = 0
        self.mechs: dict[str, MechState] = {}
        self.projectiles: list[Projectile] = []

    def reset(self, mechs: dict[str, MechState]) -> None:
        self.time_s = 0.0
        self.tick = 0
        self.mechs = mechs
        self.projectiles = []

    def living_mechs(self) -> list[MechState]:
        return [m for m in self.mechs.values() if m.alive]

    def team_alive(self, team: str) -> bool:
        return any(m.alive and m.team == team for m in self.mechs.values())

    def _collides_world(self, mech: MechState, pos: np.ndarray) -> bool:
        hs = mech.half_size
        aabb_min = pos - hs
        aabb_max = pos + hs
        return self.world.aabb_collides(aabb_min, aabb_max)

    def _collides_mechs(self, mech: MechState, pos: np.ndarray) -> bool:
        hs = mech.half_size
        a_min = pos - hs
        a_max = pos + hs
        for other in self.mechs.values():
            if other.mech_id == mech.mech_id or not other.alive:
                continue
            o_hs = other.half_size
            b_min = other.pos - o_hs
            b_max = other.pos + o_hs
            if (
                a_min[0] < b_max[0]
                and a_max[0] > b_min[0]
                and a_min[1] < b_max[1]
                and a_max[1] > b_min[1]
                and a_min[2] < b_max[2]
                and a_max[2] > b_min[2]
            ):
                return True
        return False

    def _collides_any(self, mech: MechState, pos: np.ndarray) -> bool:
        if self._collides_world(mech, pos):
            return True
        if self._collides_mechs(mech, pos):
            return True
        return False

    def _apply_movement(self, mech: MechState, action: np.ndarray) -> None:
        if mech.shutdown:
            mech.vel[:] = 0.0
            return

        forward, right = _yaw_to_forward_right(mech.yaw)
        forward_throttle = float(np.clip(action[0], -1.0, 1.0))
        strafe_throttle = float(np.clip(action[1], -1.0, 1.0))
        vertical_throttle = float(np.clip(action[2], -1.0, 1.0))

        # Cap planar speed: avoid sqrt(2) diagonal boost.
        xy_norm = math.hypot(forward_throttle, strafe_throttle)
        if xy_norm > 1.0:
            forward_throttle /= xy_norm
            strafe_throttle /= xy_norm

        desired = (forward * forward_throttle + right * strafe_throttle) * float(mech.spec.max_speed)
        mech.vel[0] = float(desired[0])
        mech.vel[1] = float(desired[1])

        # Vertical is acceleration-based (jump jets) with gravity; mechs tend to rest on the ground.
        g_vox = 9.81 / float(self.world.voxel_size_m)
        jet_acc = vertical_throttle * float(mech.spec.max_jet_accel)
        mech.vel[2] = float(mech.vel[2] + (jet_acc - g_vox) * self.dt)

        # Small damping to keep velocities tame.
        mech.vel *= 0.98

    def _apply_rotation(self, mech: MechState, action: np.ndarray) -> None:
        if mech.shutdown:
            return
        yaw_rate = float(np.clip(action[3], -1.0, 1.0)) * float(mech.spec.max_yaw_rate)
        mech.yaw = _wrap_pi(mech.yaw + yaw_rate * self.dt)

    def _apply_vent(self, mech: MechState, action: np.ndarray) -> None:
        vent = float(action[5]) > 0.0
        if vent:
            mech.heat = max(0.0, float(mech.heat - 2.0 * mech.spec.heat_dissipation * self.dt))

    def _dissipate(self, mech: MechState) -> None:
        mech.heat = max(0.0, float(mech.heat - mech.spec.heat_dissipation * self.dt))

    def _integrate(self, mech: MechState) -> None:
        pos = mech.pos.astype(np.float32, copy=True)
        vel = mech.vel.astype(np.float32, copy=False)

        # Axis-by-axis collision resolution (cheap, stable).
        trial = pos.copy()
        trial[0] = pos[0] + vel[0] * self.dt
        if not self._collides_any(mech, trial):
            pos[0] = trial[0]
        else:
            mech.vel[0] = 0.0

        trial = pos.copy()
        trial[1] = pos[1] + vel[1] * self.dt
        if not self._collides_any(mech, trial):
            pos[1] = trial[1]
        else:
            mech.vel[1] = 0.0

        trial = pos.copy()
        trial[2] = pos[2] + vel[2] * self.dt
        if not self._collides_any(mech, trial):
            pos[2] = trial[2]
        else:
            mech.vel[2] = 0.0

        # Ground plane at z = half-height.
        min_z = float(mech.half_size[2])
        if pos[2] < min_z:
            pos[2] = min_z
            mech.vel[2] = 0.0

        mech.pos[:] = pos

    def _try_fire_laser(self, shooter: MechState, action: np.ndarray) -> list[dict]:
        fire = float(action[4]) > 0.0
        if not fire or shooter.shutdown or not shooter.alive:
            return []
        if shooter.laser_cooldown > 0.0:
            return []

        best: tuple[float, MechState] | None = None
        for target in self.mechs.values():
            if not target.alive or target.team == shooter.team:
                continue
            delta = target.pos - shooter.pos
            dist = float(np.linalg.norm(delta))
            if dist > LASER.range_vox:
                continue
            # Simple yaw arc gate in XY.
            yaw_delta = _angle_between_yaw(shooter.yaw, delta[:2])
            if yaw_delta > math.radians(LASER.arc_deg * 0.5):
                continue
            if not has_los(self.world, shooter.pos, target.pos):
                continue
            if best is None or dist < best[0]:
                best = (dist, target)

        if best is None:
            return []

        target = best[1]
        shooter.laser_cooldown = LASER.cooldown_s
        shooter.heat = float(shooter.heat + LASER.heat)

        target.hp -= LASER.damage
        target.was_hit = True
        shooter.dealt_damage += LASER.damage
        target.took_damage += LASER.damage

        events: list[dict] = [
            {
                "type": "laser_hit",
                "shooter": shooter.mech_id,
                "target": target.mech_id,
                "damage": LASER.damage,
            }
        ]

        if target.hp <= 0.0 and target.alive:
            target.alive = False
            target.died = True
            shooter.kills += 1
            events.append({"type": "kill", "shooter": shooter.mech_id, "target": target.mech_id})

        return events

    def _try_fire_missile(self, shooter: MechState, action: np.ndarray) -> list[dict]:
        # Missile action is index 6 (mapped later)
        if len(action) < 8:
            return []
        fire = float(action[6]) > 0.0
        # Only heavies have missiles
        if shooter.spec.name != "heavy":
            return []
        if not fire or shooter.shutdown or not shooter.alive:
            return []
        if shooter.missile_cooldown > 0.0:
            return []

        best: tuple[float, MechState] | None = None
        for target in self.mechs.values():
            if not target.alive or target.team == shooter.team:
                continue
            delta = target.pos - shooter.pos
            dist = float(np.linalg.norm(delta))
            if dist > MISSILE.range_vox:
                continue
            
            # Lock check: Needs LOS OR Painted
            has_lock = False
            if target.painted_remaining > 0.0:
                has_lock = True
            else:
                # Fallback to LOS check within arc
                yaw_delta = _angle_between_yaw(shooter.yaw, delta[:2])
                if yaw_delta <= math.radians(MISSILE.arc_deg * 0.5):
                    if has_los(self.world, shooter.pos, target.pos):
                        has_lock = True
            
            if not has_lock:
                continue

            if best is None or dist < best[0]:
                best = (dist, target)

        if best is None:
            return []

        target = best[1]
        shooter.missile_cooldown = MISSILE.cooldown_s
        shooter.heat = float(shooter.heat + MISSILE.heat)

        # Spawn projectile
        # Initial velocity: Up and towards target to simulate arc launch
        delta = target.pos - shooter.pos
        delta_norm = delta / (np.linalg.norm(delta) + 1e-6)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        vel = (delta_norm + up * 0.5)
        vel /= np.linalg.norm(vel)
        speed = 12.0 # vox/s
        vel *= speed

        proj = Projectile(
            shooter_id=shooter.mech_id,
            target_id=target.mech_id,
            pos=shooter.pos.copy() + np.array([0, 0, shooter.half_size[2] + 0.5], dtype=np.float32),
            vel=vel,
            speed=speed,
            damage=MISSILE.damage,
            max_lifetime=5.0,
        )
        self.projectiles.append(proj)

        return [{"type": "missile_launch", "shooter": shooter.mech_id, "target": target.mech_id}]

    def _try_paint(self, shooter: MechState, action: np.ndarray) -> list[dict]:
        # Paint action is index 7
        if len(action) < 8:
            return []
        paint = float(action[7]) > 0.0
        # Only lights have painter
        if shooter.spec.name != "light":
            return []
        if not paint or shooter.shutdown or not shooter.alive:
            return []
        if shooter.painter_cooldown > 0.0:
            return []

        best: tuple[float, MechState] | None = None
        for target in self.mechs.values():
            if not target.alive or target.team == shooter.team:
                continue
            delta = target.pos - shooter.pos
            dist = float(np.linalg.norm(delta))
            if dist > PAINTER.range_vox:
                continue
            
            yaw_delta = _angle_between_yaw(shooter.yaw, delta[:2])
            if yaw_delta > math.radians(PAINTER.arc_deg * 0.5):
                continue
            if not has_los(self.world, shooter.pos, target.pos):
                continue

            if best is None or dist < best[0]:
                best = (dist, target)
        
        if best is None:
            return []
        
        target = best[1]
        shooter.painter_cooldown = PAINTER.cooldown_s
        shooter.heat = float(shooter.heat + PAINTER.heat)
        target.painted_remaining = 5.0 # 5 seconds of paint

        return [{"type": "paint", "shooter": shooter.mech_id, "target": target.mech_id}]

    def _update_projectiles(self) -> list[dict]:
        events = []
        keep = []
        for p in self.projectiles:
            if not p.alive:
                continue
            
            p.age += self.dt
            if p.age > p.max_lifetime:
                p.alive = False
                continue

            # Homing Logic
            if p.target_id in self.mechs:
                target = self.mechs[p.target_id]
                if target.alive:
                    delta = target.pos - p.pos
                    dist = np.linalg.norm(delta)
                    if dist > 0.1:
                        # Steer velocity towards target
                        desired = (delta / dist) * p.speed
                        steer_factor = 4.0 * self.dt # Turning capability
                        p.vel = p.vel * (1.0 - steer_factor) + desired * steer_factor
                        # Renormalize
                        p.vel = (p.vel / (np.linalg.norm(p.vel) + 1e-6)) * p.speed

            # Integrate
            step = p.vel * self.dt
            # Collision check (simple raycast substitute: just endpoint check)
            # Ideally we'd sweep, but small steps + chunky voxels = usually fine.
            next_pos = p.pos + step
            
            # World collision
            if self.world.aabb_collides(next_pos - 0.1, next_pos + 0.1):
                p.alive = False
                # Miss
                continue

            # Mech collision (simplified: if close to target)
            hit = False
            if p.target_id in self.mechs:
                target = self.mechs[p.target_id]
                if target.alive:
                    dist = np.linalg.norm(target.pos - next_pos)
                    # Hit radius roughly size of mech
                    hit_rad = np.max(target.half_size)
                    if dist < hit_rad:
                        hit = True
                        p.alive = False
                        
                        target.hp -= p.damage
                        target.was_hit = True
                        target.took_damage += p.damage
                        
                        shooter = self.mechs.get(p.shooter_id)
                        if shooter:
                            shooter.dealt_damage += p.damage
                            events.append({
                                "type": "missile_hit",
                                "shooter": shooter.mech_id,
                                "target": target.mech_id,
                                "damage": p.damage
                            })
                            if target.hp <= 0.0 and target.alive:
                                target.alive = False
                                target.died = True
                                shooter.kills += 1
                                events.append({"type": "kill", "shooter": shooter.mech_id, "target": target.mech_id})
            
            p.pos = next_pos
            if p.alive:
                keep.append(p)
        
        self.projectiles = keep
        return events

    def step(self, actions: dict[str, np.ndarray], num_substeps: int) -> list[dict]:
        """
        Advance the simulation by num_substeps of dt_sim, holding actions constant.

        Returns a list of emitted events (hits/kills/etc.) aggregated over the substeps.
        """
        events: list[dict] = []
        for mech in self.mechs.values():
            mech.reset_step_stats()

        for _ in range(int(num_substeps)):
            self.tick += 1
            self.time_s += self.dt

            # Update cooldowns and dissipation.
            for mech in self.mechs.values():
                if not mech.alive:
                    continue
                mech.laser_cooldown = max(0.0, float(mech.laser_cooldown - self.dt))
                mech.missile_cooldown = max(0.0, float(mech.missile_cooldown - self.dt))
                mech.painter_cooldown = max(0.0, float(mech.painter_cooldown - self.dt))
                mech.painted_remaining = max(0.0, float(mech.painted_remaining - self.dt))
                
                self._apply_vent(mech, actions.get(mech.mech_id, np.zeros(6, dtype=np.float32)))
                self._dissipate(mech)
            
            # Projectiles
            events.extend(self._update_projectiles())

            # Fire before movement (classic "simultaneous" feel; order doesn't matter for hitscan).
            for mech in self.mechs.values():
                if not mech.alive:
                    continue
                a = actions.get(mech.mech_id, np.zeros(8, dtype=np.float32)) # Expanded to 8
                events.extend(self._try_fire_laser(mech, a))
                events.extend(self._try_fire_missile(mech, a))
                events.extend(self._try_paint(mech, a))

            # Move/rotate in randomized order to reduce systematic bias.
            order = [m for m in self.mechs.values() if m.alive]
            self.rng.shuffle(order)
            for mech in order:
                a = actions.get(mech.mech_id, np.zeros(6, dtype=np.float32))
                self._apply_rotation(mech, a)
                self._apply_movement(mech, a)
                self._integrate(mech)

        return events
