from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..actions import ACTION_DIM, ActionIndex
from ..constants import PACK_SIZE
from .los import has_los, raycast_voxels
from .mech import MechState
from .projectile import Projectile
from .world import VoxelWorld


@dataclass(frozen=True)
class WeaponSpec:
    name: str
    range_vox: float
    damage: float
    stability_damage: float
    heat: float
    cooldown_s: float
    arc_deg: float
    speed_vox: float = 0.0 # 0 for hitscan
    guidance: str = "none" # "homing", "ballistic", "linear"
    splash_rad_vox: float = 0.0
    splash_dmg_scale: float = 0.0


LASER = WeaponSpec(
    name="laser",
    range_vox=8.0,
    damage=14.0,
    stability_damage=0.0,
    heat=22.0,
    cooldown_s=0.6,
    arc_deg=120.0,
)

MISSILE = WeaponSpec(
    name="missile",
    range_vox=35.0,
    damage=40.0,
    stability_damage=15.0,
    heat=45.0,
    cooldown_s=3.0,
    arc_deg=180.0,
    speed_vox=12.0,
    guidance="homing",
    splash_rad_vox=2.5,
    splash_dmg_scale=0.5,
)

GAUSS = WeaponSpec(
    name="gauss",
    range_vox=60.0,
    damage=50.0,
    stability_damage=60.0, # Huge impact
    heat=15.0,
    cooldown_s=4.0,
    arc_deg=60.0,
    speed_vox=40.0, # Very fast
    guidance="ballistic",
    splash_rad_vox=1.5,
    splash_dmg_scale=0.5,
)

AUTOCANNON = WeaponSpec(
    name="autocannon",
    range_vox=20.0,
    damage=8.0,
    stability_damage=5.0,
    heat=6.0,
    cooldown_s=0.2, # Rapid fire
    arc_deg=90.0,
    speed_vox=25.0,
    guidance="linear",
    splash_rad_vox=0.0,
)

PAINTER = WeaponSpec(
    name="painter",
    range_vox=15.0,
    damage=0.0,
    stability_damage=0.0,
    heat=5.0,
    cooldown_s=1.0,
    arc_deg=60.0,
)


def _wrap_pi(angle: float) -> float:
    # Wrap to (-pi, pi]
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    return a


def _pack_index(mech: MechState) -> int | None:
    _, sep, suffix = mech.mech_id.rpartition("_")
    if not sep:
        return None
    try:
        idx = int(suffix)
    except ValueError:
        return None
    return int(idx) // int(PACK_SIZE)


def _same_pack(a: MechState, b: MechState) -> bool:
    if a.team != b.team:
        return False
    pa = _pack_index(a)
    pb = _pack_index(b)
    if pa is None or pb is None:
        return False
    return pa == pb


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
            # No control while shutdown, but physics should continue (gravity + damping).
            g_vox = 9.81 / float(self.world.voxel_size_m)
            mech.vel[2] = float(mech.vel[2] - g_vox * self.dt)
            mech.vel *= 0.98
            return

        forward, right = _yaw_to_forward_right(mech.yaw)
        forward_throttle = float(np.clip(action[ActionIndex.FORWARD], -1.0, 1.0))
        strafe_throttle = float(np.clip(action[ActionIndex.STRAFE], -1.0, 1.0))
        vertical_throttle = float(np.clip(action[ActionIndex.VERTICAL], -1.0, 1.0))

        # Cap planar speed: avoid sqrt(2) diagonal boost.
        xy_norm = math.hypot(forward_throttle, strafe_throttle)
        if xy_norm > 1.0:
            forward_throttle /= xy_norm
            strafe_throttle /= xy_norm

        speed_scale = 1.0
        accel_scale = 1.0
        if mech.is_legged:
            speed_scale = 0.4
            accel_scale = 0.5

        desired = (forward * forward_throttle + right * strafe_throttle) * float(mech.spec.max_speed * speed_scale)
        mech.vel[0] = float(desired[0])
        mech.vel[1] = float(desired[1])

        # Vertical is acceleration-based (jump jets) with gravity; mechs tend to rest on the ground.
        g_vox = 9.81 / float(self.world.voxel_size_m)
        jet_acc = vertical_throttle * float(mech.spec.max_jet_accel * accel_scale)
        mech.vel[2] = float(mech.vel[2] + (jet_acc - g_vox) * self.dt)

        # Small damping to keep velocities tame.
        mech.vel *= 0.98

    def _apply_rotation(self, mech: MechState, action: np.ndarray) -> None:
        if mech.shutdown:
            return
        yaw_rate = float(np.clip(action[ActionIndex.YAW_RATE], -1.0, 1.0)) * float(mech.spec.max_yaw_rate)
        mech.yaw = _wrap_pi(mech.yaw + yaw_rate * self.dt)

    def _apply_vent(self, mech: MechState, action: np.ndarray) -> None:
        vent = float(action[ActionIndex.VENT]) > 0.0
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

    def _spawn_debris(self, mech: MechState) -> None:
        hs = mech.half_size
        min_x = int(round(mech.pos[0] - hs[0]))
        max_x = int(round(mech.pos[0] + hs[0]))
        min_y = int(round(mech.pos[1] - hs[1]))
        max_y = int(round(mech.pos[1] + hs[1]))
        min_z = int(round(mech.pos[2] - hs[2]))
        max_z = int(round(mech.pos[2] + hs[2]))
        self.world.set_box_solid(min_x, min_y, min_z, max_x, max_y, max_z, True)

    def _get_damage_multiplier(self, target: MechState, origin: np.ndarray) -> tuple[float, bool]:
        # Vector from origin to target
        attack_vec = target.pos[:2] - origin[:2]
        dist = np.linalg.norm(attack_vec)
        if dist < 1e-6: return 1.0, False
        attack_dir = attack_vec / dist
        
        # Target facing
        forward, _ = _yaw_to_forward_right(target.yaw)
        forward_2d = forward[:2]
        
        # Dot product
        # 1.0 = attack traveling exactly with facing (Rear Hit)
        # -1.0 = attack traveling opposite to facing (Front Hit)
        dot = float(np.dot(forward_2d, attack_dir))
        
        if dot > 0.707: # 45 degree rear cone
            return 1.5, True # 1.5x damage
        return 1.0, False

    def _handle_death(self, target: MechState, shooter_id: str) -> list[dict]:
        events = []
        if target.hp <= 0.0 and target.alive:
            target.alive = False
            target.died = True
            
            shooter = self.mechs.get(shooter_id)
            if shooter:
                shooter.kills += 1
            
            events.append({"type": "kill", "shooter": shooter_id, "target": target.mech_id})
            self._spawn_debris(target)
        return events

    def _get_paint_bonus(self, target: MechState, raw_damage: float, shooter_id: str) -> tuple[float, list[dict]]:
        events = []
        if target.painted_remaining > 0.0 and target.last_painter_id:
            painter = self.mechs.get(target.last_painter_id)
            shooter = self.mechs.get(shooter_id)
            if painter is not None and shooter is not None and _same_pack(painter, shooter):
                bonus = raw_damage * 0.10 # 10% bonus
                if target.last_painter_id != shooter_id:
                    events.append({
                        "type": "assist",
                        "painter": target.last_painter_id,
                        "shooter": shooter_id,
                        "damage": bonus
                    })
                return raw_damage + bonus, events
        return raw_damage, events

    def _try_fire_laser(self, shooter: MechState, action: np.ndarray) -> list[dict]:
        fire = float(action[ActionIndex.FIRE_LASER]) > 0.0
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

        mult, is_crit = self._get_damage_multiplier(target, shooter.pos)
        base_dmg = LASER.damage * mult
        
        # Leg Hit Check (Approximate for Hitscan)
        is_leg_hit = False
        if self.rng.random() < 0.2:
            is_leg_hit = True
            target.leg_hp -= base_dmg
            base_dmg *= 0.5

        final_dmg, bonus_events = self._get_paint_bonus(target, base_dmg, shooter.mech_id)

        target.hp -= final_dmg
        target.was_hit = True
        shooter.dealt_damage += final_dmg
        target.took_damage += final_dmg

        events: list[dict] = [
            {
                "type": "laser_hit",
                "shooter": shooter.mech_id,
                "target": target.mech_id,
                "damage": final_dmg,
                "is_crit": is_crit,
                "is_leg_hit": is_leg_hit,
                "is_painted": target.painted_remaining > 0.0 # For visualization
            }
        ]
        events.extend(bonus_events)
        events.extend(self._handle_death(target, shooter.mech_id))

        return events

    def _try_fire_missile(self, shooter: MechState, action: np.ndarray) -> list[dict]:
        if len(action) < ACTION_DIM:
            return []
        fire = float(action[ActionIndex.FIRE_MISSILE]) > 0.0
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
            yaw_delta = _angle_between_yaw(shooter.yaw, delta[:2])
            if yaw_delta > math.radians(MISSILE.arc_deg * 0.5):
                continue

            has_lock = False
            # Primary lock: direct LOS.
            if has_los(self.world, shooter.pos, target.pos):
                has_lock = True
            # Secondary lock: painted target via pack network (indirect).
            elif target.painted_remaining > 0.0 and target.last_painter_id:
                painter = self.mechs.get(target.last_painter_id)
                if painter is not None and _same_pack(painter, shooter):
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
        speed = float(MISSILE.speed_vox)
        vel *= speed

        proj = Projectile(
            shooter_id=shooter.mech_id,
            target_id=target.mech_id,
            pos=shooter.pos.copy() + np.array([0, 0, shooter.half_size[2] + 0.5], dtype=np.float32),
            vel=vel,
            speed=speed,
            damage=MISSILE.damage,
            stability_damage=MISSILE.stability_damage,
            max_lifetime=float(MISSILE.range_vox / max(1e-6, speed)),
            guidance=MISSILE.guidance,
            splash_rad=MISSILE.splash_rad_vox,
            splash_scale=MISSILE.splash_dmg_scale,
        )
        self.projectiles.append(proj)

        return [{"type": "missile_launch", "shooter": shooter.mech_id, "target": target.mech_id}]

    def _try_paint(self, shooter: MechState, action: np.ndarray) -> list[dict]:
        if len(action) < ACTION_DIM:
            return []
        paint = float(action[ActionIndex.PAINT]) > 0.0
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
        target.last_painter_id = shooter.mech_id

        return [{"type": "paint", "shooter": shooter.mech_id, "target": target.mech_id}]

    def _try_fire_kinetic(self, shooter: MechState, action: np.ndarray) -> list[dict]:
        if len(action) < ACTION_DIM:
            return []
        
        spec = None
        if shooter.spec.name == "heavy":
            spec = GAUSS
        elif shooter.spec.name == "medium":
            spec = AUTOCANNON
        
        if spec is None:
            return []

        fire = float(action[ActionIndex.FIRE_KINETIC]) > 0.0
        if not fire or shooter.shutdown or not shooter.alive or shooter.fallen_time > 0:
            return []
        if shooter.kinetic_cooldown > 0.0:
            return []

        # Kinetic is dumb-fire / slight assist.
        # It fires in the direction of the mech's yaw + some vertical pitch?
        # For RL simplicity, we'll assume it auto-aims pitch if a target is roughly centered,
        # otherwise it fires straight.
        
        # Simple implementation: Fire straight ahead of yaw
        forward, _ = _yaw_to_forward_right(shooter.yaw)
        
        # Check if there is a target roughly in crosshair to adjust pitch?
        # For now, fire flat (z=0) unless we add pitch control action.
        # Gauss has drop, so firing flat means it hits ground.
        # Let's add a simple "auto-pitch" if an enemy is in LOS and yaw-aligned.
        target_pos = None
        best_dist = spec.range_vox
        
        for t in self.mechs.values():
            if t.team != shooter.team and t.alive:
                delta = t.pos - shooter.pos
                dist = np.linalg.norm(delta)
                if dist < best_dist:
                    yaw_err = _angle_between_yaw(shooter.yaw, delta[:2])
                    if yaw_err < 0.1: # Tight cone
                        target_pos = t.pos
                        best_dist = dist
        
        start_pos = shooter.pos.copy() + np.array([0, 0, shooter.half_size[2]*0.8], dtype=np.float32)
        
        if target_pos is not None:
            delta = target_pos - start_pos
            dist = float(np.linalg.norm(delta))
            if dist > 1e-6 and spec.guidance == "ballistic":
                # Aim slightly up to compensate gravity partially.
                flight_time = dist / max(1e-6, spec.speed_vox)
                g = 9.81 / self.world.voxel_size_m
                drop = 0.5 * g * (flight_time**2)
                delta = delta.copy()
                delta[2] += drop
                vel = delta / (np.linalg.norm(delta) + 1e-6) * spec.speed_vox
            elif dist > 1e-6:
                # Auto-pitch for direct-fire ballistics (e.g., autocannon).
                vel = delta / dist * spec.speed_vox
            else:
                vel = forward * spec.speed_vox
        else:
            # Fire flat
            vel = forward * spec.speed_vox
        
        proj = Projectile(
            shooter_id=shooter.mech_id,
            target_id=None, # Dumbfire
            pos=start_pos,
            vel=vel,
            speed=spec.speed_vox,
            damage=spec.damage,
            stability_damage=spec.stability_damage,
            max_lifetime=spec.range_vox / spec.speed_vox,
            guidance=spec.guidance,
            splash_rad=spec.splash_rad_vox,
            splash_scale=spec.splash_dmg_scale,
        )
        self.projectiles.append(proj)
        
        shooter.kinetic_cooldown = spec.cooldown_s
        shooter.heat += spec.heat

        return [{"type": "kinetic_fire", "shooter": shooter.mech_id, "weapon": spec.name}]

    def _explode(self, pos: np.ndarray, proj: Projectile, exclude_mech: MechState | None = None) -> list[dict]:
        events = []
        # Check all mechs for splash
        for m in self.mechs.values():
            if not m.alive: continue
            if exclude_mech and m is exclude_mech: continue
            
            dist = np.linalg.norm(m.pos - pos)
            
            if dist <= proj.splash_rad:
                # Apply Splash
                mult = 1.0
                is_crit = False
                
                raw_dmg = proj.damage * proj.splash_scale * mult
                raw_stab = proj.stability_damage * proj.splash_scale * mult
                
                # Distance falloff? Let's keep it simple flat splash for now.
                
                m.hp -= raw_dmg
                m.stability = max(0.0, m.stability - raw_stab)
                m.was_hit = True
                m.took_damage += raw_dmg
                
                shooter = self.mechs.get(proj.shooter_id)
                if shooter:
                    shooter.dealt_damage += raw_dmg
                    
                    events.append({
                        "type": "projectile_hit",
                        "shooter": proj.shooter_id,
                        "target": m.mech_id,
                        "damage": raw_dmg,
                        "stability": raw_stab,
                        "is_crit": is_crit
                    })
                    
                    events.extend(self._handle_death(m, proj.shooter_id))
        
        events.append({
            "type": "explosion",
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "radius": proj.splash_rad,
            "shooter": proj.shooter_id
        })
        return events

    def _update_projectiles(self) -> list[dict]:
        events = []
        keep = []
        g_vox = 9.81 / self.world.voxel_size_m

        for p in self.projectiles:
            if not p.alive:
                continue
            
            p.age += self.dt
            if p.age > p.max_lifetime:
                p.alive = False
                continue

            # Guidance & Physics
            if p.guidance == "homing":
                if p.target_id in self.mechs:
                    target = self.mechs[p.target_id]
                    if target.alive:
                        delta = target.pos - p.pos
                        dist = np.linalg.norm(delta)
                        if dist > 0.1:
                            desired = (delta / dist) * p.speed
                            steer = 4.0 * self.dt
                            p.vel = p.vel * (1.0 - steer) + desired * steer
                            p.vel = (p.vel / (np.linalg.norm(p.vel) + 1e-6)) * p.speed
            elif p.guidance == "ballistic":
                # Gravity
                p.vel[2] -= g_vox * self.dt

            # Integration
            step = p.vel * self.dt
            next_pos = p.pos + step
            
            impact = False
            impact_pos = next_pos
            direct_hit_mech = None
            
            # Fast projectiles can tunnel through 1-voxel terrain if we only check the end-point.
            # Use swept collision against both world voxels and mech bounds.
            step_len2 = float(np.dot(step, step))

            best_t: float | None = None
            best_target: MechState | None = None
            best_pos: np.ndarray | None = None

            if step_len2 > 1e-12:
                for target in self.mechs.values():
                    if not target.alive:
                        continue
                    if target.mech_id == p.shooter_id and p.age < 0.2:
                        continue

                    # Approximate mech as a sphere for continuous collision.
                    radius = float(np.max(target.half_size))
                    to_center = target.pos - p.pos
                    t = float(np.dot(to_center, step) / step_len2)
                    t = float(np.clip(t, 0.0, 1.0))
                    closest = p.pos + step * t
                    delta = target.pos - closest
                    if float(np.dot(delta, delta)) <= radius * radius:
                        if best_t is None or t < best_t:
                            best_t = t
                            best_target = target
                            best_pos = closest

            if best_target is not None and best_pos is not None:
                # Ensure terrain does not block before the mech impact point.
                hit = raycast_voxels(self.world, p.pos, best_pos, include_end=True)
                if hit.blocked:
                    impact = True
                    p.alive = False
                    if hit.blocked_voxel is not None:
                        bx, by, bz = hit.blocked_voxel
                        impact_pos = np.asarray([bx + 0.5, by + 0.5, bz + 0.5], dtype=np.float32)
                else:
                    impact = True
                    p.alive = False
                    direct_hit_mech = best_target
                    impact_pos = best_pos
            else:
                hit = raycast_voxels(self.world, p.pos, next_pos, include_end=True)
                if hit.blocked or self.world.aabb_collides(next_pos - 0.1, next_pos + 0.1):
                    impact = True
                    p.alive = False
                    if hit.blocked and hit.blocked_voxel is not None:
                        bx, by, bz = hit.blocked_voxel
                        impact_pos = np.asarray([bx + 0.5, by + 0.5, bz + 0.5], dtype=np.float32)
            
            if impact:
                # 1. Apply Direct Hit Logic (if mech)
                if direct_hit_mech:
                    m = direct_hit_mech
                    
                    # Calculate trajectory origin for hit calc? 
                    # Use projectile velocity to infer 'origin' direction?
                    # Attack vector is p.vel (direction of impact).
                    # But _get_damage_multiplier expects origin point.
                    # We can fake origin: m.pos - p.vel
                    fake_origin = m.pos - p.vel
                    mult, is_crit = self._get_damage_multiplier(m, fake_origin)
                    
                    base_dmg = p.damage * mult
                    
                    is_leg_hit = False
                    min_z = float(m.pos[2] - m.half_size[2])
                    height = float(m.half_size[2] * 2.0)
                    if float(impact_pos[2]) < min_z + height * 0.3:
                        is_leg_hit = True
                        m.leg_hp -= base_dmg
                        base_dmg *= 0.5

                    final_dmg, bonus_events = self._get_paint_bonus(m, base_dmg, p.shooter_id)
                    final_stab = p.stability_damage * mult
                    
                    m.hp -= final_dmg
                    m.stability = max(0.0, m.stability - final_stab)
                    m.was_hit = True
                    m.took_damage += final_dmg
                    
                    shooter = self.mechs.get(p.shooter_id)
                    if shooter:
                        shooter.dealt_damage += final_dmg
                        events.append({
                            "type": "projectile_hit",
                            "shooter": shooter.mech_id,
                            "target": m.mech_id,
                            "damage": final_dmg,
                            "stability": final_stab,
                            "is_crit": is_crit,
                            "is_leg_hit": is_leg_hit,
                            "is_painted": m.painted_remaining > 0.0
                        })
                        events.extend(bonus_events)
                        events.extend(self._handle_death(m, p.shooter_id))
                
                # 2. Apply Splash Logic (if splash > 0)
                if p.splash_rad > 0.0:
                    splash_pos = direct_hit_mech.pos if direct_hit_mech else impact_pos
                    events.extend(self._explode(splash_pos, p, exclude_mech=direct_hit_mech))

            else:
                p.pos = next_pos
                keep.append(p)
        
        self.projectiles = keep
        return events

    def step(self, actions: dict[str, np.ndarray], num_substeps: int) -> list[dict]:
        """
        Advance the simulation by num_substeps of dt_sim, holding actions constant.

        Returns a list of emitted events (hits/kills/etc.) aggregated over the substeps.
        """
        events: list[dict] = []
        zero_action = np.zeros(ACTION_DIM, dtype=np.float32)
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
                mech.kinetic_cooldown = max(0.0, float(mech.kinetic_cooldown - self.dt))
                mech.painter_cooldown = max(0.0, float(mech.painter_cooldown - self.dt))
                mech.painted_remaining = max(0.0, float(mech.painted_remaining - self.dt))
                
                # Stability Logic
                if mech.fallen_time > 0.0:
                    # Stunned/Fallen
                    mech.fallen_time -= self.dt
                    if mech.fallen_time <= 0.0:
                        # Stand up
                        mech.fallen_time = 0.0
                        mech.stability = mech.max_stability * 0.5 # Recover with half stability
                else:
                    if mech.stability <= 0.0:
                        # Fall down
                        mech.fallen_time = 3.0 # 3 seconds stun
                        mech.vel[:] = 0.0
                    else:
                        # Regen
                        regen = 10.0 * self.dt
                        max_stab = mech.max_stability
                        
                        if mech.is_legged:
                            regen *= 0.5
                            max_stab *= 0.5
                            
                        if np.linalg.norm(mech.vel) > 1.0:
                            regen *= 0.5 # Slower while moving
                        
                        mech.stability = min(max_stab, mech.stability + regen)

                self._apply_vent(mech, actions.get(mech.mech_id, zero_action))
                self._dissipate(mech)
            
            # Projectiles
            events.extend(self._update_projectiles())

            # Fire before movement (classic "simultaneous" feel; order doesn't matter for hitscan).
            for mech in self.mechs.values():
                if not mech.alive:
                    continue
                a = actions.get(mech.mech_id, zero_action)
                events.extend(self._try_fire_laser(mech, a))
                events.extend(self._try_fire_missile(mech, a))
                events.extend(self._try_fire_kinetic(mech, a))
                events.extend(self._try_paint(mech, a))

            # Move/rotate in randomized order to reduce systematic bias.
            order = [m for m in self.mechs.values() if m.alive]
            self.rng.shuffle(order)
            for mech in order:
                if mech.fallen_time > 0.0: 
                    # Fallen mechs cannot move/rotate. 
                    # Apply gravity/damping only (no horizontal control, no rotation).
                    self._apply_movement(mech, zero_action)
                    self._integrate(mech)
                    continue

                a = actions.get(mech.mech_id, zero_action)
                self._apply_rotation(mech, a)
                self._apply_movement(mech, a)
                self._integrate(mech)

        return events
