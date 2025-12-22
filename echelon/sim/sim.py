from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..actions import ACTION_DIM, ActionIndex
from ..constants import PACK_SIZE
from ..config import (
    AMS_COOLDOWN_S,
    AMS_INTERCEPT_PROB,
    AMS_RANGE_VOX,
    AUTOCANNON,
    ECCM_HEAT_PER_S,
    ECCM_RADIUS_VOX,
    ECCM_WEIGHT,
    ECM_HEAT_PER_S,
    ECM_RADIUS_VOX,
    ECM_WEIGHT,
    FLAME_HEAT_TRANSFER,
    FLAMER,
    GAUSS,
    LASER,
    LASER_HEAT_TRANSFER,
    LAVA_DMG_PER_S,
    LAVA_HEAT_PER_S,
    MISSILE,
    PAINT_LOCK_MIN_QUALITY,
    PAINTER,
    SENSOR_QUALITY_MAX,
    SENSOR_QUALITY_MIN,
    SMOKE,
    SUPPRESS_DURATION_S,
    SUPPRESS_REGEN_SCALE,
    WATER_COOLING_PER_S,
    WATER_SPEED_MULT,
)
from .los import has_los, raycast_voxels
from .mech import MechState
from .projectile import Projectile
from .world import VoxelWorld

# Lightweight "role" extensions (kept simple and readable in replays).

@dataclass
class SmokeCloud:
    pos: np.ndarray # float32[3]
    radius: float
    remaining_life: float
    alive: bool = True

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
        self.smoke_clouds: list[SmokeCloud] = []

    def reset(self, mechs: dict[str, MechState]) -> None:
        self.time_s = 0.0
        self.tick = 0
        self.mechs = mechs
        self.projectiles = []
        self.smoke_clouds = []

    def living_mechs(self) -> list[MechState]:
        return [m for m in self.mechs.values() if m.alive]

    def team_alive(self, team: str) -> bool:
        return any(m.alive and m.team == team for m in self.mechs.values())

    def has_los(self, start_xyz: np.ndarray, end_xyz: np.ndarray) -> bool:
        # Check world voxels
        if not has_los(self.world, start_xyz, end_xyz):
            return False
        
        # Check smoke clouds
        # A simple ray-sphere intersection check.
        p1 = start_xyz
        p2 = end_xyz
        d = p2 - p1
        mag2 = float(np.dot(d, d))
        if mag2 < 1e-9:
            return True
        
        for cloud in self.smoke_clouds:
            if not cloud.alive:
                continue
            
            # Distance from point to line segment
            # t = projection of (cloud.pos - p1) onto d
            v = cloud.pos - p1
            t = float(np.dot(v, d) / mag2)
            t = max(0.0, min(1.0, t))
            closest = p1 + t * d
            dist2 = float(np.dot(cloud.pos - closest, cloud.pos - closest))
            if dist2 <= cloud.radius * cloud.radius:
                return False
        
        return True

    def _ewar_levels(self, viewer: MechState) -> tuple[float, float]:
        # Returns (jam_level, eccm_level) in [0, 1], based on strongest nearby sources.
        if not viewer.alive:
            return 0.0, 0.0
        if viewer.shutdown:
            return 0.0, 0.0

        jam = 0.0
        eccm = 0.0
        for other in self.mechs.values():
            if not other.alive or other.mech_id == viewer.mech_id:
                continue
            if other.shutdown:
                continue

            d = other.pos - viewer.pos
            dist = float(np.linalg.norm(d))

            if other.team != viewer.team and other.ecm_on and dist <= ECM_RADIUS_VOX:
                jam = max(jam, max(0.0, 1.0 - dist / max(1e-6, ECM_RADIUS_VOX)))
            if other.team == viewer.team and other.eccm_on and dist <= ECCM_RADIUS_VOX:
                eccm = max(eccm, max(0.0, 1.0 - dist / max(1e-6, ECCM_RADIUS_VOX)))

        return float(jam), float(eccm)

    def _sensor_quality(self, viewer: MechState) -> float:
        jam, eccm = self._ewar_levels(viewer)
        q = 1.0 - ECM_WEIGHT * jam + ECCM_WEIGHT * eccm
        return float(np.clip(q, SENSOR_QUALITY_MIN, SENSOR_QUALITY_MAX))

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
            speed_scale *= 0.4
            accel_scale *= 0.5
        
        # Hazard: Water Slow
        ix, iy, iz = int(mech.pos[0]), int(mech.pos[1]), int(mech.pos[2] - mech.half_size[2])
        if self.world.get_voxel(ix, iy, iz) == VoxelWorld.WATER:
            speed_scale *= WATER_SPEED_MULT

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
        if mech.shutdown:
            return
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

        # Clamp to world boundaries.
        hs = mech.half_size
        pos[0] = float(np.clip(pos[0], hs[0], float(self.world.size_x) - hs[0]))
        pos[1] = float(np.clip(pos[1], hs[1], float(self.world.size_y) - hs[1]))
        pos[2] = float(max(hs[2], pos[2])) # Ground floor

        mech.pos[:] = pos

    def _spawn_debris(self, mech: MechState) -> None:
        hs = mech.half_size
        aabb_min = mech.pos - hs
        aabb_max = mech.pos + hs

        min_ix = int(math.floor(float(aabb_min[0])))
        min_iy = int(math.floor(float(aabb_min[1])))
        min_iz = int(math.floor(float(aabb_min[2])))
        max_ix = int(math.ceil(float(aabb_max[0])))
        max_iy = int(math.ceil(float(aabb_max[1])))
        max_iz = int(math.ceil(float(aabb_max[2])))

        min_ix = max(0, min_ix)
        min_iy = max(0, min_iy)
        min_iz = max(0, min_iz)
        max_ix = min(self.world.size_x, max_ix)
        max_iy = min(self.world.size_y, max_iy)
        max_iz = min(self.world.size_z, max_iz)

        # Probabilistic hull survival based on size
        # Heavy: High chance of solid hull. Scout: mostly debris.
        probs = {"scout": 0.1, "light": 0.3, "medium": 0.6, "heavy": 0.9}
        p_solid = probs.get(mech.spec.name, 0.5)
        
        # Roll once for the whole wreck or per-voxel? 
        # Whole wreck is cleaner tactically.
        is_solid_wreck = self.rng.random() < p_solid
        voxel_type = VoxelWorld.KILLED_HULL if is_solid_wreck else VoxelWorld.HOT_DEBRIS

        # Avoid embedding living mechs
        living_aabbs: list[tuple[np.ndarray, np.ndarray]] = []
        for other in self.mechs.values():
            if not other.alive or other.mech_id == mech.mech_id:
                continue
            o_hs = other.half_size
            living_aabbs.append((other.pos - o_hs, other.pos + o_hs))

        for iz in range(min_iz, max_iz):
            cell_min_z = float(iz)
            cell_max_z = float(iz + 1)
            for iy in range(min_iy, max_iy):
                cell_min_y = float(iy)
                cell_max_y = float(iy + 1)
                for ix in range(min_ix, max_ix):
                    vox = int(self.world.voxels[iz, iy, ix])
                    if vox not in (VoxelWorld.AIR, VoxelWorld.DIRT):
                        continue

                    cell_min_x = float(ix)
                    cell_max_x = float(ix + 1)

                    intersects = False
                    for other_min, other_max in living_aabbs:
                        if (
                            cell_min_x < float(other_max[0])
                            and cell_max_x > float(other_min[0])
                            and cell_min_y < float(other_max[1])
                            and cell_max_y > float(other_min[1])
                            and cell_min_z < float(other_max[2])
                            and cell_max_z > float(other_min[2])
                        ):
                            intersects = True
                            break
                    if intersects:
                        continue

                    self.world.voxels[iz, iy, ix] = voxel_type

    def _get_damage_multiplier(self, target: MechState, origin: np.ndarray) -> tuple[float, bool]:
        # Vector from origin to target
        attack_vec = target.pos[:2] - origin[:2]
        dist = np.linalg.norm(attack_vec)
        if dist < 1e-6:
            return 1.0, False
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

    def _apply_hazards(self, mech: MechState) -> list[dict]:
        if not mech.alive:
            return []
        
        events = []
        ix, iy, iz = int(mech.pos[0]), int(mech.pos[1]), int(mech.pos[2] - mech.half_size[2])
        vox = self.world.get_voxel(ix, iy, iz)
        
        if vox == VoxelWorld.LAVA:
            mech.heat = float(mech.heat + LAVA_HEAT_PER_S * self.dt)
            dmg = LAVA_DMG_PER_S * self.dt
            mech.hp -= dmg
            mech.took_damage += dmg
            events.extend(self._handle_death(mech, "lava"))
        elif vox == VoxelWorld.HOT_DEBRIS:
            # Hot debris is less lethal than lava but still hurts
            mech.heat = float(mech.heat + LAVA_HEAT_PER_S * 0.5 * self.dt)
            dmg = LAVA_DMG_PER_S * 0.25 * self.dt
            mech.hp -= dmg
            mech.took_damage += dmg
            events.extend(self._handle_death(mech, "debris"))
        elif vox == VoxelWorld.WATER:
            mech.heat = float(max(0.0, mech.heat - WATER_COOLING_PER_S * self.dt))
            
        return events

    def _try_fire_laser(self, shooter: MechState, action: np.ndarray) -> list[dict]:
        fire_laser = float(action[ActionIndex.FIRE_LASER]) > 0.0 and shooter.spec.name != "scout"
        fire_flame = float(action[ActionIndex.FIRE_KINETIC]) > 0.0 if shooter.spec.name == "light" else False
        
        if not (fire_laser or fire_flame) or shooter.shutdown or not shooter.alive:
            return []
        
        if fire_laser and shooter.laser_cooldown > 0.0:
            fire_laser = False
        if fire_flame and shooter.kinetic_cooldown > 0.0:
            fire_flame = False
            
        if not (fire_laser or fire_flame):
            return []

        # Which one to fire? If both requested, let's prioritize laser for now or just fire both?
        specs = []
        if fire_laser:
            specs.append(LASER)
        if fire_flame:
            specs.append(FLAMER)

        all_events = []
        for spec in specs:
            focus: MechState | None = None
            if shooter.focus_target_id:
                cand = self.mechs.get(shooter.focus_target_id)
                if cand is not None and cand.alive and cand.team != shooter.team:
                    focus = cand

            best: tuple[float, MechState] | None = None
            if focus is not None:
                delta = focus.pos - shooter.pos
                dist = float(np.linalg.norm(delta))
                if dist <= spec.range_vox:
                    yaw_delta = _angle_between_yaw(shooter.yaw, delta[:2])
                    if yaw_delta <= math.radians(spec.arc_deg * 0.5) and self.has_los(shooter.pos, focus.pos):
                        best = (dist, focus)

            for target in self.mechs.values():
                if not target.alive or target.team == shooter.team:
                    continue
                if best is not None and target is focus:
                    continue
                delta = target.pos - shooter.pos
                dist = float(np.linalg.norm(delta))
                if dist > spec.range_vox:
                    continue
                # Simple yaw arc gate in XY.
                yaw_delta = _angle_between_yaw(shooter.yaw, delta[:2])
                if yaw_delta > math.radians(spec.arc_deg * 0.5):
                    continue
                if not self.has_los(shooter.pos, target.pos):
                    continue
                if best is None or dist < best[0]:
                    best = (dist, target)

            if best is None:
                continue

            target = best[1]
            if spec == LASER:
                shooter.laser_cooldown = LASER.cooldown_s
                heat_transfer = LASER_HEAT_TRANSFER
            else:
                shooter.kinetic_cooldown = FLAMER.cooldown_s
                heat_transfer = FLAME_HEAT_TRANSFER
            
            shooter.heat = float(shooter.heat + spec.heat)

            mult, is_crit = self._get_damage_multiplier(target, shooter.pos)
            base_dmg = spec.damage * mult

            final_dmg, bonus_events = self._get_paint_bonus(target, base_dmg, shooter.mech_id)

            # Leg Hit Check (Consistent with Projectiles)
            is_leg_hit = False
            # Assume laser hits with some vertical jitter around the target center.
            hit_z = target.pos[2] + float(self.rng.uniform(-target.half_size[2], target.half_size[2])) * 0.5
            min_z = float(target.pos[2] - target.half_size[2])
            height = float(target.half_size[2] * 2.0)
            if hit_z < min_z + height * 0.3:
                is_leg_hit = True
                target.leg_hp -= final_dmg
                final_dmg *= 0.5

            target.hp -= final_dmg
            target.heat = float(target.heat + heat_transfer)
            target.was_hit = True
            shooter.dealt_damage += final_dmg
            target.took_damage += final_dmg

            all_events.append(
                {
                    "type": "laser_hit",
                    "weapon": spec.name,
                    "shooter": shooter.mech_id,
                    "target": target.mech_id,
                    "pos": [float(target.pos[0]), float(target.pos[1]), float(target.pos[2])],
                    "damage": final_dmg,
                    "is_crit": is_crit,
                    "is_leg_hit": is_leg_hit,
                    "is_painted": target.painted_remaining > 0.0 # For visualization
                }
            )
            all_events.extend(bonus_events)
            all_events.extend(self._handle_death(target, shooter.mech_id))

        return all_events

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

        def _lock_type(target: MechState) -> str | None:
            if self.has_los(shooter.pos, target.pos):
                return "los"
            if target.painted_remaining > 0.0 and target.last_painter_id:
                painter = self.mechs.get(target.last_painter_id)
                if painter is not None and _same_pack(painter, shooter):
                    if self._sensor_quality(shooter) >= PAINT_LOCK_MIN_QUALITY:
                        return "paint"
            return None

        focus: MechState | None = None
        if shooter.focus_target_id:
            cand = self.mechs.get(shooter.focus_target_id)
            if cand is not None and cand.alive and cand.team != shooter.team:
                focus = cand

        best: tuple[float, MechState, str] | None = None
        if focus is not None:
            delta = focus.pos - shooter.pos
            dist = float(np.linalg.norm(delta))
            if dist <= MISSILE.range_vox:
                yaw_delta = _angle_between_yaw(shooter.yaw, delta[:2])
                if yaw_delta <= math.radians(MISSILE.arc_deg * 0.5):
                    lt = _lock_type(focus)
                    if lt is not None:
                        best = (dist, focus, lt)

        for target in self.mechs.values():
            if not target.alive or target.team == shooter.team:
                continue
            if best is not None and target is focus:
                continue
            delta = target.pos - shooter.pos
            dist = float(np.linalg.norm(delta))
            if dist > MISSILE.range_vox:
                continue
            
            # Lock check: Needs LOS OR Painted
            yaw_delta = _angle_between_yaw(shooter.yaw, delta[:2])
            if yaw_delta > math.radians(MISSILE.arc_deg * 0.5):
                continue

            lt = _lock_type(target)
            if lt is None:
                continue

            if best is None or dist < best[0]:
                best = (dist, target, lt)

        if best is None:
            return []

        target = best[1]
        lock_type = best[2]
        shooter.missile_cooldown = MISSILE.cooldown_s
        shooter.heat = float(shooter.heat + MISSILE.heat)

        # Spawn projectile
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
            weapon=MISSILE.name,
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

        return [
            {
                "type": "missile_launch",
                "weapon": MISSILE.name,
                "lock": lock_type,
                "shooter": shooter.mech_id,
                "target": target.mech_id,
                "pos": [float(proj.pos[0]), float(proj.pos[1]), float(proj.pos[2])],
            }
        ]

    def _try_fire_smoke(self, shooter: MechState, action: np.ndarray) -> list[dict]:
        if shooter.spec.name != "light":
            return []
        
        fire = float(action[ActionIndex.FIRE_MISSILE]) > 0.0
        if not fire or shooter.shutdown or not shooter.alive:
            return []
        if shooter.missile_cooldown > 0.0:
            return []

        target_pos = None
        if shooter.focus_target_id:
            tgt = self.mechs.get(shooter.focus_target_id)
            if tgt and tgt.alive:
                dist = np.linalg.norm(tgt.pos - shooter.pos)
                if dist <= SMOKE.range_vox:
                    target_pos = tgt.pos
        
        forward, _ = _yaw_to_forward_right(shooter.yaw)
        if target_pos is not None:
            vel = (target_pos - shooter.pos)
            vel[2] = 0 # keep it level
            dist = np.linalg.norm(vel)
            if dist > 1e-6:
                vel = vel / dist * SMOKE.speed_vox
            else:
                vel = forward * SMOKE.speed_vox
        else:
            vel = forward * SMOKE.speed_vox
            
        proj = Projectile(
            shooter_id=shooter.mech_id,
            target_id=None,
            weapon=SMOKE.name,
            pos=shooter.pos.copy() + np.array([0, 0, shooter.half_size[2]], dtype=np.float32),
            vel=vel,
            speed=SMOKE.speed_vox,
            damage=0.0,
            stability_damage=0.0,
            max_lifetime=SMOKE.range_vox / SMOKE.speed_vox,
            guidance="linear",
            splash_rad=SMOKE.splash_rad_vox,
            splash_scale=1.0, 
        )
        self.projectiles.append(proj)
        shooter.missile_cooldown = SMOKE.cooldown_s
        shooter.heat += SMOKE.heat

        return [
            {
                "type": "smoke_launch",
                "weapon": SMOKE.name,
                "shooter": shooter.mech_id,
                "pos": [float(proj.pos[0]), float(proj.pos[1]), float(proj.pos[2])],
            }
        ]

    def _try_paint(self, shooter: MechState, action: np.ndarray) -> list[dict]:
        if len(action) < ACTION_DIM:
            return []
        paint = float(action[ActionIndex.PAINT]) > 0.0
        # Only scouts have painter
        if shooter.spec.name != "scout":
            return []
        if not paint or shooter.shutdown or not shooter.alive:
            return []
        if shooter.painter_cooldown > 0.0:
            return []

        focus: MechState | None = None
        if shooter.focus_target_id:
            cand = self.mechs.get(shooter.focus_target_id)
            if cand is not None and cand.alive and cand.team != shooter.team:
                focus = cand

        best: tuple[float, MechState] | None = None
        if focus is not None:
            delta = focus.pos - shooter.pos
            dist = float(np.linalg.norm(delta))
            if dist <= PAINTER.range_vox:
                yaw_delta = _angle_between_yaw(shooter.yaw, delta[:2])
                if yaw_delta <= math.radians(PAINTER.arc_deg * 0.5) and self.has_los(shooter.pos, focus.pos):
                    best = (dist, focus)

        for target in self.mechs.values():
            if not target.alive or target.team == shooter.team:
                continue
            if best is not None and target is focus:
                continue
            delta = target.pos - shooter.pos
            dist = float(np.linalg.norm(delta))
            if dist > PAINTER.range_vox:
                continue
            
            yaw_delta = _angle_between_yaw(shooter.yaw, delta[:2])
            if yaw_delta > math.radians(PAINTER.arc_deg * 0.5):
                continue
            if not self.has_los(shooter.pos, target.pos):
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

        return [
            {
                "type": "paint",
                "weapon": PAINTER.name,
                "shooter": shooter.mech_id,
                "target": target.mech_id,
                "pos": [float(target.pos[0]), float(target.pos[1]), float(target.pos[2])],
            }
        ]

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

        forward, _ = _yaw_to_forward_right(shooter.yaw)
        target_pos = None
        best_dist = spec.range_vox

        focus: MechState | None = None
        if shooter.focus_target_id:
            cand = self.mechs.get(shooter.focus_target_id)
            if cand is not None and cand.alive and cand.team != shooter.team:
                focus = cand

        if focus is not None:
            delta = focus.pos - shooter.pos
            dist = float(np.linalg.norm(delta))
            if dist <= best_dist:
                yaw_err = _angle_between_yaw(shooter.yaw, delta[:2])
                if yaw_err < 0.2:
                    target_pos = focus.pos
                    best_dist = dist
        
        for t in self.mechs.values():
            if t.team != shooter.team and t.alive:
                if focus is not None and t is focus:
                    continue
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
                flight_time = dist / max(1e-6, spec.speed_vox)
                g = 9.81 / self.world.voxel_size_m
                drop = 0.5 * g * (flight_time**2)
                delta = delta.copy()
                delta[2] += drop
                vel = delta / (np.linalg.norm(delta) + 1e-6) * spec.speed_vox
            elif dist > 1e-6:
                vel = delta / dist * spec.speed_vox
            else:
                vel = forward * spec.speed_vox
        else:
            vel = forward * spec.speed_vox
        
        proj = Projectile(
            shooter_id=shooter.mech_id,
            target_id=None, # Dumbfire
            weapon=spec.name,
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

        return [
            {
                "type": "kinetic_fire",
                "weapon": spec.name,
                "shooter": shooter.mech_id,
                "pos": [float(proj.pos[0]), float(proj.pos[1]), float(proj.pos[2])],
            }
        ]

    def _explode(self, pos: np.ndarray, proj: Projectile, exclude_mech: MechState | None = None) -> list[dict]:
        events = []
        for m in self.mechs.values():
            if not m.alive:
                continue
            if exclude_mech and m is exclude_mech:
                continue
            
            dist = np.linalg.norm(m.pos - pos)
            
            if dist <= proj.splash_rad:
                # Terrain occlusion: splash damage should not pass through solid voxels.
                if raycast_voxels(self.world, pos, m.pos).blocked:
                    continue
                # Apply Splash
                mult = 1.0
                is_crit = False
                raw_dmg = proj.damage * proj.splash_scale * mult
                raw_stab = proj.stability_damage * proj.splash_scale * mult
                m.hp -= raw_dmg
                m.stability = max(0.0, m.stability - raw_stab)
                m.was_hit = True
                m.took_damage += raw_dmg
                
                shooter = self.mechs.get(proj.shooter_id)
                if shooter:
                    shooter.dealt_damage += raw_dmg
                    events.append({
                        "type": "projectile_hit",
                        "weapon": proj.weapon,
                        "shooter": proj.shooter_id,
                        "target": m.mech_id,
                        "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                        "damage": raw_dmg,
                        "stability": raw_stab,
                        "is_crit": is_crit
                    })
                    events.extend(self._handle_death(m, proj.shooter_id))
        
        events.append({
            "type": "explosion",
            "weapon": proj.weapon,
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "radius": proj.splash_rad,
            "shooter": proj.shooter_id
        })
        return events

    def _impact_pos_before_voxel(
        self, start_pos: np.ndarray, end_pos: np.ndarray, blocked_voxel: tuple[int, int, int]
    ) -> np.ndarray:
        """
        Return a point just before entering `blocked_voxel` when traveling from start_pos to end_pos.

        This is used so splash/impact effects occur on the near side of cover, allowing cover to
        occlude splash damage.
        """
        start = np.asarray(start_pos, dtype=np.float64)
        end = np.asarray(end_pos, dtype=np.float64)
        d = end - start
        if float(np.dot(d, d)) <= 1e-18:
            return start.astype(np.float32, copy=False)

        bx, by, bz = blocked_voxel
        bmin = np.asarray([bx, by, bz], dtype=np.float64)
        bmax = bmin + 1.0

        t_entry = 0.0
        for axis in range(3):
            da = float(d[axis])
            if abs(da) <= 1e-18:
                continue
            if da > 0.0:
                t = float((bmin[axis] - start[axis]) / da)
            else:
                t = float((bmax[axis] - start[axis]) / da)
            if t > t_entry:
                t_entry = t

        # Step slightly back so we're outside the solid voxel.
        t = float(np.clip(t_entry - 1e-4, 0.0, 1.0))
        return (start + d * t).astype(np.float32, copy=False)

    def _update_projectiles(self) -> list[dict]:
        events = []
        keep = []
        g_vox = 9.81 / self.world.voxel_size_m

        for p in self.projectiles:
            if not p.alive:
                continue

            shooter = self.mechs.get(p.shooter_id)
            shooter_team = shooter.team if shooter is not None else None
            
            p.age += self.dt
            if p.age > p.max_lifetime:
                p.alive = False
                continue

            # Point-defense
            if p.weapon == MISSILE.name and p.guidance == "homing" and p.target_id:
                missile_target = self.mechs.get(p.target_id)
                if missile_target:
                    for defender in self.mechs.values():
                        if (
                            defender.team == missile_target.team
                            and defender.alive
                            and (not defender.shutdown)
                            and defender.spec.name == "heavy"
                            and defender.ams_cooldown <= 0.0
                        ):
                            dist = float(np.linalg.norm(defender.pos - p.pos))
                            if dist <= AMS_RANGE_VOX and self.rng.random() < AMS_INTERCEPT_PROB:
                                p.alive = False
                                defender.ams_cooldown = AMS_COOLDOWN_S
                                events.append(
                                    {
                                        "type": "ams_intercept",
                                        "weapon": "ams",
                                        "defender": defender.mech_id,
                                        "shooter": p.shooter_id,
                                        "target": missile_target.mech_id,
                                        "pos": [float(p.pos[0]), float(p.pos[1]), float(p.pos[2])],
                                    }
                                )
                                break
                    if not p.alive:
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
                p.vel[2] -= g_vox * self.dt

            # Integration
            step = p.vel * self.dt
            next_pos = p.pos + step
            
            impact = False
            impact_pos = next_pos
            direct_hit_mech = None
            step_len2 = float(np.dot(step, step))

            best_t: float | None = None
            best_target: MechState | None = None
            best_pos: np.ndarray | None = None

            if step_len2 > 1e-12:
                for target in self.mechs.values():
                    if not target.alive:
                        continue
                    if shooter_team is not None and target.team == shooter_team:
                        continue
                    if target.mech_id == p.shooter_id and p.age < 0.2:
                        continue
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
                hit = raycast_voxels(self.world, p.pos, best_pos, include_end=True)
                if hit.blocked:
                    impact = True
                    p.alive = False
                    if hit.blocked_voxel is not None:
                        impact_pos = self._impact_pos_before_voxel(p.pos, best_pos, hit.blocked_voxel)
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
                        impact_pos = self._impact_pos_before_voxel(p.pos, next_pos, hit.blocked_voxel)
            
            if impact:
                if direct_hit_mech:
                    m = direct_hit_mech
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
                    if p.weapon == AUTOCANNON.name:
                        m.suppressed_time = max(float(m.suppressed_time), float(SUPPRESS_DURATION_S))
                    m.was_hit = True
                    m.took_damage += final_dmg
                    shooter = self.mechs.get(p.shooter_id)
                    if shooter:
                        shooter.dealt_damage += final_dmg
                        events.append({
                            "type": "projectile_hit",
                            "weapon": p.weapon,
                            "shooter": shooter.mech_id,
                            "target": m.mech_id,
                            "pos": [float(impact_pos[0]), float(impact_pos[1]), float(impact_pos[2])],
                            "damage": final_dmg,
                            "stability": final_stab,
                            "is_crit": is_crit,
                            "is_leg_hit": is_leg_hit,
                            "is_painted": m.painted_remaining > 0.0
                        })
                        events.extend(bonus_events)
                        events.extend(self._handle_death(m, p.shooter_id))
                
                if p.weapon == SMOKE.name:
                    splash_pos = direct_hit_mech.pos if direct_hit_mech else impact_pos
                    cloud = SmokeCloud(pos=splash_pos.copy(), radius=p.splash_rad, remaining_life=10.0)
                    self.smoke_clouds.append(cloud)
                    events.append({
                        "type": "smoke_cloud",
                        "pos": [float(cloud.pos[0]), float(cloud.pos[1]), float(cloud.pos[2])],
                        "radius": cloud.radius
                    })
                elif p.splash_rad > 0.0:
                    splash_pos = direct_hit_mech.pos if direct_hit_mech else impact_pos
                    events.extend(self._explode(splash_pos, p, exclude_mech=direct_hit_mech))
            else:
                p.pos = next_pos
                keep.append(p)
        self.projectiles = keep
        return events

    def step(self, actions: dict[str, np.ndarray], num_substeps: int) -> list[dict]:
        events: list[dict] = []
        zero_action = np.zeros(ACTION_DIM, dtype=np.float32)
        for mech in self.mechs.values():
            mech.reset_step_stats()
        for _ in range(int(num_substeps)):
            self.tick += 1
            self.time_s += self.dt
            for cloud in self.smoke_clouds:
                if cloud.alive:
                    cloud.remaining_life -= self.dt
                    if cloud.remaining_life <= 0:
                        cloud.alive = False
            for mech in self.mechs.values():
                if not mech.alive:
                    continue
                mech.laser_cooldown = max(0.0, float(mech.laser_cooldown - self.dt))
                mech.missile_cooldown = max(0.0, float(mech.missile_cooldown - self.dt))
                mech.kinetic_cooldown = max(0.0, float(mech.kinetic_cooldown - self.dt))
                mech.painter_cooldown = max(0.0, float(mech.painter_cooldown - self.dt))
                mech.painted_remaining = max(0.0, float(mech.painted_remaining - self.dt))
                mech.suppressed_time = max(0.0, float(mech.suppressed_time - self.dt))
                mech.ams_cooldown = max(0.0, float(mech.ams_cooldown - self.dt))
                if not mech.shutdown:
                    if mech.ecm_on:
                        mech.heat = float(mech.heat + ECM_HEAT_PER_S * self.dt)
                    if mech.eccm_on:
                        mech.heat = float(mech.heat + ECCM_HEAT_PER_S * self.dt)
                events.extend(self._apply_hazards(mech))
                if mech.fallen_time > 0.0:
                    mech.fallen_time -= self.dt
                    if mech.fallen_time <= 0.0:
                        mech.fallen_time = 0.0
                        mech.stability = mech.max_stability * 0.5
                else:
                    if mech.stability <= 0.0:
                        mech.fallen_time = 3.0
                        mech.vel[:] = 0.0
                    else:
                        regen = 10.0 * self.dt
                        max_stab = mech.max_stability
                        if mech.is_legged:
                            regen *= 0.5
                            max_stab *= 0.5
                        if np.linalg.norm(mech.vel) > 1.0:
                            regen *= 0.5
                        if mech.suppressed_time > 0.0:
                            regen *= SUPPRESS_REGEN_SCALE
                        mech.stability = min(max_stab, mech.stability + regen)
                self._apply_vent(mech, actions.get(mech.mech_id, zero_action))
                self._dissipate(mech)
            events.extend(self._update_projectiles())
            for mech in self.mechs.values():
                if not mech.alive:
                    continue
                a = actions.get(mech.mech_id, zero_action)
                events.extend(self._try_fire_laser(mech, a))
                events.extend(self._try_fire_missile(mech, a))
                events.extend(self._try_fire_kinetic(mech, a))
                events.extend(self._try_paint(mech, a))
                events.extend(self._try_fire_smoke(mech, a))
            order = [m for m in self.mechs.values() if m.alive]
            self.rng.shuffle(order)
            for mech in order:
                if mech.fallen_time > 0.0: 
                    self._apply_movement(mech, zero_action)
                    self._integrate(mech)
                    continue
                a = actions.get(mech.mech_id, zero_action)
                self._apply_rotation(mech, a)
                self._apply_movement(mech, a)
                self._integrate(mech)
        return events
