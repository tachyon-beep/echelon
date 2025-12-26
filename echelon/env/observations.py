"""Observation computation for EchelonEnv.

This module encapsulates observation logic, making it easier to:
1. Understand observation composition
2. Add new observation components (e.g., mission embedding for curriculum)
3. Test observation logic in isolation

Design:
- ObservationContext: All state needed for observation computation
- ObservationBuilder: Configurable observation construction
- Helper functions: Isolated computation for testability
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config import (
    AMS_COOLDOWN_S,
    ECCM_RADIUS_VOX,
    ECCM_WEIGHT,
    ECM_RADIUS_VOX,
    ECM_WEIGHT,
    GAUSS,
    LASER,
    MISSILE,
    PAINT_LOCK_MIN_QUALITY,
    PAINTER,
    SENSOR_QUALITY_MAX,
    SENSOR_QUALITY_MIN,
    SUPPRESS_DURATION_S,
    EnvConfig,
)
from ..constants import PACK_SIZE
from ..gen.objective import capture_zone_params
from ..rl.suite import SUITE_DESCRIPTOR_DIM, CombatSuiteSpec, build_suite_descriptor

if TYPE_CHECKING:
    from ..sim.mech import MechState
    from ..sim.sim import Sim
    from ..sim.world import VoxelWorld
    from .env import EchelonEnv


# ============================================================================
# Constants
# ============================================================================

# Mech class indices for one-hot encoding (must match order in class_onehot)
MECH_CLASS_INDEX = {"scout": 0, "light": 1, "medium": 2, "heavy": 3}
NUM_MECH_CLASSES = len(MECH_CLASS_INDEX)

# Mech class ranks for contact sorting (higher = priority target)
MECH_CLASS_RANK = {"scout": 1, "light": 1, "medium": 2, "heavy": 3}

# Mech class weights for threat/friendly mass calculation
MECH_CLASS_WEIGHT = {"scout": 0.5, "light": 0.7, "medium": 1.0, "heavy": 1.5}

# Expected contact feature dimension (validated at runtime)
CONTACT_FEATURE_DIM = 25

# Angular constants
HALF_PI = math.pi / 2


# ============================================================================
# Helper Functions
# ============================================================================


def _wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def compute_contact_features(
    viewer: MechState,
    other: MechState,
    world: VoxelWorld,
    relation: str,
    painted_by_pack: bool,
    contact_dim: int,
) -> np.ndarray:
    """Compute feature vector for a single contact.

    Args:
        viewer: The observing mech
        other: The observed mech
        world: The voxel world
        relation: "friendly", "hostile", or "neutral"
        painted_by_pack: Whether target is painted by viewer's pack
        contact_dim: Expected dimension of output (must be CONTACT_FEATURE_DIM)

    Returns:
        Feature vector of shape (contact_dim,)

    Raises:
        AssertionError: If contact_dim != CONTACT_FEATURE_DIM
    """
    assert contact_dim == CONTACT_FEATURE_DIM, f"contact_dim must be {CONTACT_FEATURE_DIM}, got {contact_dim}"
    feat = np.zeros(contact_dim, dtype=np.float32)

    rel = (other.pos - viewer.pos).astype(np.float32, copy=False)
    max_dim = float(max(world.size_x, world.size_y, world.size_z))
    rel /= max(1e-6, max_dim)
    rel_vel = (other.vel - viewer.vel).astype(np.float32, copy=False) / 10.0

    yaw_sin = float(math.sin(other.yaw))
    yaw_cos = float(math.cos(other.yaw))
    hp_norm = float(np.clip(other.hp / max(1.0, other.spec.hp), 0.0, 1.0))
    heat_norm = float(np.clip(other.heat / max(1.0, other.spec.heat_cap), 0.0, 2.0))
    stab_norm = float(np.clip(other.stability / max(1.0, other.max_stability), 0.0, 1.0))
    fallen = 1.0 if other.fallen_time > 0.0 else 0.0
    is_legged = 1.0 if other.is_legged else 0.0

    rel_onehot = np.zeros(3, dtype=np.float32)
    if relation == "friendly":
        rel_onehot[0] = 1.0
    elif relation == "hostile":
        rel_onehot[1] = 1.0
    elif relation == "neutral":
        rel_onehot[2] = 1.0
    else:
        raise ValueError(f"Unknown relation: {relation!r}")

    class_onehot = np.zeros(NUM_MECH_CLASSES, dtype=np.float32)
    cls_idx = MECH_CLASS_INDEX.get(other.spec.name)
    if cls_idx is not None:
        class_onehot[cls_idx] = 1.0

    feat[0:3] = rel
    feat[3:6] = rel_vel
    feat[6:8] = np.asarray([yaw_sin, yaw_cos], dtype=np.float32)
    feat[8:10] = np.asarray([hp_norm, heat_norm], dtype=np.float32)
    feat[10:13] = np.asarray([stab_norm, fallen, is_legged], dtype=np.float32)
    feat[13:16] = rel_onehot
    feat[16:20] = class_onehot
    feat[20] = 1.0 if painted_by_pack else 0.0
    feat[21] = 1.0  # visible

    # lead_pitch: Suggested pitch for a ballistic hit (Gauss speed 40vox/s)
    GAUSS_SPEED = 40.0
    dist_xy = float(math.hypot(rel[0], rel[1])) * max_dim
    if dist_xy > 0.1:
        dz = rel[2] * max_dim
        flight_time = dist_xy / GAUSS_SPEED
        g = 9.81 / world.voxel_size_m
        drop = 0.5 * g * (flight_time**2)
        pitch = math.atan2(dz + drop, dist_xy)
        feat[22] = float(np.clip(pitch / (math.pi / 4), -1.0, 1.0))
    else:
        feat[22] = 0.0

    # Closing rate: positive = approaching, negative = separating
    rel_vec_raw = other.pos - viewer.pos
    rel_vel_raw = other.vel - viewer.vel
    dist = float(np.linalg.norm(rel_vec_raw))
    if dist > 0.1:
        closing = float(-np.dot(rel_vel_raw, rel_vec_raw) / dist)
        feat[23] = float(np.clip(closing / 10.0, -1.0, 1.0))
    else:
        feat[23] = 0.0

    # Crossing angle: 0 = parallel to LOS, 1 = perpendicular
    other_speed = float(np.linalg.norm(other.vel))
    if dist > 0.1 and other_speed > 0.5:
        los_dir = rel_vec_raw / dist
        vel_dir = other.vel / other_speed
        cos_angle = float(np.dot(los_dir, vel_dir))
        feat[24] = 1.0 - abs(cos_angle)
    else:
        feat[24] = 0.0

    return feat


def compute_ewar_levels(viewer: MechState, sim: Sim) -> tuple[float, float, float]:
    """Compute electronic warfare levels for a mech.

    Args:
        viewer: The observing mech
        sim: The simulation state

    Returns:
        (sensor_quality, jam_level, eccm_level)
    """
    if not viewer.alive or viewer.shutdown:
        return float(SENSOR_QUALITY_MIN), 0.0, 0.0

    jam = 0.0
    eccm = 0.0
    for other in sim.mechs.values():
        if not other.alive or other.mech_id == viewer.mech_id:
            continue
        if other.shutdown:
            continue

        d = other.pos - viewer.pos
        dist = float(np.linalg.norm(d))
        if other.team != viewer.team and other.ecm_on and dist <= ECM_RADIUS_VOX:
            jam = max(jam, max(0.0, 1.0 - dist / max(1e-6, float(ECM_RADIUS_VOX))))
        if other.team == viewer.team and other.eccm_on and dist <= ECCM_RADIUS_VOX:
            eccm = max(eccm, max(0.0, 1.0 - dist / max(1e-6, float(ECCM_RADIUS_VOX))))

    sensor_quality = 1.0 - float(ECM_WEIGHT) * jam + float(ECCM_WEIGHT) * eccm
    sensor_quality = float(np.clip(sensor_quality, float(SENSOR_QUALITY_MIN), float(SENSOR_QUALITY_MAX)))
    return sensor_quality, float(jam), float(eccm)


def compute_local_map(
    viewer: MechState,
    world: VoxelWorld,
    occupancy_2d: np.ndarray,
    local_map_r: int,
) -> np.ndarray:
    """Compute ego-centric local occupancy map.

    Args:
        viewer: The observing mech
        world: The voxel world
        occupancy_2d: Pre-computed 2D occupancy grid
        local_map_r: Radius of local map

    Returns:
        Flattened local map of shape ((2*r+1)^2,)
    """
    r = int(local_map_r)
    size = 2 * r + 1

    yaw = float(viewer.yaw)
    c, s = math.cos(yaw), math.sin(yaw)

    # Create grid of (fwd, right) relative offsets
    fwd_grid, right_grid = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")

    # Rotate grid to world XY
    dx = c * fwd_grid - s * right_grid
    dy = s * fwd_grid + c * right_grid

    # World coordinates
    wx = (viewer.pos[0] + dx).astype(np.int32)
    wy = (viewer.pos[1] + dy).astype(np.int32)

    # Mask for in-bounds
    mask = (wx >= 0) & (wx < world.size_x) & (wy >= 0) & (wy < world.size_y)

    out = np.ones((size, size), dtype=np.float32)
    # Only sample if in bounds, otherwise leave as 1.0 (solid)
    out[mask] = occupancy_2d[wy[mask], wx[mask]].astype(np.float32)

    return out.reshape(-1)


def compute_acoustic_intensities(
    viewer: MechState,
    sim: Sim,
    possible_agents: list[str],
    los_cache: dict[tuple[str, str], bool],
) -> np.ndarray:
    """Compute acoustic intensity from 4 relative quadrants.

    Args:
        viewer: The observing mech
        sim: The simulation state
        possible_agents: List of all agent IDs
        los_cache: Cache for LOS checks

    Returns:
        4-dim array of acoustic intensities [front-left, front-right, rear-left, rear-right]
    """
    acoustic_intensities = np.zeros(4, dtype=np.float32)
    aid = viewer.mech_id

    for other_id in possible_agents:
        if other_id == aid:
            continue
        other = sim.mechs.get(other_id)
        if other is None or not other.alive or other.noise_level <= 0:
            continue

        delta = other.pos - viewer.pos
        dist_sq = float(np.dot(delta, delta))

        # Inverse square law (clamped)
        intensity = other.noise_level / (1.0 + dist_sq)

        # Attenuation by terrain
        key = (aid, other_id) if aid < other_id else (other_id, aid)
        if key not in los_cache:
            los_cache[key] = bool(sim.has_los(viewer.pos, other.pos))
        if not los_cache[key]:
            intensity *= 0.20  # 80% loss through walls

        # Determine relative quadrant
        angle = _wrap_pi(math.atan2(float(delta[1]), float(delta[0])) - float(viewer.yaw))
        if angle >= 0:  # Left side
            q = 0 if angle < HALF_PI else 2  # Front-Left vs Rear-Left
        else:  # Right side
            q = 1 if angle > -HALF_PI else 3  # Front-Right vs Rear-Right
        acoustic_intensities[q] += float(intensity)

    # Log scaling to [0, 1] range, clipped to prevent overflow
    acoustic_intensities = np.clip(np.log1p(acoustic_intensities) / 5.0, 0.0, 1.0).astype(np.float32)
    return acoustic_intensities


@dataclass
class ObservationContext:
    """All state needed for observation computation.

    This bundles environment state into a single object for cleaner interfaces.
    """

    # Core simulation state
    sim: Sim
    world: VoxelWorld
    config: EnvConfig

    # Agent lists
    agents: list[str]
    possible_agents: list[str]

    # Pack relationships
    packmates: dict[str, list[str]]

    # Observation control state
    contact_sort_mode: dict[str, int]
    contact_filter_hostile: dict[str, bool]
    last_contact_slots: dict[str, list[str | None]]

    # Communication state
    comm_last: dict[str, np.ndarray]
    comm_dim: int

    # Combat suites
    mech_suites: dict[str, CombatSuiteSpec]

    # Cached data
    cached_occupancy_2d: np.ndarray
    cached_telemetry: np.ndarray

    # Zone score state
    team_zone_score: dict[str, float]
    zone_score_to_win: float

    # Episode stats (for tracking)
    episode_stats: dict[str, float]

    @classmethod
    def from_env(cls, env: EchelonEnv) -> ObservationContext:
        """Create ObservationContext from environment state.

        Note: Should only be called after env.reset() when sim/world are initialized.
        """
        # These are guaranteed to be set after reset()
        assert env.sim is not None, "from_env() called before reset()"
        assert env.world is not None, "from_env() called before reset()"
        assert env._cached_occupancy_2d is not None, "from_env() called before reset()"
        assert env._cached_telemetry is not None, "from_env() called before reset()"

        return cls(
            sim=env.sim,
            world=env.world,
            config=env.config,
            agents=env.agents,
            possible_agents=env.possible_agents,
            packmates=env._packmates,
            contact_sort_mode=env._contact_sort_mode,
            contact_filter_hostile=env._contact_filter_hostile,
            last_contact_slots=env._last_contact_slots,
            comm_last=env._comm_last,
            comm_dim=env.comm_dim,
            mech_suites=env._mech_suites,
            cached_occupancy_2d=env._cached_occupancy_2d,
            cached_telemetry=env._cached_telemetry,
            team_zone_score=env.team_zone_score,
            zone_score_to_win=env.zone_score_to_win,
            episode_stats=env._episode_stats,
        )


# Observation dimension constants
ORDER_OBS_DIM = 20  # order_type(6) + time(1) + target_pos(3) + progress(1) + override(6) + flags(3)
PANEL_STATS_DIM = 8  # contact_count + intel + order + squad + threat + friendly + alert + detail


class ObservationBuilder:
    """Configurable observation construction.

    This class encapsulates observation logic and can be extended
    for curriculum-based observation modifications.
    """

    def __init__(
        self,
        config: EnvConfig,
        max_contact_slots: int,
        contact_dim: int,
        local_map_r: int,
        comm_dim: int,
    ):
        self.config = config
        self.max_contact_slots = max_contact_slots
        self.contact_dim = contact_dim
        self.local_map_r = local_map_r
        self.local_map_dim = (2 * local_map_r + 1) ** 2
        self.comm_dim = comm_dim
        self.telemetry_dim = 16 * 16

        # Self features dimension breakdown:
        # acoustic_quadrants(4) + hull_type_onehot(4) + suite_descriptor(14) +
        # targeted, under_fire, painted, shutdown, crit_heat, self_hp_norm, self_heat_norm,
        # heat_headroom, stability_risk, damage_dir_local(3), incoming_missile, sensor_quality,
        # jam_level, ecm_on, eccm_on, suppressed, ams_cd, self_vel(3), cooldowns(4), in_zone,
        # vec_to_zone(3), zone_radius, my_control, my_score, enemy_score, time_frac,
        # obs_sort_onehot(3), hostile_only = 47
        self.self_dim = 47

    def obs_dim(self) -> int:
        """Compute total observation dimension."""
        comm_dim = PACK_SIZE * int(max(0, self.comm_dim))
        return (
            self.max_contact_slots * self.contact_dim
            + comm_dim
            + self.local_map_dim
            + self.telemetry_dim
            + self.self_dim
            + SUITE_DESCRIPTOR_DIM
            + ORDER_OBS_DIM
            + PANEL_STATS_DIM
        )

    def build(self, ctx: ObservationContext) -> dict[str, np.ndarray]:
        """Build observations for all agents.

        WARNING: This method mutates ctx.last_contact_slots and ctx.episode_stats
        as a side effect. These mutations are required for observation history
        tracking and episode statistics collection.

        Args:
            ctx: Observation context with all required state (WILL BE MUTATED)

        Returns:
            Dict mapping agent_id -> observation vector
        """
        sim = ctx.sim
        world = ctx.world
        assert sim is not None and world is not None

        base_radar_range = 14.0
        zone_cx, zone_cy, zone_r = capture_zone_params(world.meta, size_x=world.size_x, size_y=world.size_y)
        max_xy = float(max(world.size_x, world.size_y))
        zone_r_norm = float(np.clip(zone_r / max(1.0, max_xy), 0.0, 1.0))

        # Compute zone control
        in_zone_tonnage: dict[str, float] = {"blue": 0.0, "red": 0.0}
        for m in sim.mechs.values():
            if not m.alive:
                continue
            dist = float(math.hypot(float(m.pos[0] - zone_cx), float(m.pos[1] - zone_cy)))
            if dist < zone_r:
                in_zone_tonnage[m.team] += float(m.spec.tonnage)

        total_tonnage = in_zone_tonnage["blue"] + in_zone_tonnage["red"]
        if total_tonnage > 0:
            zone_control = (in_zone_tonnage["blue"] - in_zone_tonnage["red"]) / total_tonnage
        else:
            zone_control = 0.0

        time_frac = float(
            np.clip(float(sim.time_s) / max(1e-6, float(ctx.config.max_episode_seconds)), 0.0, 1.0)
        )
        los_cache: dict[tuple[str, str], bool] = {}
        smoke_cache: dict[tuple[str, str], bool] = {}

        def _cached_los(a: str, b: str, a_pos: np.ndarray, b_pos: np.ndarray) -> bool:
            key = (a, b) if a < b else (b, a)
            if key in los_cache:
                return los_cache[key]
            ok = bool(sim.has_los(a_pos, b_pos))
            los_cache[key] = ok
            return ok

        def _cached_smoke_los(a: str, b: str, a_pos: np.ndarray, b_pos: np.ndarray) -> bool:
            key = (a, b) if a < b else (b, a)
            if key in smoke_cache:
                return smoke_cache[key]
            ok = bool(sim.has_smoke_los(a_pos, b_pos))
            smoke_cache[key] = ok
            return ok

        telemetry_flat = ctx.cached_telemetry
        reserved = {"friendly": 3, "hostile": 1, "neutral": 1}
        repurpose_priority = ["hostile", "friendly", "neutral"]

        obs: dict[str, np.ndarray] = {}
        for aid in ctx.agents:
            viewer = sim.mechs[aid]
            if not viewer.alive:
                obs[aid] = np.zeros(self.obs_dim(), dtype=np.float32)
                ctx.last_contact_slots[aid] = [None] * self.max_contact_slots
                continue

            # Get combat suite for this mech
            suite = ctx.mech_suites.get(aid)
            effective_contact_slots = suite.visual_contact_slots if suite else 5

            pack_ids = ctx.packmates.get(aid, [])
            sort_mode = int(ctx.contact_sort_mode.get(aid, 0))
            hostile_only = bool(ctx.contact_filter_hostile.get(aid, False))

            sensor_quality, jam_level, _eccm_level = compute_ewar_levels(viewer, sim)
            radar_range = float(base_radar_range) * float(sensor_quality)

            # Acoustic sensing
            acoustic_intensities = compute_acoustic_intensities(viewer, sim, ctx.possible_agents, los_cache)

            # Top-K contact table
            contacts: dict[str, list[tuple[tuple[float, ...] | tuple[int, float], float, str, bool]]] = {
                "friendly": [],
                "hostile": [],
                "neutral": [],
            }
            for other_id in ctx.possible_agents:
                if other_id == aid:
                    continue
                other = sim.mechs[other_id]
                if not other.alive:
                    continue

                delta = other.pos - viewer.pos
                dist = float(np.linalg.norm(delta))

                painted_by_pack = bool(
                    other.painted_remaining > 0.0
                    and other.last_painter_id is not None
                    and other.last_painter_id in pack_ids
                )

                if ctx.config.observation_mode == "full":
                    visible = True
                else:
                    smoke_ok = _cached_smoke_los(aid, other_id, viewer.pos, other.pos)
                    painted_visible = painted_by_pack and (sensor_quality >= float(PAINT_LOCK_MIN_QUALITY))
                    heat_bloom = ((other.heat / max(1.0, other.spec.heat_cap)) > 0.80) and smoke_ok
                    ewar_bloom = (other.ecm_on or other.eccm_on) and smoke_ok

                    if (
                        painted_visible
                        or dist <= radar_range
                        or (heat_bloom and other.team != viewer.team)
                        or (ewar_bloom and other.team != viewer.team)
                    ):
                        visible = True
                    else:
                        visible = _cached_los(aid, other_id, viewer.pos, other.pos)

                if not visible:
                    continue

                relation = "friendly" if other.team == viewer.team else "hostile"
                if sort_mode == 0:
                    key: tuple[float, ...] | tuple[int, float] = (dist,)
                elif sort_mode == 1:
                    rank = MECH_CLASS_RANK.get(other.spec.name, 0)
                    key = (-rank, dist)
                elif sort_mode == 2:
                    hp_norm = float(np.clip(other.hp / max(1.0, other.spec.hp), 0.0, 1.0))
                    key = (hp_norm, dist)
                else:
                    key = (dist,)
                contacts[relation].append((key, dist, other_id, painted_by_pack))

            for rel in contacts:
                contacts[rel].sort(key=lambda t: t[0])

            selected: list[tuple[str, str, bool]] = []
            used_counts = {"friendly": 0, "hostile": 0, "neutral": 0}
            max_contact_count = effective_contact_slots

            if hostile_only:
                reserved_local = {"friendly": 0, "hostile": max_contact_count, "neutral": 0}
                repurpose_local = ["hostile"]
            else:
                reserved_local = reserved
                repurpose_local = repurpose_priority

            # Fill reserved slots
            for rel in ("friendly", "hostile", "neutral"):
                take = min(int(reserved_local[rel]), len(contacts[rel]))
                for i in range(take):
                    _, _, oid, painted = contacts[rel][i]
                    selected.append((rel, oid, painted))
                used_counts[rel] += take

            # Repurpose unused slots
            while len(selected) < max_contact_count:
                filled = False
                for rel in repurpose_local:
                    i = used_counts[rel]
                    if i < len(contacts[rel]):
                        _, _, oid, painted = contacts[rel][i]
                        selected.append((rel, oid, painted))
                        used_counts[rel] += 1
                        filled = True
                        break
                if not filled:
                    break

            # Build contact features
            contact_feats = np.zeros((self.max_contact_slots, self.contact_dim), dtype=np.float32)
            for i, (rel, oid, painted) in enumerate(selected[:max_contact_count]):
                contact_feats[i, :] = compute_contact_features(
                    viewer,
                    sim.mechs[oid],
                    world,
                    relation=rel,
                    painted_by_pack=painted,
                    contact_dim=self.contact_dim,
                )

            slot_ids: list[str | None] = [None] * self.max_contact_slots
            for i, (_, oid, _) in enumerate(selected[:max_contact_count]):
                slot_ids[i] = oid
            ctx.last_contact_slots[aid] = slot_ids

            contact_count = min(len(selected), max_contact_count)

            # Track perception metrics
            ctx.episode_stats["visible_contacts_sum"] = ctx.episode_stats.get(
                "visible_contacts_sum", 0.0
            ) + float(contact_count)
            ctx.episode_stats["visible_contacts_count"] = (
                ctx.episode_stats.get("visible_contacts_count", 0.0) + 1.0
            )
            if hostile_only:
                ctx.episode_stats["hostile_filter_on_count"] = (
                    ctx.episode_stats.get("hostile_filter_on_count", 0.0) + 1.0
                )

            parts: list[np.ndarray] = []
            parts.append(contact_feats.reshape(-1))

            # Pack comms
            if ctx.comm_dim > 0:
                comm = np.zeros((PACK_SIZE, ctx.comm_dim), dtype=np.float32)
                for i, pid in enumerate(pack_ids[:PACK_SIZE]):
                    if not sim.mechs[pid].alive:
                        continue
                    msg = ctx.comm_last.get(pid)
                    if msg is None:
                        continue
                    comm[i, :] = msg
                parts.append(comm.reshape(-1))

            # Local map
            parts.append(compute_local_map(viewer, world, ctx.cached_occupancy_2d, self.local_map_r))

            # Telemetry
            parts.append(telemetry_flat)

            # Self state features
            targeted = 1.0 if self._is_targeted(viewer, sim) else 0.0
            under_fire = 1.0 if viewer.was_hit else 0.0
            painted_flag = 1.0 if viewer.painted_remaining > 0.0 else 0.0
            shutdown = 1.0 if viewer.shutdown else 0.0
            crit_heat = 1.0 if (viewer.heat / max(1.0, viewer.spec.heat_cap)) > 0.85 else 0.0

            self_hp_norm = float(np.clip(viewer.hp / max(1.0, viewer.spec.hp), 0.0, 1.0))
            self_heat_norm = float(np.clip(viewer.heat / max(1.0, viewer.spec.heat_cap), 0.0, 2.0))
            heat_headroom = float(
                np.clip((viewer.spec.heat_cap - viewer.heat) / max(1.0, viewer.spec.heat_cap), 0.0, 1.0)
            )
            stability_risk = float(
                np.clip(1.0 - (viewer.stability / max(1.0, viewer.max_stability)), 0.0, 1.0)
            )

            if viewer.last_damage_dir is not None:
                cos_y, sin_y = math.cos(-viewer.yaw), math.sin(-viewer.yaw)
                dx = float(viewer.last_damage_dir[0])
                dy = float(viewer.last_damage_dir[1])
                dz = float(viewer.last_damage_dir[2])
                # Transform to local frame and clip to [-1, 1] per component
                damage_dir_local = np.clip(
                    np.array(
                        [dx * cos_y - dy * sin_y, dx * sin_y + dy * cos_y, dz],
                        dtype=np.float32,
                    ),
                    -1.0,
                    1.0,
                )
            else:
                damage_dir_local = np.zeros(3, dtype=np.float32)

            # Missile Warning System: Intentionally bypasses sensor checks.
            # In-universe justification: All mechs have onboard missile warning receivers
            # that detect active homing radar locks regardless of ECM interference.
            # This gives agents time to deploy smoke or take evasive action.
            incoming_missile = 0.0
            for p in sim.projectiles:
                if not p.alive:
                    continue
                if p.weapon != MISSILE.name:
                    continue
                if p.guidance != "homing":
                    continue
                if p.target_id == viewer.mech_id:
                    incoming_missile = 1.0
                    break

            sensor_quality_norm = float(np.clip(sensor_quality / float(SENSOR_QUALITY_MAX), 0.0, 1.0))
            jam_norm = float(np.clip(jam_level, 0.0, 1.0))

            ecm_on = 1.0 if (viewer.ecm_on and (not viewer.shutdown)) else 0.0
            eccm_on = 1.0 if (viewer.eccm_on and (not viewer.shutdown)) else 0.0
            suppressed_norm = float(
                np.clip(viewer.suppressed_time / max(1e-6, float(SUPPRESS_DURATION_S)), 0.0, 1.0)
            )
            ams_cd_norm = float(np.clip(viewer.ams_cooldown / max(1e-6, float(AMS_COOLDOWN_S)), 0.0, 1.0))

            if viewer.shutdown:
                self_vel = np.zeros(3, dtype=np.float32)
            else:
                self_vel = (viewer.vel / 10.0).astype(np.float32, copy=False)

            laser_cd = float(np.clip(viewer.laser_cooldown / LASER.cooldown_s, 0.0, 1.0))
            missile_cd = float(np.clip(viewer.missile_cooldown / MISSILE.cooldown_s, 0.0, 1.0))
            kinetic_cd = float(np.clip(viewer.kinetic_cooldown / GAUSS.cooldown_s, 0.0, 1.0))
            painter_cd = float(np.clip(viewer.painter_cooldown / PAINTER.cooldown_s, 0.0, 1.0))

            # Objective features
            in_zone = 0.0
            dist_to_zone = float(math.hypot(float(viewer.pos[0] - zone_cx), float(viewer.pos[1] - zone_cy)))
            if dist_to_zone < zone_r:
                in_zone = 1.0
            zone_rel = np.array([zone_cx, zone_cy, 0.0], dtype=np.float32) - viewer.pos
            zone_rel /= max(1.0, max_xy)

            my_team = viewer.team
            enemy_team = "red" if my_team == "blue" else "blue"
            my_control = zone_control if my_team == "blue" else -zone_control
            my_score = float(ctx.team_zone_score.get(my_team, 0.0))
            enemy_score = float(ctx.team_zone_score.get(enemy_team, 0.0))
            win_scale = max(1e-6, float(ctx.zone_score_to_win))
            my_score_norm = float(np.clip(my_score / win_scale, 0.0, 1.0))
            enemy_score_norm = float(np.clip(enemy_score / win_scale, 0.0, 1.0))

            # Hull type one-hot
            hull_type = np.zeros(NUM_MECH_CLASSES, dtype=np.float32)
            hull_idx = MECH_CLASS_INDEX.get(viewer.spec.name)
            if hull_idx is not None:
                hull_type[hull_idx] = 1.0

            # Suite descriptor
            if suite is not None:
                suite_desc = np.array(build_suite_descriptor(suite), dtype=np.float32)
            else:
                suite_desc = np.zeros(SUITE_DESCRIPTOR_DIM, dtype=np.float32)

            # Received order
            order_obs = self._encode_received_order(viewer, sim)

            # Panel stats
            panel_stats = self._build_panel_stats(viewer, sim, contact_count, suite, ctx.last_contact_slots)

            parts.append(acoustic_intensities)
            parts.append(hull_type)
            parts.append(suite_desc)
            parts.append(order_obs)
            parts.append(panel_stats)

            parts.append(
                np.asarray(
                    [
                        targeted,
                        under_fire,
                        painted_flag,
                        shutdown,
                        crit_heat,
                        self_hp_norm,
                        self_heat_norm,
                        heat_headroom,
                        stability_risk,
                        damage_dir_local[0],
                        damage_dir_local[1],
                        damage_dir_local[2],
                        incoming_missile,
                        sensor_quality_norm,
                        jam_norm,
                        ecm_on,
                        eccm_on,
                        suppressed_norm,
                        ams_cd_norm,
                        self_vel[0],
                        self_vel[1],
                        self_vel[2],
                        laser_cd,
                        missile_cd,
                        kinetic_cd,
                        painter_cd,
                        in_zone,
                        zone_rel[0],
                        zone_rel[1],
                        zone_rel[2],
                        zone_r_norm,
                        my_control,
                        my_score_norm,
                        enemy_score_norm,
                        time_frac,
                        1.0 if sort_mode == 0 else 0.0,
                        1.0 if sort_mode == 1 else 0.0,
                        1.0 if sort_mode == 2 else 0.0,
                        1.0 if hostile_only else 0.0,
                    ],
                    dtype=np.float32,
                )
            )

            vec = np.concatenate(parts).astype(np.float32, copy=False)
            if not np.all(np.isfinite(vec)):
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                ctx.episode_stats[f"bad_obs_{viewer.team}"] = float(
                    ctx.episode_stats.get(f"bad_obs_{viewer.team}", 0.0) + 1.0
                )
            obs[aid] = vec

        return obs

    def _is_targeted(self, mech: MechState, sim: Sim) -> bool:
        """Check if mech is being targeted by any enemy."""
        return any(sim.has_los(other.pos, mech.pos) for other in sim.enemies_in_range(mech, 8.0))

    def _encode_received_order(self, viewer: MechState, sim: Sim) -> np.ndarray:
        """Encode received order into observation vector (20 dims)."""
        from ..actions import OrderType

        order_obs = np.zeros(ORDER_OBS_DIM, dtype=np.float32)

        order = viewer.current_order
        if order is None or order.is_expired(sim.time_s):
            return order_obs

        # Order type one-hot
        order_type_idx = min(order.order_type, 5)
        order_obs[order_type_idx] = 1.0

        # Time since order
        time_since = sim.time_s - order.issued_at
        order_obs[6] = float(np.clip(time_since / 10.0, 0.0, 1.0))

        # Target relative position
        if order.order_type == OrderType.FOCUS_FIRE and order.target_id is not None:
            target = sim.mechs.get(order.target_id)
            if target is not None and target.alive:
                world = sim.world
                max_dim = float(max(world.size_x, world.size_y, world.size_z))
                rel_pos = (target.pos - viewer.pos) / max_dim
                order_obs[7:10] = rel_pos

        # Progress
        order_obs[10] = float(np.clip(order.progress, 0.0, 1.0))

        # Override reason one-hot
        override_idx = min(order.override_reason, 5)
        order_obs[11 + override_idx] = 1.0

        # Status flags
        order_obs[17] = 1.0 if order.is_engaged else 0.0
        order_obs[18] = 1.0 if order.is_blocked else 0.0
        order_obs[19] = 1.0 if order.needs_support else 0.0

        return order_obs

    def _build_panel_stats(
        self,
        viewer: MechState,
        sim: Sim,
        contact_count: int,
        suite: CombatSuiteSpec | None,
        last_contact_slots: dict[str, list[str | None]],
    ) -> np.ndarray:
        """Build panel stats observation vector (8 dims)."""
        stats = np.zeros(PANEL_STATS_DIM, dtype=np.float32)

        max_contacts = suite.visual_contact_slots if suite else self.max_contact_slots
        stats[0] = float(contact_count) / max(1, max_contacts)
        stats[1] = 0.0  # Intel placeholder

        has_order = viewer.current_order is not None and not viewer.current_order.is_expired(sim.time_s)
        stats[2] = 1.0 if has_order else 0.0

        stats[3] = float(PACK_SIZE - 1) / float(PACK_SIZE)

        threat_mass = 0.0
        friendly_mass = 0.0

        for slot_id in last_contact_slots.get(viewer.mech_id, []):
            if slot_id is None:
                continue
            other = sim.mechs.get(slot_id)
            if other is None or not other.alive:
                continue
            weight = MECH_CLASS_WEIGHT.get(other.spec.name, 1.0)
            if other.team != viewer.team:
                threat_mass += weight
            else:
                friendly_mass += weight

        stats[4] = min(1.0, threat_mass / 5.0)
        stats[5] = min(1.0, friendly_mass / 5.0)

        has_alert = (
            viewer.was_hit
            or viewer.hp < viewer.spec.hp * 0.25
            or viewer.heat > viewer.spec.heat_cap * 0.9
            or viewer.stability < 20.0
        )
        stats[6] = 1.0 if has_alert else 0.0

        stats[7] = suite.squad_detail_norm if suite else 0.0

        return stats
