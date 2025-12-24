from __future__ import annotations

import math

import numpy as np

# Gym space definitions (optional - graceful fallback if not installed)
try:
    from gymnasium import spaces as gym_spaces

    _HAS_GYMNASIUM = True
except ImportError:
    try:
        from gym import spaces as gym_spaces

        _HAS_GYMNASIUM = True
    except ImportError:
        gym_spaces = None
        _HAS_GYMNASIUM = False

from ..actions import ACTION_DIM as ACTION_DIM_CONST
from ..actions import ActionIndex
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
    MechClassConfig,
)
from ..constants import PACK_SIZE
from ..gen.corridors import carve_macro_corridors
from ..gen.objective import capture_zone_params, clear_capture_zone, sample_capture_zone
from ..gen.recipe import build_recipe
from ..gen.transforms import (
    TransformType,
    apply_transform_voxels,
    list_transforms,
    opposite_corner,
    transform_corner,
)
from ..gen.validator import ConnectivityValidator
from ..nav.graph import NavGraph, NodeID
from ..nav.planner import Planner
from ..sim.mech import MechState
from ..sim.sim import Sim
from ..sim.world import VoxelWorld


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def default_mech_classes() -> dict[str, MechClassConfig]:
    return {
        "scout": MechClassConfig(
            name="scout",
            size_voxels=(1.2, 1.2, 1.8),
            max_speed=7.0,
            max_yaw_rate=1.8,
            max_jet_accel=18.0,
            hp=60.0,
            leg_hp=30.0,
            heat_cap=75.0,
            heat_dissipation=13.0,
            tonnage=20.0,
        ),
        "light": MechClassConfig(
            name="light",
            size_voxels=(1.5, 1.5, 2.0),
            max_speed=6.0,
            max_yaw_rate=1.5,  # Fast but catchable
            max_jet_accel=16.0,
            hp=80.0,
            leg_hp=40.0,
            heat_cap=80.0,
            heat_dissipation=12.0,
            tonnage=35.0,
        ),
        "medium": MechClassConfig(
            name="medium",
            size_voxels=(2.5, 2.5, 3.0),
            max_speed=4.5,
            max_yaw_rate=1.0,  # Moderate
            max_jet_accel=0.0,
            hp=120.0,
            leg_hp=60.0,
            heat_cap=100.0,
            heat_dissipation=11.0,
            tonnage=55.0,
        ),
        "heavy": MechClassConfig(
            name="heavy",
            size_voxels=(3.5, 3.5, 4.0),
            max_speed=3.3,
            max_yaw_rate=0.6,  # Slow turret
            max_jet_accel=0.0,
            hp=200.0,
            leg_hp=100.0,
            heat_cap=130.0,
            heat_dissipation=10.0,
            tonnage=85.0,
        ),
    }


def _team_ids(num_packs: int) -> tuple[list[str], list[str]]:
    total = num_packs * PACK_SIZE
    blue = [f"blue_{i}" for i in range(total)]
    red = [f"red_{i}" for i in range(total)]
    return blue, red


def _roster_for_index(i: int) -> str:
    # Pack structure (10 mechs): 1 Heavy, 5 Medium, 3 Light, 1 Scout
    idx_in_pack = i % PACK_SIZE
    if idx_in_pack < 1:
        return "heavy"
    elif idx_in_pack < 6:  # 1+5 = 6
        return "medium"
    elif idx_in_pack < 9:  # 1+5+3 = 9
        return "light"
    else:
        return "scout"


class EchelonEnv:
    """
    Minimal multi-agent env with a PettingZoo-parallel-like API:

      obs, infos = env.reset(seed)
      obs, rewards, terminations, truncations, infos = env.step(actions)

    actions: continuous float32 vectors with at least the first 9 dims:
      [forward, strafe, vertical, yaw_rate, fire_laser, vent, fire_missile, paint, fire_kinetic]

    Extra control dims are appended (target selection, EWAR toggles, observation-controls, and optional comms).

    Action layout (by default):
    - base (9): movement + weapons
    - target selection (5): argmax selects a contact slot from the last obs
    - EWAR (2): [ECM, ECCM] (scout-only; ECCM wins if both set)
    - obs control (4): sort(3) + hostile-only(1) for the *next* obs
    - comm (comm_dim): pack-local message tail (1 decision-tick delayed)

    Observation-control action fields (always present):
    - sort mode (3 floats): argmax selects {closest, biggest, most_damaged}
    - filter (1 float): >0 selects hostile-only contacts
    """

    CONTACT_SLOTS = 5
    # rel(3) + rel_vel(3) + yaw(2) + hp/heat(2) + stab/fallen/legged(3)
    # + relation_onehot(3) + class_onehot(4) + painted(1) + visible(1) + lead_pitch(1)
    CONTACT_DIM = 25  # Added closing_rate and crossing_angle for targeting prediction

    BASE_ACTION_DIM = ACTION_DIM_CONST
    OBS_SORT_DIM = 3
    OBS_CTRL_DIM = OBS_SORT_DIM + 1  # + hostile-only filter

    # Target selection preferences (argmax selects one of the CONTACT_SLOTS from the last obs).
    TARGET_DIM = CONTACT_SLOTS
    TARGET_START = BASE_ACTION_DIM

    # Observation selection controls (applies to the next returned obs).
    OBS_CTRL_START = TARGET_START + TARGET_DIM

    # Optional pack comm message tail.
    COMM_START = OBS_CTRL_START + OBS_CTRL_DIM

    # Ego-centric local occupancy map (bool footprint) around the mech.
    LOCAL_MAP_R = 5
    LOCAL_MAP_SIZE = 2 * LOCAL_MAP_R + 1
    LOCAL_MAP_DIM = LOCAL_MAP_SIZE * LOCAL_MAP_SIZE

    def __init__(self, config: EnvConfig):
        self.config = config
        self.comm_dim = int(max(0, int(getattr(config, "comm_dim", 0))))
        self.ACTION_DIM = int(self.COMM_START + self.comm_dim)
        self.mech_classes = default_mech_classes()

        self.rng = np.random.default_rng(config.seed)
        self.world: VoxelWorld | None = None
        self.sim: Sim | None = None
        self.last_outcome: dict | None = None
        self._last_reset_seed: int | None = None
        self._spawn_clear: int | None = None

        self.blue_ids, self.red_ids = _team_ids(config.num_packs)
        self.possible_agents = [*self.blue_ids, *self.red_ids]
        self.agents = list(self.possible_agents)

        # Pack-local communication buffers (pack is the deployed unit).
        self._packmates: dict[str, list[str]] = {}
        for ids in (self.blue_ids, self.red_ids):
            packs = max(1, len(ids) // PACK_SIZE)
            for p in range(packs):
                pack_ids = ids[p * PACK_SIZE : (p + 1) * PACK_SIZE]
                for aid in pack_ids:
                    self._packmates[aid] = list(pack_ids)
        self._comm_last: dict[str, np.ndarray] = {}
        if self.comm_dim > 0:
            self._comm_last = {aid: np.zeros(self.comm_dim, dtype=np.float32) for aid in self.possible_agents}

        # Per-mech observation selection controls (applied on the next returned obs).
        self._contact_sort_mode: dict[str, int] = dict.fromkeys(self.possible_agents, 0)  # 0=closest
        self._contact_filter_hostile: dict[str, bool] = dict.fromkeys(self.possible_agents, False)
        # Contact slot identity from the last emitted observation (for target selection).
        self._last_contact_slots: dict[str, list[str | None]] = {
            aid: [None] * self.CONTACT_SLOTS for aid in self.possible_agents
        }

        self.max_steps = math.ceil(config.max_episode_seconds / (config.dt_sim * config.decision_repeat))
        self.step_count = 0
        self._replay: list[dict] | None = None
        self._replay_world: dict | None = None
        self.team_zone_score: dict[str, float] = {"blue": 0.0, "red": 0.0}
        self.zone_score_to_win: float = float(config.max_episode_seconds * 0.5)
        self._max_team_hp: dict[str, float] = {"blue": 0.0, "red": 0.0}
        self._episode_stats: dict[str, float] = {}
        self._prev_fallen: dict[str, bool] = {}
        self._prev_legged: dict[str, bool] = {}
        self._prev_shutdown: dict[str, bool] = {}

        # Navigation Graph (Built once per reset)
        self.nav_graph: NavGraph | None = None
        self._cached_paths: dict[str, list[NodeID]] = {}
        self._path_update_tick: dict[str, int] = {}

        # Performance caches (MED-11, MED-12): Static terrain data computed once at reset
        self._cached_telemetry: np.ndarray | None = None
        self._cached_occupancy_2d: np.ndarray | None = None

        # Gym-compatible space definitions (HIGH-5)
        # These enable compatibility with RL libraries (SB3, RLlib, CleanRL)
        if _HAS_GYMNASIUM and gym_spaces is not None:
            obs_dim = self._obs_dim()
            self.observation_space = gym_spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = gym_spaces.Box(low=-1.0, high=1.0, shape=(self.ACTION_DIM,), dtype=np.float32)
        else:
            # Fallback: store dimensions for manual access
            self.observation_space = None
            self.action_space = None

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            episode_seed = int(seed)
        else:
            episode_seed = int(self.rng.integers(0, 2**31 - 1))
        self._last_reset_seed = episode_seed

        self.step_count = 0
        self.last_outcome = None
        self.team_zone_score = {"blue": 0.0, "red": 0.0}
        self.zone_score_to_win = float(self.config.max_episode_seconds * 0.5)
        self._max_team_hp = {"blue": 0.0, "red": 0.0}
        self._episode_stats = {
            "kills_blue": 0.0,
            "kills_red": 0.0,
            "assists_blue": 0.0,
            "assists_red": 0.0,
            "paints_blue": 0.0,
            "paints_red": 0.0,
            "laser_hits_blue": 0.0,
            "laser_hits_red": 0.0,
            "missile_launches_blue": 0.0,
            "missile_launches_red": 0.0,
            "kinetic_fires_blue": 0.0,
            "kinetic_fires_red": 0.0,
            "ams_intercepts_blue": 0.0,
            "ams_intercepts_red": 0.0,
            "damage_blue": 0.0,
            "damage_red": 0.0,
            "knockdowns_blue": 0.0,
            "knockdowns_red": 0.0,
            "legged_blue": 0.0,
            "legged_red": 0.0,
            "shutdowns_blue": 0.0,
            "shutdowns_red": 0.0,
            "bad_actions_blue": 0.0,
            "bad_actions_red": 0.0,
            "bad_obs_blue": 0.0,
            "bad_obs_red": 0.0,
        }
        self._prev_fallen = {}
        self._prev_legged = {}
        self._prev_shutdown = {}

        self._cached_paths = {}
        self._path_update_tick = {}

        seq = np.random.SeedSequence(episode_seed)
        rng_world, rng_sim, rng_variants = (np.random.default_rng(s) for s in seq.spawn(3))

        world = VoxelWorld.generate(self.config.world, rng_world)

        # Variant choices: choose spawn corner (canonical) and a map reorientation transform.
        blue_canon = str(rng_variants.choice(["BL", "BR", "TL", "TR"]))
        red_canon = opposite_corner(blue_canon)
        transform: TransformType = rng_variants.choice(list_transforms())
        world.voxels = apply_transform_voxels(world.voxels, transform)
        world.meta["transform"] = transform

        # Clear spawn regions in corners.
        # Ensure clear area covers the mech scatter (approx 20m)
        max_hs_x = max(float(spec.size_voxels[0] * 0.5) for spec in self.mech_classes.values())
        max_hs_y = max(float(spec.size_voxels[1] * 0.5) for spec in self.mech_classes.values())
        cols = 5
        n_per_team = len(self.blue_ids)
        cols_used = min(cols, max(1, n_per_team))
        rows = int((n_per_team + cols - 1) // cols)
        required_x = 2.0 + float(max(0, cols_used - 1)) * 3.5 + 2.0 * max_hs_x
        required_y = 2.0 + float(max(0, rows - 1)) * 3.5 + 2.0 * max_hs_y
        required_clear = math.ceil(max(required_x, required_y) + 1.0)
        spawn_clear = max(
            25, int(min(self.config.world.size_x, self.config.world.size_y) * 0.25), required_clear
        )
        spawn_clear = min(spawn_clear, world.size_x, world.size_y)
        self._spawn_clear = spawn_clear

        spawn_corners = {
            "blue": transform_corner(blue_canon, transform),
            "red": transform_corner(red_canon, transform),
        }
        world.meta["spawn_corners"] = dict(spawn_corners)
        world.meta["spawn_clear"] = int(spawn_clear)

        def clear_corner(corner: str) -> None:
            if corner == "BL":
                world.set_box_solid(0, 0, 0, spawn_clear, spawn_clear, world.size_z, VoxelWorld.AIR)
            elif corner == "BR":
                world.set_box_solid(
                    world.size_x - spawn_clear, 0, 0, world.size_x, spawn_clear, world.size_z, VoxelWorld.AIR
                )
            elif corner == "TL":
                world.set_box_solid(
                    0, world.size_y - spawn_clear, 0, spawn_clear, world.size_y, world.size_z, VoxelWorld.AIR
                )
            elif corner == "TR":
                world.set_box_solid(
                    world.size_x - spawn_clear,
                    world.size_y - spawn_clear,
                    0,
                    world.size_x,
                    world.size_y,
                    world.size_z,
                    VoxelWorld.AIR,
                )
            else:
                raise ValueError(f"Unknown corner: {corner!r}")

        clear_corner(spawn_corners["blue"])
        clear_corner(spawn_corners["red"])

        world.meta["capture_zone"] = sample_capture_zone(
            world, rng_variants, spawn_clear=spawn_clear, spawn_corners=spawn_corners
        )
        clear_capture_zone(world, meta=world.meta)

        carve_macro_corridors(
            world,
            spawn_corners=spawn_corners,
            spawn_clear=spawn_clear,
            meta=world.meta,
            rng=rng_variants,
        )

        if self.config.world.ensure_connectivity:
            validator = ConnectivityValidator(
                (world.size_z, world.size_y, world.size_x),
                clearance_z=self.config.world.connectivity_clearance_z,
                obstacle_inflate_radius=self.config.world.connectivity_obstacle_inflate_radius,
                wall_cost=self.config.world.connectivity_wall_cost,
                penalty_radius=self.config.world.connectivity_penalty_radius,
                penalty_cost=self.config.world.connectivity_penalty_cost,
                carve_width=self.config.world.connectivity_carve_width,
            )
            world.voxels = validator.validate_and_fix(
                world.voxels,
                spawn_corners=dict(world.meta.get("spawn_corners", spawn_corners)),
                spawn_clear=spawn_clear,
                meta=world.meta,
            )

        # The simulation has an implicit ground plane at z=0 (below is solid), but for rendering we
        # want a contiguous "dirt" layer on the first voxel layer (z=0), with hazards (water/lava)
        # embedded as patches.
        world.ensure_ground_layer()

        # Build Navigation Graph
        # Use mech_radius=1 for safety (conservative for heavy mechs)
        self.nav_graph = NavGraph.build(
            world,
            clearance_z=self.config.world.connectivity_clearance_z,
            mech_radius=self.config.world.connectivity_obstacle_inflate_radius,
        )

        world.meta["recipe"] = build_recipe(
            generator_id="legacy_voxel_archetypes",
            generator_version="1",
            seed=episode_seed,
            world_config=self.config.world,
            world_meta=world.meta,
            solids_zyx=world.voxels,
        )

        zone_cx, zone_cy, _ = capture_zone_params(world.meta, size_x=world.size_x, size_y=world.size_y)
        mechs: dict[str, MechState] = {}
        for team, ids in (("blue", self.blue_ids), ("red", self.red_ids)):
            for i, mech_id in enumerate(ids):
                cls_name = _roster_for_index(i)
                spec = self.mech_classes[cls_name]
                hs = np.asarray(spec.size_voxels, dtype=np.float32) * 0.5

                corner = spawn_corners[team]
                cols = 5
                if corner in ("BL", "TL"):
                    x = 2.0 + float(i % cols) * 3.5 + float(hs[0])
                else:
                    x = float(world.size_x) - 2.0 - float(i % cols) * 3.5 - float(hs[0])

                if corner in ("BL", "BR"):
                    y = 2.0 + float(i // cols) * 3.5 + float(hs[1])
                else:
                    y = float(world.size_y) - 2.0 - float(i // cols) * 3.5 - float(hs[1])

                # Small spawn jitter improves replay variety and reduces brittle overfitting
                # to a fixed formation grid.
                jitter = 0.25
                x = float(x + float(rng_variants.uniform(-jitter, jitter)))
                y = float(y + float(rng_variants.uniform(-jitter, jitter)))
                x = float(np.clip(x, float(hs[0]), float(world.size_x) - float(hs[0])))
                y = float(np.clip(y, float(hs[1]), float(world.size_y) - float(hs[1])))

                yaw = float(math.atan2(zone_cy - y, zone_cx - x) + float(rng_variants.uniform(-0.15, 0.15)))

                z = float(hs[2])
                pos = np.asarray([x, y, z], dtype=np.float32)
                vel = np.zeros(3, dtype=np.float32)
                mech = MechState(
                    mech_id=mech_id,
                    team=team,
                    spec=spec,
                    pos=pos,
                    vel=vel,
                    yaw=yaw,
                    hp=spec.hp,
                    leg_hp=spec.leg_hp,
                    heat=0.0,
                    stability=100.0,
                )
                mechs[mech_id] = mech
                self._max_team_hp[team] += float(spec.hp)

        sim = Sim(world=world, dt_sim=self.config.dt_sim, rng=rng_sim)
        sim.reset(mechs)

        self.world = world
        self.sim = sim
        self.agents = list(self.possible_agents)

        # Compute static terrain caches (MED-11, MED-12)
        self._compute_terrain_caches(world)

        if self.comm_dim > 0:
            for aid in self._comm_last:
                self._comm_last[aid].fill(0.0)
        for aid in self._contact_sort_mode:
            self._contact_sort_mode[aid] = 0
        for aid in self._contact_filter_hostile:
            self._contact_filter_hostile[aid] = False
        for aid in self._last_contact_slots:
            self._last_contact_slots[aid] = [None] * self.CONTACT_SLOTS

        if self.config.record_replay:
            self._replay = []
            self._replay_world = self._serialize_world(world, seed=episode_seed)
        else:
            self._replay = None
            self._replay_world = None

        # Seed transition trackers.
        for mid, m in sim.mechs.items():
            self._prev_fallen[mid] = bool(m.fallen_time > 0.0)
            self._prev_legged[mid] = bool(m.is_legged)
            self._prev_shutdown[mid] = bool(m.shutdown)

        obs = self._obs()
        infos: dict[str, dict] = {aid: {} for aid in self.agents}
        return obs, infos

    def _compute_terrain_caches(self, world: VoxelWorld) -> None:
        """
        Compute static terrain caches at reset time (MED-11, MED-12).

        - _cached_telemetry: 16x16 downsampled solid terrain map (for satellite view)
        - _cached_occupancy_2d: 2D occupancy grid (Z-collapsed) for local maps
        """
        # MED-12: Compute 2D occupancy once for all agents' local maps
        clearance_z = int(
            max(1, min(getattr(self.config.world, "connectivity_clearance_z", 4), world.size_z))
        )
        solid_slice = world.voxels[:clearance_z, :, :]
        self._cached_occupancy_2d = np.any(
            (solid_slice == VoxelWorld.SOLID) | (solid_slice == VoxelWorld.SOLID_DEBRIS),
            axis=0,
        )

        # MED-11: Compute telemetry (16x16 downsampled map) once
        TELEMETRY_SIZE = 16
        world_2d = np.any(
            (world.voxels == VoxelWorld.SOLID) | (world.voxels == VoxelWorld.SOLID_DEBRIS),
            axis=0,
        )
        sy, sx = world_2d.shape
        y_bins = np.linspace(0, sy, TELEMETRY_SIZE + 1).astype(int)
        x_bins = np.linspace(0, sx, TELEMETRY_SIZE + 1).astype(int)

        telemetry = np.zeros((TELEMETRY_SIZE, TELEMETRY_SIZE), dtype=np.float32)
        for iy in range(TELEMETRY_SIZE):
            for ix in range(TELEMETRY_SIZE):
                region = world_2d[y_bins[iy] : y_bins[iy + 1], x_bins[ix] : x_bins[ix + 1]]
                if region.size > 0 and np.any(region):
                    telemetry[iy, ix] = 1.0
        self._cached_telemetry = telemetry.reshape(-1)

    def team_hp(self) -> dict[str, float]:
        sim = self.sim
        if sim is None:
            return {"blue": 0.0, "red": 0.0}
        hp: dict[str, float] = {"blue": 0.0, "red": 0.0}
        for m in sim.mechs.values():
            hp[m.team] += max(0.0, float(m.hp)) if m.alive else 0.0
        return hp

    def _contact_features(
        self,
        viewer: MechState,
        other: MechState,
        *,
        relation: str,
        painted_by_pack: bool,
    ) -> np.ndarray:
        world = self.world
        assert world is not None

        feat = np.zeros(self.CONTACT_DIM, dtype=np.float32)

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

        cls = other.spec.name
        class_onehot = np.zeros(4, dtype=np.float32)
        if cls == "scout":
            class_onehot[0] = 1.0
        elif cls == "light":
            class_onehot[1] = 1.0
        elif cls == "medium":
            class_onehot[2] = 1.0
        elif cls == "heavy":
            class_onehot[3] = 1.0

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
        # pitch = atan2(dz + drop, dist_xy)
        # Normalized to roughly [-1, 1] range.
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
        # Helps agents predict interception timing
        rel_vec_raw = other.pos - viewer.pos
        rel_vel_raw = other.vel - viewer.vel
        dist = float(np.linalg.norm(rel_vec_raw))
        if dist > 0.1:
            # Rate of change of distance (negative dot product = closing)
            closing = float(-np.dot(rel_vel_raw, rel_vec_raw) / dist)
            feat[23] = float(np.clip(closing / 10.0, -1.0, 1.0))
        else:
            feat[23] = 0.0

        # Crossing angle: 0 = moving parallel to LOS, 1 = moving perpendicular
        # Perpendicular targets are harder to hit with projectiles
        other_speed = float(np.linalg.norm(other.vel))
        if dist > 0.1 and other_speed > 0.5:
            los_dir = rel_vec_raw / dist
            vel_dir = other.vel / other_speed
            cos_angle = float(np.dot(los_dir, vel_dir))
            feat[24] = 1.0 - abs(cos_angle)  # 0 = parallel, 1 = perpendicular
        else:
            feat[24] = 0.0

        return feat

    def _ewar_levels(self, viewer: MechState) -> tuple[float, float, float]:
        """
        Returns (sensor_quality, jam_level, eccm_level).

        - jam_level/eccm_level are in [0, 1] based on strongest nearby sources.
        - sensor_quality is clipped to [SENSOR_QUALITY_MIN, SENSOR_QUALITY_MAX].
        """
        sim = self.sim
        assert sim is not None

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

    def _local_map(self, viewer: MechState) -> np.ndarray:
        world = self.world
        assert world is not None

        r = int(self.LOCAL_MAP_R)
        size = 2 * r + 1

        # MED-12: Use cached occupancy_2d (computed once at reset)
        occupancy_2d = self._cached_occupancy_2d
        assert occupancy_2d is not None

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

    def _obs(self) -> dict[str, np.ndarray]:
        sim = self.sim
        world = self.world
        assert sim is not None and world is not None

        base_radar_range = 14.0
        zone_cx, zone_cy, zone_r = capture_zone_params(world.meta, size_x=world.size_x, size_y=world.size_y)
        max_xy = float(max(world.size_x, world.size_y))
        zone_r_norm = float(np.clip(zone_r / max(1.0, max_xy), 0.0, 1.0))

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
            np.clip(float(sim.time_s) / max(1e-6, float(self.config.max_episode_seconds)), 0.0, 1.0)
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

        # MED-11: Use cached telemetry (computed once at reset)
        telemetry_flat = self._cached_telemetry
        assert telemetry_flat is not None

        reserved = {"friendly": 3, "hostile": 1, "neutral": 1}
        repurpose_priority = ["hostile", "friendly", "neutral"]

        obs: dict[str, np.ndarray] = {}
        for aid in self.agents:
            viewer = sim.mechs[aid]
            if not viewer.alive:
                obs[aid] = np.zeros(self._obs_dim(), dtype=np.float32)
                self._last_contact_slots[aid] = [None] * self.CONTACT_SLOTS
                continue

            pack_ids = self._packmates.get(aid, [])
            sort_mode = int(self._contact_sort_mode.get(aid, 0))
            hostile_only = bool(self._contact_filter_hostile.get(aid, False))

            sensor_quality, jam_level, _eccm_level = self._ewar_levels(viewer)
            radar_range = float(base_radar_range) * float(sensor_quality)

            # Acoustic Sensing: Total intensity from 4 relative quadrants
            # Optimization: could be vectorized across all 'other' mechs.
            acoustic_intensities = np.zeros(4, dtype=np.float32)

            # Pre-calculate positions and noise for all alive mechs once per _obs call if needed,
            # but for now let's just make the inner loop tighter.
            for other_id in self.possible_agents:
                if other_id == aid:
                    continue
                other = sim.mechs[other_id]
                if not other.alive or other.noise_level <= 0:
                    continue

                delta = other.pos - viewer.pos
                dist_sq = float(np.dot(delta, delta))

                # Inverse square law (clamped)
                intensity = other.noise_level / (1.0 + dist_sq)

                # Attenuation by terrain
                if not _cached_los(aid, other_id, viewer.pos, other.pos):
                    intensity *= 0.20  # 80% loss through walls

                # Determine relative quadrant
                # Use atan2 and wrap_pi
                angle = _wrap_pi(math.atan2(float(delta[1]), float(delta[0])) - float(viewer.yaw))
                if angle >= 0:  # Left side
                    q = 0 if angle < 1.570796 else 2  # Front-Left vs Rear-Left
                else:  # Right side
                    q = 1 if angle > -1.570796 else 3  # Front-Right vs Rear-Right
                acoustic_intensities[q] += float(intensity)

            # Normalize acoustic intensities (Log scaling to keep in [0, 1] range roughly)
            acoustic_intensities = (np.log1p(acoustic_intensities) / 5.0).astype(
                np.float32
            )  # 5.0 is approx ln(150)

            # Top-K contact table (fixed size).
            # Structure: (sort_key, dist, other_id, painted_by_pack)
            # sort_key can be tuple[float] | tuple[int, float] | tuple[float, float]
            contacts: dict[str, list[tuple[tuple[float, ...] | tuple[int, float], float, str, bool]]] = {
                "friendly": [],
                "hostile": [],
                "neutral": [],
            }
            for other_id in self.possible_agents:
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

                if self.config.observation_mode == "full":
                    visible = True
                else:
                    smoke_ok = _cached_smoke_los(aid, other_id, viewer.pos, other.pos)
                    painted_visible = painted_by_pack and (sensor_quality >= float(PAINT_LOCK_MIN_QUALITY))

                    # Thermal Bloom: High-heat mechs leaked through terrain, but BLOCKED by smoke
                    heat_bloom = ((other.heat / max(1.0, other.spec.heat_cap)) > 0.80) and smoke_ok

                    # EWAR Emitters: visible through terrain, but BLOCKED by smoke
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
                    cls_rank = {"scout": 1, "light": 1, "medium": 2, "heavy": 3}.get(other.spec.name, 0)
                    key = (-cls_rank, dist)
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

            if hostile_only:
                reserved_local = {"friendly": 0, "hostile": self.CONTACT_SLOTS, "neutral": 0}
                repurpose_local = ["hostile"]
            else:
                reserved_local = reserved
                repurpose_local = repurpose_priority

            # Fill reserved slots by category.
            for rel in ("friendly", "hostile", "neutral"):
                take = min(int(reserved_local[rel]), len(contacts[rel]))
                for i in range(take):
                    _, _, oid, painted = contacts[rel][i]
                    selected.append((rel, oid, painted))
                used_counts[rel] += take

            # Repurpose any unused slots based on priority.
            while len(selected) < self.CONTACT_SLOTS:
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

            contact_feats = np.zeros((self.CONTACT_SLOTS, self.CONTACT_DIM), dtype=np.float32)
            for i, (rel, oid, painted) in enumerate(selected[: self.CONTACT_SLOTS]):
                contact_feats[i, :] = self._contact_features(
                    viewer, sim.mechs[oid], relation=rel, painted_by_pack=painted
                )

            slot_ids: list[str | None] = [None] * self.CONTACT_SLOTS
            for i, (_, oid, _) in enumerate(selected[: self.CONTACT_SLOTS]):
                slot_ids[i] = oid
            self._last_contact_slots[aid] = slot_ids

            parts: list[np.ndarray] = []
            parts.append(contact_feats.reshape(-1))

            # Pack-local comms (1-step delayed); flattened [PACK_SIZE * comm_dim].
            if self.comm_dim > 0:
                comm = np.zeros((PACK_SIZE, self.comm_dim), dtype=np.float32)
                for i, pid in enumerate(pack_ids[:PACK_SIZE]):
                    if not sim.mechs[pid].alive:
                        continue
                    msg = self._comm_last.get(pid)
                    if msg is None:
                        continue
                    comm[i, :] = msg
                parts.append(comm.reshape(-1))

            # Ego-centric local map (footprint occupancy).
            parts.append(self._local_map(viewer))

            # Global satellite telemetry (downsampled 2D solid map).
            parts.append(telemetry_flat)

            # Extra per-agent scalars.
            targeted = 1.0 if self._is_targeted(viewer) else 0.0
            under_fire = 1.0 if viewer.was_hit else 0.0
            painted_flag = 1.0 if viewer.painted_remaining > 0.0 else 0.0
            shutdown = 1.0 if viewer.shutdown else 0.0
            crit_heat = 1.0 if (viewer.heat / max(1.0, viewer.spec.heat_cap)) > 0.85 else 0.0

            # Continuous self-state for heat/damage management (policy needs these to learn
            # proper vent timing and risk assessment, not just binary flags).
            self_hp_norm = float(np.clip(viewer.hp / max(1.0, viewer.spec.hp), 0.0, 1.0))
            self_heat_norm = float(np.clip(viewer.heat / max(1.0, viewer.spec.heat_cap), 0.0, 2.0))
            # Heat headroom: how much heat budget remains before capacity (1.0 = cool, 0.0 = at cap)
            heat_headroom = float(
                np.clip((viewer.spec.heat_cap - viewer.heat) / max(1.0, viewer.spec.heat_cap), 0.0, 1.0)
            )
            # Stability risk: how close to knockdown (0.0 = stable, 1.0 = about to fall)
            stability_risk = float(
                np.clip(1.0 - (viewer.stability / max(1.0, viewer.max_stability)), 0.0, 1.0)
            )

            incoming_missile = 0.0
            for p in sim.projectiles:
                if not getattr(p, "alive", False):
                    continue
                if getattr(p, "weapon", None) != MISSILE.name:
                    continue
                if getattr(p, "guidance", None) != "homing":
                    continue
                if getattr(p, "target_id", None) == viewer.mech_id:
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

            # Self kinematics (needed for jump jets and movement control).
            self_vel = (viewer.vel / 10.0).astype(np.float32, copy=False)

            # Ability cooldowns (normalized to each weapon's actual max cooldown).
            laser_cd = float(np.clip(viewer.laser_cooldown / LASER.cooldown_s, 0.0, 1.0))
            missile_cd = float(np.clip(viewer.missile_cooldown / MISSILE.cooldown_s, 0.0, 1.0))
            kinetic_cd = float(np.clip(viewer.kinetic_cooldown / GAUSS.cooldown_s, 0.0, 1.0))
            painter_cd = float(np.clip(viewer.painter_cooldown / PAINTER.cooldown_s, 0.0, 1.0))

            # Objective features (zone vector is in world frame).
            in_zone = 0.0
            dist_to_zone = float(math.hypot(float(viewer.pos[0] - zone_cx), float(viewer.pos[1] - zone_cy)))
            if dist_to_zone < zone_r:
                in_zone = 1.0
            zone_rel = np.array([zone_cx, zone_cy, 0.0], dtype=np.float32) - viewer.pos
            zone_rel /= max(1.0, max_xy)

            # Team-relative objective score/control (so a policy can play as either team).
            my_team = viewer.team
            enemy_team = "red" if my_team == "blue" else "blue"
            my_control = zone_control if my_team == "blue" else -zone_control
            my_score = float(self.team_zone_score.get(my_team, 0.0))
            enemy_score = float(self.team_zone_score.get(enemy_team, 0.0))
            win_scale = max(1e-6, float(self.zone_score_to_win))
            my_score_norm = float(np.clip(my_score / win_scale, 0.0, 1.0))
            enemy_score_norm = float(np.clip(enemy_score / win_scale, 0.0, 1.0))

            # Hull Type One-Hot: [scout, light, medium, heavy]
            hull_map = {"scout": 0, "light": 1, "medium": 2, "heavy": 3}
            hull_type = np.zeros(4, dtype=np.float32)
            hull_idx = hull_map.get(viewer.spec.name)
            if hull_idx is not None:
                hull_type[hull_idx] = 1.0

            parts.append(acoustic_intensities)
            parts.append(hull_type)

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
                self._episode_stats[f"bad_obs_{viewer.team}"] = float(
                    self._episode_stats.get(f"bad_obs_{viewer.team}", 0.0) + 1.0
                )
            obs[aid] = vec
        return obs

    def _obs_dim(self) -> int:
        comm_dim = PACK_SIZE * int(max(0, int(getattr(self.config, "comm_dim", 0))))
        # self features =
        #   acoustic_quadrants(4) + hull_type_onehot(4) +
        #   targeted, under_fire, painted, shutdown, crit_heat, self_hp_norm, self_heat_norm,
        #   heat_headroom, stability_risk, incoming_missile, sensor_quality, jam_level,
        #   ecm_on, eccm_on, suppressed, ams_cd, self_vel(3), cooldowns(4), in_zone,
        #   vec_to_zone(3), zone_radius, my_control, my_score, enemy_score, time_frac,
        #   obs_sort_onehot(3), hostile_only = 44
        self_dim = 44
        telemetry_dim = 16 * 16
        return (
            self.CONTACT_SLOTS * self.CONTACT_DIM
            + comm_dim
            + int(self.LOCAL_MAP_DIM)
            + telemetry_dim
            + self_dim
        )

    def _is_targeted(self, mech: MechState) -> bool:
        sim = self.sim
        world = self.world
        assert sim is not None and world is not None

        # Simple "someone has LOS to me and is in range".
        # HIGH-12: Use spatial grid to query only nearby enemies
        return any(sim.has_los(other.pos, mech.pos) for other in sim.enemies_in_range(mech, 8.0))

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        sim = self.sim
        assert sim is not None

        self.step_count += 1

        # Normalize and fill missing actions.
        act: dict[str, np.ndarray] = {}
        for aid in self.agents:
            a = actions.get(aid)
            if a is None:
                a = np.zeros(self.ACTION_DIM, dtype=np.float32)
            else:
                a = np.asarray(a, dtype=np.float32)
                if a.size != self.ACTION_DIM:
                    raise ValueError(
                        f"action[{aid!r}] has size {a.size}, expected {self.ACTION_DIM} "
                        f"(base={self.BASE_ACTION_DIM}, target={self.TARGET_DIM}, "
                        f"obs_ctrl={self.OBS_CTRL_DIM}, comm_dim={self.comm_dim})"
                    )
                a = a.reshape(self.ACTION_DIM)

            bad = bool(not np.all(np.isfinite(a)))
            if bad:
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            a = np.asarray(a, dtype=np.float32)
            np.clip(a, -1.0, 1.0, out=a)
            act[aid] = a

            if bad:
                m = sim.mechs.get(aid)
                if m is not None:
                    key = f"bad_actions_{m.team}"
                    self._episode_stats[key] = float(self._episode_stats.get(key, 0.0) + 1.0)

        # Apply target selection + EWAR toggles immediately (affects this sim step).
        for aid in self.agents:
            m = sim.mechs[aid]
            a = act[aid]

            if not m.alive:
                m.focus_target_id = None
                m.ecm_on = False
                m.eccm_on = False
                continue

            if bool(getattr(self.config, "enable_target_selection", True)):
                # Target selection is an argmax over the last-observed contact slots.
                prefs = a[self.TARGET_START : self.TARGET_START + self.TARGET_DIM]
                focus_id: str | None = None
                if prefs.size:
                    i = int(np.argmax(prefs))
                    if float(prefs[i]) > 0.0:
                        slots = self._last_contact_slots.get(aid) or []
                        if i < len(slots):
                            cand = slots[i]
                            if cand is not None:
                                tgt = sim.mechs.get(cand)
                                if tgt is not None and tgt.alive and tgt.team != m.team:
                                    focus_id = str(cand)
                m.focus_target_id = focus_id
            else:
                m.focus_target_id = None

            if bool(getattr(self.config, "enable_ewar", True)):
                # Scout: SECONDARY slot toggles EWAR.
                # > 0.5: ECCM, (0.0, 0.5]: ECM, <= 0: Off
                ewar_val = float(a[ActionIndex.SECONDARY])
                if m.spec.name == "scout":
                    if ewar_val > 0.5:
                        m.eccm_on = True
                        m.ecm_on = False
                    elif ewar_val > 0.0:
                        m.ecm_on = True
                        m.eccm_on = False
                    else:
                        m.ecm_on = False
                        m.eccm_on = False
                else:
                    m.ecm_on = False
                    m.eccm_on = False
            else:
                m.ecm_on = False
                m.eccm_on = False

        # Update per-mech observation controls from the chosen action (applies to returned obs).
        if bool(getattr(self.config, "enable_obs_control", True)):
            for aid in self.agents:
                a = act[aid]
                sort_pref = a[self.OBS_CTRL_START : self.OBS_CTRL_START + self.OBS_SORT_DIM]
                self._contact_sort_mode[aid] = int(np.argmax(sort_pref))
                self._contact_filter_hostile[aid] = bool(
                    float(a[self.OBS_CTRL_START + self.OBS_SORT_DIM]) > 0.0
                )

        # Baselines for rewards/shaping.
        dt_act = float(self.config.dt_sim * self.config.decision_repeat)
        zone_cx, zone_cy, zone_r = capture_zone_params(
            sim.world.meta, size_x=sim.world.size_x, size_y=sim.world.size_y
        )
        max_xy = float(max(sim.world.size_x, sim.world.size_y))

        dist_to_zone_before: dict[str, float] = {}
        hp_before_by_agent: dict[str, float] = {}
        for aid in self.agents:
            m = sim.mechs[aid]
            if not m.alive:
                continue
            dist_to_zone_before[aid] = float(math.hypot(float(m.pos[0] - zone_cx), float(m.pos[1] - zone_cy)))
            hp_before_by_agent[aid] = max(0.0, float(m.hp))

        self.team_hp()

        # Nav-Assist: Post-process actions using NavGraph
        if self.config.nav_mode == "assist" and self.nav_graph:
            planner = Planner(self.nav_graph)
            for aid in self.agents:
                m = sim.mechs[aid]
                if not m.alive or m.shutdown:
                    continue

                # Goal: Capture Zone Center
                goal_pos = (float(zone_cx), float(zone_cy), 0.0)

                # Periodically update path (every 10 decision steps)
                if aid not in self._cached_paths or (self.step_count % 10 == 0):
                    start_node = self.nav_graph.get_nearest_node(tuple(m.pos))
                    goal_node = self.nav_graph.get_nearest_node(goal_pos)
                    if start_node and goal_node:
                        found_path, pstats = planner.find_path(start_node, goal_node)
                        if pstats.found:
                            self._cached_paths[aid] = found_path

                # If we have a path, nudge the RL action towards it
                cached_path = self._cached_paths.get(aid)
                if cached_path is not None and len(cached_path) > 1:
                    # Find a waypoint ~10m ahead on the path
                    lookahead = 2
                    target_node_id = cached_path[min(lookahead, len(cached_path) - 1)]
                    target_node = self.nav_graph.nodes[target_node_id]

                    # Vector to waypoint (XY plane)
                    dx = target_node.pos[0] - m.pos[0]
                    dy = target_node.pos[1] - m.pos[1]
                    dist = math.hypot(dx, dy)

                    if dist > 1.0:
                        # Normalize desired direction
                        dir_x, dir_y = dx / dist, dy / dist

                        # Project into mech's local frame
                        c, s = math.cos(m.yaw), math.sin(m.yaw)
                        # local_fwd = world_x * cos + world_y * sin
                        # local_side = -world_x * sin + world_y * cos
                        fwd_desired = dir_x * c + dir_y * s
                        side_desired = -dir_x * s + dir_y * c

                        # Apply a nudge (30% influence)
                        alpha = 0.3
                        a = act[aid]
                        a[ActionIndex.FORWARD] = (1.0 - alpha) * a[ActionIndex.FORWARD] + alpha * fwd_desired
                        a[ActionIndex.STRAFE] = (1.0 - alpha) * a[ActionIndex.STRAFE] + alpha * side_desired

                        # Re-clip to be safe
                        np.clip(a, -1.0, 1.0, out=a)

        events = sim.step(act, num_substeps=self.config.decision_repeat)

        # Per-agent combat event tracking for reward shaping (HIGH-6)
        step_damage_dealt: dict[str, float] = dict.fromkeys(self.agents, 0.0)
        step_damage_received: dict[str, float] = dict.fromkeys(self.agents, 0.0)
        step_kills: dict[str, int] = dict.fromkeys(self.agents, 0)
        step_assists: dict[str, int] = dict.fromkeys(self.agents, 0)
        step_deaths: dict[str, bool] = dict.fromkeys(self.agents, False)

        # Episode stats (for logging/debugging) and per-agent combat tracking (HIGH-6).
        if events:
            for ev in events:
                et = ev.get("type")
                if et == "kill":
                    shooter_id = str(ev.get("shooter"))
                    victim_id = str(ev.get("victim", ""))
                    shooter = sim.mechs.get(shooter_id)
                    if shooter is not None:
                        self._episode_stats[f"kills_{shooter.team}"] = float(
                            self._episode_stats.get(f"kills_{shooter.team}", 0.0) + 1.0
                        )
                        step_kills[shooter_id] = step_kills.get(shooter_id, 0) + 1
                    if victim_id in step_deaths:
                        step_deaths[victim_id] = True
                elif et == "assist":
                    painter_id = str(ev.get("painter"))
                    painter = sim.mechs.get(painter_id)
                    if painter is not None:
                        self._episode_stats[f"assists_{painter.team}"] = float(
                            self._episode_stats.get(f"assists_{painter.team}", 0.0) + 1.0
                        )
                        step_assists[painter_id] = step_assists.get(painter_id, 0) + 1
                elif et == "paint":
                    shooter = sim.mechs.get(str(ev.get("shooter")))
                    if shooter is not None:
                        self._episode_stats[f"paints_{shooter.team}"] = float(
                            self._episode_stats.get(f"paints_{shooter.team}", 0.0) + 1.0
                        )
                elif et == "laser_hit":
                    shooter_id = str(ev.get("shooter"))
                    target_id = str(ev.get("target", ""))
                    shooter = sim.mechs.get(shooter_id)
                    if shooter is not None:
                        self._episode_stats[f"laser_hits_{shooter.team}"] = float(
                            self._episode_stats.get(f"laser_hits_{shooter.team}", 0.0) + 1.0
                        )
                        dmg = float(ev.get("damage", 0.0) or 0.0)
                        self._episode_stats[f"damage_{shooter.team}"] = float(
                            self._episode_stats.get(f"damage_{shooter.team}", 0.0) + dmg
                        )
                        step_damage_dealt[shooter_id] = step_damage_dealt.get(shooter_id, 0.0) + dmg
                        if target_id in step_damage_received:
                            step_damage_received[target_id] = step_damage_received.get(target_id, 0.0) + dmg
                elif et == "projectile_hit":
                    shooter_id = str(ev.get("shooter"))
                    target_id = str(ev.get("target", ""))
                    shooter = sim.mechs.get(shooter_id)
                    if shooter is not None:
                        dmg = float(ev.get("damage", 0.0) or 0.0)
                        self._episode_stats[f"damage_{shooter.team}"] = float(
                            self._episode_stats.get(f"damage_{shooter.team}", 0.0) + dmg
                        )
                        step_damage_dealt[shooter_id] = step_damage_dealt.get(shooter_id, 0.0) + dmg
                        if target_id in step_damage_received:
                            step_damage_received[target_id] = step_damage_received.get(target_id, 0.0) + dmg
                elif et == "missile_hit":
                    shooter_id = str(ev.get("shooter"))
                    target_id = str(ev.get("target", ""))
                    shooter = sim.mechs.get(shooter_id)
                    if shooter is not None:
                        dmg = float(ev.get("damage", 0.0) or 0.0)
                        step_damage_dealt[shooter_id] = step_damage_dealt.get(shooter_id, 0.0) + dmg
                        if target_id in step_damage_received:
                            step_damage_received[target_id] = step_damage_received.get(target_id, 0.0) + dmg
                elif et == "missile_launch":
                    shooter = sim.mechs.get(str(ev.get("shooter")))
                    if shooter is not None:
                        self._episode_stats[f"missile_launches_{shooter.team}"] = float(
                            self._episode_stats.get(f"missile_launches_{shooter.team}", 0.0) + 1.0
                        )
                elif et == "kinetic_fire":
                    shooter = sim.mechs.get(str(ev.get("shooter")))
                    if shooter is not None:
                        self._episode_stats[f"kinetic_fires_{shooter.team}"] = float(
                            self._episode_stats.get(f"kinetic_fires_{shooter.team}", 0.0) + 1.0
                        )
                elif et == "ams_intercept":
                    defender = sim.mechs.get(str(ev.get("defender")))
                    if defender is not None:
                        self._episode_stats[f"ams_intercepts_{defender.team}"] = float(
                            self._episode_stats.get(f"ams_intercepts_{defender.team}", 0.0) + 1.0
                        )

        # Transition-derived episode stats.
        for mid, m in sim.mechs.items():
            now_fallen = bool(m.fallen_time > 0.0)
            if now_fallen and (not self._prev_fallen.get(mid, False)):
                self._episode_stats[f"knockdowns_{m.team}"] = float(
                    self._episode_stats.get(f"knockdowns_{m.team}", 0.0) + 1.0
                )
            self._prev_fallen[mid] = now_fallen

            now_legged = bool(m.is_legged)
            if now_legged and (not self._prev_legged.get(mid, False)):
                self._episode_stats[f"legged_{m.team}"] = float(
                    self._episode_stats.get(f"legged_{m.team}", 0.0) + 1.0
                )
            self._prev_legged[mid] = now_legged

            now_shutdown = bool(m.shutdown)
            if now_shutdown and (not self._prev_shutdown.get(mid, False)):
                self._episode_stats[f"shutdowns_{m.team}"] = float(
                    self._episode_stats.get(f"shutdowns_{m.team}", 0.0) + 1.0
                )
            self._prev_shutdown[mid] = now_shutdown

        # Update comm buffers after the sim step (dead mechs do not broadcast).
        if self.comm_dim > 0:
            if not bool(getattr(self.config, "enable_comm", True)):
                for aid in self.agents:
                    self._comm_last[aid].fill(0.0)
            else:
                for aid in self.agents:
                    m = sim.mechs[aid]
                    if not m.alive:
                        self._comm_last[aid].fill(0.0)
                        continue
                    msg = act[aid][self.COMM_START : self.COMM_START + self.comm_dim]
                    self._comm_last[aid] = np.clip(msg, -1.0, 1.0).astype(np.float32, copy=False)

        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        # Reward encodes desired behaviors:
        # 1) Move toward the objective (dense shaping).
        # 2) Control the objective (tonnage-based ratio reward).
        # nothing else.
        in_zone_tonnage: dict[str, float] = {"blue": 0.0, "red": 0.0}
        in_zone_by_agent: dict[str, bool] = {}
        dist_to_zone_after: dict[str, float] = {}

        for aid in self.agents:
            m = sim.mechs[aid]
            alive = bool(m.alive)
            terminations[aid] = bool(not alive)
            truncations[aid] = False
            infos[aid] = {"events": [], "alive": alive}

            in_zone = False
            if alive:
                dist = float(math.hypot(float(m.pos[0] - zone_cx), float(m.pos[1] - zone_cy)))
                dist_to_zone_after[aid] = dist
                in_zone = dist < zone_r
                if in_zone:
                    in_zone_tonnage[m.team] += float(m.spec.tonnage)
            in_zone_by_agent[aid] = in_zone

        # Territory scoring: whichever team has more tonnage in the zone gains score.
        # Ratio-based: if contested, reward is (my_tonnage / total_tonnage).
        total_tonnage = in_zone_tonnage["blue"] + in_zone_tonnage["red"]
        if total_tonnage > 0:
            blue_tick = in_zone_tonnage["blue"] / total_tonnage
            red_tick = in_zone_tonnage["red"] / total_tonnage
        else:
            blue_tick = 0.0
            red_tick = 0.0

        # Update scores (using ticks as progress towards winning)
        self.team_zone_score["blue"] = float(self.team_zone_score["blue"] + blue_tick * dt_act)
        self.team_zone_score["red"] = float(self.team_zone_score["red"] + red_tick * dt_act)

        # Reward weights (HIGH-6: added combat shaping)
        W_ZONE_TICK = 0.15  # Bumped: being IN zone should be more valuable than approaching
        W_APPROACH = 0.05  # Reduced: was causing agents to hover near zone entrance
        W_DAMAGE = 0.005  # Per point of damage dealt
        W_KILL = 1.0  # Per kill
        W_ASSIST = 0.5  # Per assist
        W_DEATH = -0.5  # Death penalty (negative)

        # Count teammates in zone for scaled breadcrumb decay (HIGH-7)
        teammates_in_zone: dict[str, int] = {"blue": 0, "red": 0}
        for aid2, in_z in in_zone_by_agent.items():
            if in_z:
                m2 = sim.mechs.get(aid2)
                if m2 is not None:
                    teammates_in_zone[m2.team] += 1

        for aid in self.agents:
            m = sim.mechs[aid]

            # Dead agents get 0 after the death step (terminal rewards handled at episode end - MED-3).
            if not (m.alive or m.died):
                rewards[aid] = 0.0
                continue

            r = 0.0

            # (1) Move toward objective: potential-style shaping on distance to zone center.
            # HIGH-7: Use exponential decay based on teammates in zone instead of binary cutoff.
            # More teammates in zone = less approach reward (encourages spreading out).
            n_teammates = teammates_in_zone.get(m.team, 0)
            approach_scale = 0.5**n_teammates  # 1.0 if none, 0.5 if 1, 0.25 if 2, etc.
            if not in_zone_by_agent.get(aid, False):  # Only give approach reward if not already in zone
                d0 = dist_to_zone_before.get(aid)
                d1 = dist_to_zone_after.get(aid)
                if d0 is not None and d1 is not None and max_xy > 0.0:
                    phi0 = -float(d0 / max_xy)
                    phi1 = -float(d1 / max_xy)
                    r += W_APPROACH * (phi1 - phi0) * approach_scale

            # (2) Zone Control Reward: ratio of tonnage.
            team_tick = blue_tick if m.team == "blue" else red_tick
            r += W_ZONE_TICK * team_tick

            # (3) Combat shaping rewards (HIGH-6)
            r += W_DAMAGE * step_damage_dealt.get(aid, 0.0)
            r += W_KILL * step_kills.get(aid, 0)
            r += W_ASSIST * step_assists.get(aid, 0)
            if step_deaths.get(aid, False):
                r += W_DEATH

            rewards[aid] = float(r)

        # Episode end conditions (King of the Hill is the primary win condition).
        hp_after = self.team_hp()
        control = blue_tick - red_tick

        blue_alive = sim.team_alive("blue")
        red_alive = sim.team_alive("red")
        time_up = self.step_count >= self.max_steps

        winner: str | None = None
        reason: str | None = None

        if (
            self.team_zone_score["blue"] >= self.zone_score_to_win
            or self.team_zone_score["red"] >= self.zone_score_to_win
        ):
            winner = "blue" if self.team_zone_score["blue"] > self.team_zone_score["red"] else "red"
            reason = "zone_control"
            # MED-2: Zone win is a natural termination (objective achieved), not a truncation
            for aid in terminations:
                terminations[aid] = True
        elif (not blue_alive) or (not red_alive):
            if blue_alive and not red_alive:
                winner = "blue"
            elif red_alive and not blue_alive:
                winner = "red"
            else:
                winner = "draw"
            reason = "elimination"
            for aid in terminations:
                terminations[aid] = True
        elif time_up:
            for aid in truncations:
                truncations[aid] = True
            eps = 1e-3
            if self.team_zone_score["blue"] > self.team_zone_score["red"] + eps:
                winner = "blue"
            elif self.team_zone_score["red"] > self.team_zone_score["blue"] + eps:
                winner = "red"
            else:
                # Fallback tiebreaker: remaining team HP.
                hp = hp_after
                if hp["blue"] > hp["red"] + eps:
                    winner = "blue"
                elif hp["red"] > hp["blue"] + eps:
                    winner = "red"
                else:
                    winner = "draw"
            reason = "time_up"

        # Attach score/state to infos.
        for aid in infos:
            infos[aid]["zone_score"] = dict(self.team_zone_score)
            infos[aid]["zone_control"] = float(control)
            infos[aid]["in_zone"] = bool(in_zone_by_agent.get(aid, False))

        if reason is not None:
            self.last_outcome = {
                "reason": reason,
                "winner": winner or "draw",
                "hp": hp_after,
                "zone_score": dict(self.team_zone_score),
                "zone_score_to_win": float(self.zone_score_to_win),
                "stats": dict(self._episode_stats),
            }

            # MED-3: Terminal reward distribution for all agents (including dead ones).
            # Win/loss signal helps dead agents learn what led to the outcome.
            W_WIN = 5.0
            W_LOSE = -5.0
            W_DRAW = 0.0
            for aid in self.agents:
                m = sim.mechs[aid]
                if winner == "draw":
                    terminal_r = W_DRAW
                elif winner == m.team:
                    terminal_r = W_WIN
                else:
                    terminal_r = W_LOSE
                rewards[aid] = float(rewards.get(aid, 0.0) + terminal_r)

        # Attach aggregated events to all infos (for now).
        if events:
            for aid in infos:
                infos[aid]["events"] = events
        if self.last_outcome is not None:
            for aid in infos:
                infos[aid]["outcome"] = self.last_outcome

        if self._replay is not None:
            self._replay.append(self._replay_frame(events))

        obs = self._obs()
        return obs, rewards, terminations, truncations, infos

    def _replay_frame(self, events: list[dict]) -> dict:
        sim = self.sim
        world = self.world
        assert sim is not None
        assert world is not None
        zone_cx, zone_cy, zone_r = capture_zone_params(world.meta, size_x=world.size_x, size_y=world.size_y)
        return {
            "t": float(sim.time_s),
            "tick": int(sim.tick),
            "objective": {
                "zone_center": [float(zone_cx), float(zone_cy)],
                "zone_radius": float(zone_r),
                "zone_score": dict(self.team_zone_score),
                "zone_score_to_win": float(self.zone_score_to_win),
            },
            "mechs": {
                mid: {
                    "team": m.team,
                    "class": m.spec.name,
                    "pos": [float(x) for x in m.pos],
                    "vel": [float(v) for v in m.vel],
                    "yaw": float(m.yaw),
                    "hp": float(m.hp),
                    "hp_max": float(m.spec.hp),
                    "heat": float(m.heat),
                    "heat_cap": float(m.spec.heat_cap),
                    "stability": float(m.stability),
                    "stability_max": float(m.max_stability),
                    "alive": bool(m.alive),
                    "fallen": bool(m.fallen_time > 0.0),
                    "legged": bool(m.is_legged),
                    "suppressed_time": float(m.suppressed_time),
                    "ams_cooldown": float(m.ams_cooldown),
                    "ecm_on": bool(m.ecm_on and (not m.shutdown)),
                    "eccm_on": bool(m.eccm_on and (not m.shutdown)),
                }
                for mid, m in sim.mechs.items()
            },
            "smoke_clouds": [
                {"pos": [float(c.pos[0]), float(c.pos[1]), float(c.pos[2])], "radius": float(c.radius)}
                for c in sim.smoke_clouds
                if c.alive
            ],
            "events": events,
        }

    def get_replay(self) -> dict | None:
        if self._replay is None:
            return None

        # Use the world snapshot captured at reset so replays stay consistent
        # even when the sim mutates voxels (e.g., death debris).
        world_data = self._replay_world
        if world_data is None and self.world is not None:
            world_data = self._serialize_world(self.world, seed=self._last_reset_seed)

        return {"world": world_data or {}, "frames": self._replay}

    def _serialize_world(self, world: VoxelWorld, *, seed: int | None) -> dict:
        world_data: dict[str, object] = {
            "size": [world.size_x, world.size_y, world.size_z],
            "voxel_size_m": float(world.voxel_size_m),
        }
        if seed is not None:
            world_data["seed"] = int(seed)
        if self._spawn_clear is not None:
            world_data["spawn_clear"] = int(self._spawn_clear)
        if getattr(world, "meta", None):
            world_data["meta"] = dict(world.meta)

        idx_zyx = np.argwhere(world.voxels > 0)
        if idx_zyx.size:
            walls: list[list[int]] = []
            for z, y, x in idx_zyx:
                walls.append([int(x), int(y), int(z), int(world.voxels[z, y, x])])
            world_data["walls"] = walls
        else:
            world_data["walls"] = []

        return world_data
