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
from ..actions import (
    ORDER_PARAM_DIM,
    ORDER_RECIPIENT_DIM,
    ORDER_TYPE_DIM,
    STATUS_ACKNOWLEDGE_DIM,
    STATUS_FLAGS_DIM,
    STATUS_OVERRIDE_DIM,
    STATUS_PROGRESS_DIM,
    ActionIndex,
    OrderType,
)
from ..config import (
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
from ..rl.suite import CombatSuiteSpec, get_suite_for_role
from ..sim.mech import MechState, ReceivedOrder
from ..sim.sim import Sim
from ..sim.world import VoxelWorld
from .observations import ObservationBuilder, ObservationContext
from .rewards import RewardComputer, RewardWeights, StepContext, compute_zone_ticks


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
            max_speed=6.8,  # Was 6.0 - faster to survive and contribute
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
            max_speed=5.5,  # Was 4.5 - faster to keep up and contribute
            max_yaw_rate=1.0,  # Moderate
            max_jet_accel=0.0,
            hp=150.0,  # Was 120 - more armor for frontline durability
            leg_hp=75.0,  # Was 60 - proportional increase
            heat_cap=100.0,
            heat_dissipation=11.0,
            tonnage=55.0,
        ),
        "heavy": MechClassConfig(
            name="heavy",
            size_voxels=(3.5, 3.5, 4.0),
            max_speed=2.5,  # Was 3.3 - slower so lights can race to zone
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
    """Generate mech IDs for each team.

    With 2+ packs, adds a squad leader at the end (index = num_packs * PACK_SIZE).
    """
    pack_total = num_packs * PACK_SIZE
    # Add squad leader when we have a full squad (2+ packs)
    total = pack_total + 1 if num_packs >= 2 else pack_total
    blue = [f"blue_{i}" for i in range(total)]
    red = [f"red_{i}" for i in range(total)]
    return blue, red


def _roster_for_index(i: int, num_packs: int) -> str:
    """Assign mech class based on position in pack/squad.

    Pack structure (6 mechs):
      - idx 0: Scout (recon, painting)
      - idx 1: Light (flanker)
      - idx 2, 3: Medium (line infantry)
      - idx 4: Heavy (fire support)
      - idx 5: Pack Leader (light chassis, command suite)

    Squad structure (13 mechs):
      - Pack 0: indices 0-5
      - Pack 1: indices 6-11
      - Squad Leader: index 12 (medium chassis, squad command suite)
    """
    from ..constants import (
        PACK_HEAVY_IDX,
        PACK_LEADER_IDX,
        PACK_LIGHT_IDX,
        PACK_MEDIUM_A_IDX,
        PACK_MEDIUM_B_IDX,
        PACK_SCOUT_IDX,
    )

    total_in_packs = num_packs * PACK_SIZE

    # Squad leader is the last mech when we have 2+ packs
    if num_packs >= 2 and i == total_in_packs:
        return "medium"  # Squad leader chassis

    idx_in_pack = i % PACK_SIZE

    if idx_in_pack == PACK_SCOUT_IDX:
        return "scout"
    elif idx_in_pack == PACK_LIGHT_IDX:
        return "light"
    elif idx_in_pack in (PACK_MEDIUM_A_IDX, PACK_MEDIUM_B_IDX):
        return "medium"
    elif idx_in_pack == PACK_HEAVY_IDX:
        return "heavy"
    elif idx_in_pack == PACK_LEADER_IDX:
        return "light"  # Pack leader chassis
    else:
        return "medium"  # Fallback


def _command_role_for_index(i: int, num_packs: int) -> str | None:
    """Return command role if this index is a command mech, else None.

    Returns:
        "pack_leader" - Can issue orders to pack members, sees pack sensors
        "squad_leader" - Can issue orders to anyone, sees full squad telemetry
        None - Regular line mech
    """
    from ..constants import PACK_LEADER_IDX

    total_in_packs = num_packs * PACK_SIZE

    # Squad leader (when we have 2+ packs)
    if num_packs >= 2 and i == total_in_packs:
        return "squad_leader"

    # Pack leader (last position in each pack)
    if i % PACK_SIZE == PACK_LEADER_IDX:
        return "pack_leader"

    return None


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

    # Maximum contact slots (global cap for array allocation).
    # Each suite fills a subset based on visual_contact_slots.
    # Scout=20, Light=10, Medium=8, Heavy=5.
    MAX_CONTACT_SLOTS = 20
    CONTACT_SLOTS = MAX_CONTACT_SLOTS  # Alias for backwards compatibility

    # rel(3) + rel_vel(3) + yaw(2) + hp/heat(2) + stab/fallen/legged(3)
    # + relation_onehot(3) + class_onehot(4) + painted(1) + visible(1) + lead_pitch(1)
    # + closing_rate(1) + crossing_angle(1) = 25
    CONTACT_DIM = 25

    # Received order observation dimensions
    # order_type(6) + time(1) + target_pos(3) + progress(1) + override(6) + flags(3) = 20
    ORDER_OBS_DIM = 20

    # Panel stats dimensions (compensates for mean pooling losing count information)
    # contact_count + intel + order + squad + threat + friendly + alert + detail = 8
    PANEL_STATS_DIM = 8

    BASE_ACTION_DIM = ACTION_DIM_CONST
    OBS_SORT_DIM = 3
    OBS_CTRL_DIM = OBS_SORT_DIM + 1  # + hostile-only filter

    # Target selection preferences (argmax selects one of the MAX_CONTACT_SLOTS from the last obs).
    TARGET_DIM = MAX_CONTACT_SLOTS
    TARGET_START = BASE_ACTION_DIM

    # Observation selection controls (applies to the next returned obs).
    OBS_CTRL_START = TARGET_START + TARGET_DIM

    # Optional pack comm message tail.
    COMM_START = OBS_CTRL_START + OBS_CTRL_DIM

    # Command actions (for pack/squad leaders to issue orders).
    # Layout: [order_type(6), recipient(6), param(5)] = 17 dims
    # Non-command mechs ignore these dimensions.
    CMD_ORDER_TYPE_START = -1  # Set in __init__ based on comm_dim
    CMD_RECIPIENT_START = -1
    CMD_PARAM_START = -1

    # Status report actions (for subordinates reporting to command).
    # Layout: [acknowledge(1), override(6), progress(1), flags(3)] = 11 dims
    # Command mechs ignore these dimensions.
    STATUS_ACK_START = -1  # Set in __init__
    STATUS_OVERRIDE_START = -1
    STATUS_PROGRESS_START = -1
    STATUS_FLAGS_START = -1

    # Ego-centric local occupancy map (bool footprint) around the mech.
    LOCAL_MAP_R = 5
    LOCAL_MAP_SIZE = 2 * LOCAL_MAP_R + 1
    LOCAL_MAP_DIM = LOCAL_MAP_SIZE * LOCAL_MAP_SIZE

    def __init__(self, config: EnvConfig):
        self.config = config
        self.num_packs = config.num_packs
        self.comm_dim = int(max(0, config.comm_dim))

        # Command action layout comes after comm
        self.CMD_ORDER_TYPE_START = self.COMM_START + self.comm_dim
        self.CMD_RECIPIENT_START = self.CMD_ORDER_TYPE_START + ORDER_TYPE_DIM
        self.CMD_PARAM_START = self.CMD_RECIPIENT_START + ORDER_RECIPIENT_DIM

        # Status report action layout comes after command actions
        self.STATUS_ACK_START = self.CMD_PARAM_START + ORDER_PARAM_DIM
        self.STATUS_OVERRIDE_START = self.STATUS_ACK_START + STATUS_ACKNOWLEDGE_DIM
        self.STATUS_PROGRESS_START = self.STATUS_OVERRIDE_START + STATUS_OVERRIDE_DIM
        self.STATUS_FLAGS_START = self.STATUS_PROGRESS_START + STATUS_PROGRESS_DIM

        # Total action dimension includes command + status report actions
        self.ACTION_DIM = int(self.STATUS_FLAGS_START + STATUS_FLAGS_DIM)
        self.mech_classes = default_mech_classes()

        self.rng = np.random.default_rng(config.seed)
        self.world: VoxelWorld | None = None
        self.sim: Sim | None = None
        self.last_outcome: dict | None = None
        self._last_reset_seed: int | None = None
        self._spawn_clear: int | None = None

        self.blue_ids, self.red_ids = _team_ids(self.num_packs)
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
        self._ever_in_zone: dict[str, bool] = {}  # Per-agent tracking for arrival bonus

        # Navigation Graph (Built once per reset)
        self.nav_graph: NavGraph | None = None
        self._cached_paths: dict[str, list[NodeID]] = {}
        self._path_update_tick: dict[str, int] = {}

        # Combat Suite assignments (determines observation richness per mech)
        self._mech_suites: dict[str, CombatSuiteSpec] = {}

        # Command hierarchy: maps command mech ID -> list of subordinate IDs
        # Populated during reset() when suites are assigned
        self._subordinates: dict[str, list[str]] = {}

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

        # Reward computation (extracted for curriculum integration)
        self._reward_weights = RewardWeights(
            shaping_gamma=config.shaping_gamma,
            team_reward_alpha=config.team_reward_alpha,
        )
        self._reward_computer = RewardComputer(self._reward_weights)

        # Observation building (extracted for testability and curriculum integration)
        self._obs_builder = ObservationBuilder(
            config=self.config,
            max_contact_slots=self.MAX_CONTACT_SLOTS,
            contact_dim=self.CONTACT_DIM,
            local_map_r=self.LOCAL_MAP_R,
            comm_dim=self.comm_dim,
        )

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
            "paint_lock_uses_blue": 0.0,  # Missiles fired using teammate paint locks
            "paint_lock_uses_red": 0.0,
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
            # Zone control metrics
            "zone_ticks_blue": 0.0,
            "zone_ticks_red": 0.0,
            "contested_ticks": 0.0,
            "first_zone_entry_step": -1.0,  # -1 means never entered
            # Coordination metrics
            "pack_dispersion_sum": 0.0,
            "pack_dispersion_count": 0.0,
            "centroid_zone_dist_sum": 0.0,
            "centroid_zone_dist_count": 0.0,
            "focus_fire_concentration": 0.0,
            # Perception metrics
            "visible_contacts_sum": 0.0,
            "visible_contacts_count": 0.0,
            "hostile_filter_on_count": 0.0,
            # EWAR usage metrics (lights have ECM/ECCM since 2025-12-27)
            "ecm_on_ticks": 0.0,
            "eccm_on_ticks": 0.0,
            "light_ticks": 0.0,  # ECM/ECCM is on Light now, not Scout
        }
        self._damage_by_target: dict[str, float] = {}  # target_id -> damage received
        self._prev_fallen = {}
        self._prev_legged = {}
        self._prev_shutdown = {}
        self._ever_in_zone = {}  # Reset per-agent zone entry tracking

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
        self._mech_suites = {}  # Reset suite assignments
        for team, ids in (("blue", self.blue_ids), ("red", self.red_ids)):
            for i, mech_id in enumerate(ids):
                cls_name = _roster_for_index(i, self.num_packs)
                command_role = _command_role_for_index(i, self.num_packs)
                spec = self.mech_classes[cls_name]
                # Assign combat suite based on class and command role
                self._mech_suites[mech_id] = get_suite_for_role(cls_name, command_role)
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

        # Build command hierarchy: map command mechs to their subordinates
        self._subordinates = {}
        for _team, ids in (("blue", self.blue_ids), ("red", self.red_ids)):
            for i, mech_id in enumerate(ids):
                suite = self._mech_suites[mech_id]
                if suite.issues_orders:
                    if suite.order_scope == "pack":
                        # Pack leader: subordinates are pack members (excluding self)
                        pack_idx = i // PACK_SIZE
                        pack_start = pack_idx * PACK_SIZE
                        pack_end = min(pack_start + PACK_SIZE, len(ids))
                        self._subordinates[mech_id] = [
                            aid for aid in ids[pack_start:pack_end] if aid != mech_id
                        ]
                    elif suite.order_scope == "squad":
                        # Squad leader: subordinates are all pack leaders in the squad
                        pack_leaders = [aid for aid in ids if self._mech_suites[aid].order_scope == "pack"]
                        self._subordinates[mech_id] = pack_leaders

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
        clearance_z = int(max(1, min(self.config.world.connectivity_clearance_z, world.size_z)))
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

    def _compute_pack_dispersion(self, team: str) -> float:
        """Compute mean pairwise distance between pack members."""
        sim = self.sim
        if sim is None:
            return 0.0

        ids = self.blue_ids if team == "blue" else self.red_ids
        positions: list[np.ndarray] = []
        for mid in ids:
            m = sim.mechs.get(mid)
            if m is not None and m.alive:
                positions.append(m.pos[:2])  # XY only

        if len(positions) < 2:
            return 0.0

        # Mean pairwise distance
        total_dist = 0.0
        count = 0
        for i, p1 in enumerate(positions):
            for p2 in positions[i + 1 :]:
                total_dist += float(np.linalg.norm(p1 - p2))
                count += 1

        return total_dist / max(count, 1)

    def _obs(self) -> dict[str, np.ndarray]:
        """Build observations for all agents using ObservationBuilder."""
        ctx = ObservationContext.from_env(self)
        return self._obs_builder.build(ctx)

    def _obs_dim(self) -> int:
        """Return observation dimension (delegated to builder)."""
        return self._obs_builder.obs_dim()

    def _process_command_actions(self, act: dict[str, np.ndarray], sim: Sim) -> None:
        """Process command actions from pack/squad leaders and issue orders to subordinates.

        Command actions are only processed for mechs that have issues_orders=True in their suite.
        Orders are stored on the receiving mech's current_order field.
        """
        current_time = sim.time_s

        for commander_id, subordinates in self._subordinates.items():
            m = sim.mechs.get(commander_id)
            if m is None or not m.alive:
                continue

            a = act.get(commander_id)
            if a is None:
                continue

            # Parse command action fields
            order_type_prefs = a[self.CMD_ORDER_TYPE_START : self.CMD_ORDER_TYPE_START + ORDER_TYPE_DIM]
            recipient_prefs = a[self.CMD_RECIPIENT_START : self.CMD_RECIPIENT_START + ORDER_RECIPIENT_DIM]
            order_params = a[self.CMD_PARAM_START : self.CMD_PARAM_START + ORDER_PARAM_DIM]

            # Determine order type (argmax)
            order_type = int(np.argmax(order_type_prefs))

            # Skip if no order (OrderType.NONE = 0)
            if order_type == OrderType.NONE:
                continue

            # Determine recipient (argmax over subordinates, capped to actual subordinate count)
            recipient_idx = int(np.argmax(recipient_prefs[: len(subordinates)]))
            if recipient_idx >= len(subordinates):
                continue

            recipient_id = subordinates[recipient_idx]
            recipient = sim.mechs.get(recipient_id)
            if recipient is None or not recipient.alive:
                continue

            # For FOCUS_FIRE, determine target from order_params (argmax over contact slots)
            target_id: str | None = None
            if order_type == OrderType.FOCUS_FIRE:
                # Use commander's last contact slots to pick target
                contact_slots = self._last_contact_slots.get(commander_id) or []
                if order_params.size > 0:
                    target_idx = int(np.argmax(order_params[: len(contact_slots)]))
                    if target_idx < len(contact_slots):
                        target_id = contact_slots[target_idx]
                        # Validate target is a valid hostile
                        if target_id is not None:
                            tgt = sim.mechs.get(target_id)
                            if tgt is None or not tgt.alive or tgt.team == m.team:
                                target_id = None

            # Issue the order to the recipient
            recipient.current_order = ReceivedOrder(
                order_type=order_type,
                issuer_id=commander_id,
                target_id=target_id,
                issued_at=current_time,
                acknowledged=False,
            )

    def _process_status_reports(self, act: dict[str, np.ndarray], sim: Sim) -> None:
        """Process status report actions from subordinates.

        Subordinates update their current_order state via status report actions.
        This allows commanders to observe subordinate status.

        Status report action layout (11 dims):
        - acknowledge (1): > 0.5 means acknowledge order receipt
        - override_reason (6): one-hot for override reason (argmax)
        - progress (1): order completion progress [0, 1]
        - flags (3): is_engaged, is_blocked, needs_support (> 0.5 means set)
        """
        for aid in self.agents:
            m = sim.mechs.get(aid)
            if m is None or not m.alive:
                continue

            # Only process status reports for mechs with an active order
            order = m.current_order
            if order is None or order.is_expired(sim.time_s):
                continue

            a = act.get(aid)
            if a is None:
                continue

            # Parse status report action fields
            ack_val = float(a[self.STATUS_ACK_START])
            override_prefs = a[self.STATUS_OVERRIDE_START : self.STATUS_OVERRIDE_START + STATUS_OVERRIDE_DIM]
            progress_val = float(a[self.STATUS_PROGRESS_START])
            flags_vals = a[self.STATUS_FLAGS_START : self.STATUS_FLAGS_START + STATUS_FLAGS_DIM]

            # Update order state
            if ack_val > 0.5:
                order.acknowledged = True

            # Override reason (argmax)
            override_reason = int(np.argmax(override_prefs))
            order.override_reason = override_reason

            # Progress (clamp to [0, 1])
            order.progress = float(np.clip(progress_val, 0.0, 1.0))

            # Status flags (threshold at 0.5)
            order.is_engaged = bool(flags_vals[0] > 0.5) if flags_vals.size > 0 else False
            order.is_blocked = bool(flags_vals[1] > 0.5) if flags_vals.size > 1 else False
            order.needs_support = bool(flags_vals[2] > 0.5) if flags_vals.size > 2 else False

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict,  # infos: per-agent dicts + global "events" list
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

            if self.config.enable_target_selection:
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

            if self.config.enable_ewar:
                # Light: SPECIAL slot toggles EWAR (ECM broadcasts position - bad for stealth scout)
                # > 0.5: ECCM, (0.0, 0.5]: ECM, <= 0: Off
                # Light uses ECM/ECCM for combat disruption, losing smoke in exchange.
                ewar_val = float(a[ActionIndex.SPECIAL])
                if m.spec.name == "light":
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
        if self.config.enable_obs_control:
            for aid in self.agents:
                a = act[aid]
                sort_pref = a[self.OBS_CTRL_START : self.OBS_CTRL_START + self.OBS_SORT_DIM]
                self._contact_sort_mode[aid] = int(np.argmax(sort_pref))
                self._contact_filter_hostile[aid] = bool(
                    float(a[self.OBS_CTRL_START + self.OBS_SORT_DIM]) > 0.0
                )

        # Process command actions for pack/squad leaders
        self._process_command_actions(act, sim)

        # Process status reports from subordinates (updates their order state)
        self._process_status_reports(act, sim)

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
        step_shots_fired: dict[str, int] = dict.fromkeys(self.agents, 0)  # Aggression tracking

        # Episode stats (for logging/debugging) and per-agent combat tracking (HIGH-6).
        # NOTE: Event dict access uses [] not .get() to fail loudly on schema mismatch.
        if events:
            for ev in events:
                et = ev["type"]
                if et == "kill":
                    shooter_id = str(ev["shooter"])
                    target_id = str(ev["target"])
                    shooter = sim.mechs.get(shooter_id)
                    if shooter is not None:
                        self._episode_stats[f"kills_{shooter.team}"] = float(
                            self._episode_stats.get(f"kills_{shooter.team}", 0.0) + 1.0
                        )
                        step_kills[shooter_id] = step_kills.get(shooter_id, 0) + 1
                    if target_id in step_deaths:
                        step_deaths[target_id] = True
                elif et == "assist":
                    painter_id = str(ev["painter"])
                    painter = sim.mechs.get(painter_id)
                    if painter is not None:
                        self._episode_stats[f"assists_{painter.team}"] = float(
                            self._episode_stats.get(f"assists_{painter.team}", 0.0) + 1.0
                        )
                        step_assists[painter_id] = step_assists.get(painter_id, 0) + 1
                elif et == "paint":
                    shooter = sim.mechs.get(str(ev["shooter"]))
                    if shooter is not None:
                        self._episode_stats[f"paints_{shooter.team}"] = float(
                            self._episode_stats.get(f"paints_{shooter.team}", 0.0) + 1.0
                        )
                elif et == "laser_hit":
                    shooter_id = str(ev["shooter"])
                    target_id = str(ev["target"])
                    shooter = sim.mechs.get(shooter_id)
                    if shooter is not None:
                        self._episode_stats[f"laser_hits_{shooter.team}"] = float(
                            self._episode_stats.get(f"laser_hits_{shooter.team}", 0.0) + 1.0
                        )
                        dmg = float(ev["damage"])
                        self._episode_stats[f"damage_{shooter.team}"] = float(
                            self._episode_stats.get(f"damage_{shooter.team}", 0.0) + dmg
                        )
                        step_damage_dealt[shooter_id] = step_damage_dealt.get(shooter_id, 0.0) + dmg
                        step_shots_fired[shooter_id] = step_shots_fired.get(shooter_id, 0) + 1
                        if target_id in step_damage_received:
                            step_damage_received[target_id] = step_damage_received.get(target_id, 0.0) + dmg
                        # Track damage by target for focus fire metric
                        self._damage_by_target[target_id] = self._damage_by_target.get(target_id, 0.0) + dmg
                elif et == "projectile_hit":
                    shooter_id = str(ev["shooter"])
                    target_id = str(ev["target"])
                    shooter = sim.mechs.get(shooter_id)
                    if shooter is not None:
                        dmg = float(ev["damage"])
                        self._episode_stats[f"damage_{shooter.team}"] = float(
                            self._episode_stats.get(f"damage_{shooter.team}", 0.0) + dmg
                        )
                        step_damage_dealt[shooter_id] = step_damage_dealt.get(shooter_id, 0.0) + dmg
                        if target_id in step_damage_received:
                            step_damage_received[target_id] = step_damage_received.get(target_id, 0.0) + dmg
                        # Track damage by target for focus fire metric
                        self._damage_by_target[target_id] = self._damage_by_target.get(target_id, 0.0) + dmg
                elif et == "missile_launch":
                    shooter_id = str(ev["shooter"])
                    shooter = sim.mechs.get(shooter_id)
                    if shooter is not None:
                        self._episode_stats[f"missile_launches_{shooter.team}"] = float(
                            self._episode_stats.get(f"missile_launches_{shooter.team}", 0.0) + 1.0
                        )
                        # Track paint lock usage separately (missiles using teammate paint locks)
                        if ev.get("lock") == "paint":
                            self._episode_stats[f"paint_lock_uses_{shooter.team}"] = float(
                                self._episode_stats.get(f"paint_lock_uses_{shooter.team}", 0.0) + 1.0
                            )
                        step_shots_fired[shooter_id] = step_shots_fired.get(shooter_id, 0) + 1
                elif et == "kinetic_fire":
                    shooter_id = str(ev["shooter"])
                    shooter = sim.mechs.get(shooter_id)
                    if shooter is not None:
                        self._episode_stats[f"kinetic_fires_{shooter.team}"] = float(
                            self._episode_stats.get(f"kinetic_fires_{shooter.team}", 0.0) + 1.0
                        )
                        step_shots_fired[shooter_id] = step_shots_fired.get(shooter_id, 0) + 1
                elif et == "ams_intercept":
                    defender = sim.mechs.get(str(ev["defender"]))
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
            if not self.config.enable_comm:
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
        # NOTE: infos has per-agent dicts + global "events" list
        infos: dict = {}

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
        # Zone reward with contested trickle + dominance bonus.
        blue_tick, red_tick = compute_zone_ticks(
            in_zone_tonnage,
            contested_floor=self._reward_weights.contested_floor,
        )

        # Update scores (using ticks as progress towards winning)
        self.team_zone_score["blue"] = float(self.team_zone_score["blue"] + blue_tick * dt_act)
        self.team_zone_score["red"] = float(self.team_zone_score["red"] + red_tick * dt_act)

        # Track zone presence for metrics
        blue_in_zone = any(in_zone_by_agent.get(bid, False) for bid in self.blue_ids)
        red_in_zone = any(in_zone_by_agent.get(rid, False) for rid in self.red_ids)

        if blue_in_zone:
            self._episode_stats["zone_ticks_blue"] += 1.0
        if red_in_zone:
            self._episode_stats["zone_ticks_red"] += 1.0
        if blue_in_zone and red_in_zone:
            self._episode_stats["contested_ticks"] += 1.0
        if blue_in_zone and self._episode_stats["first_zone_entry_step"] < 0:
            self._episode_stats["first_zone_entry_step"] = float(self.step_count)

        # Coordination metrics (computed every step, accumulated)
        dispersion = self._compute_pack_dispersion("blue")
        self._episode_stats["pack_dispersion_sum"] += dispersion
        self._episode_stats["pack_dispersion_count"] += 1.0

        # Centroid to zone distance
        blue_positions = [sim.mechs[bid].pos[:2] for bid in self.blue_ids if sim.mechs[bid].alive]
        if blue_positions:
            centroid = np.mean(blue_positions, axis=0)
            centroid_dist = float(np.linalg.norm(centroid - np.array([zone_cx, zone_cy])))
            self._episode_stats["centroid_zone_dist_sum"] += centroid_dist
            self._episode_stats["centroid_zone_dist_count"] += 1.0

        # Track EWAR usage (lights have ECM/ECCM since 2025-12-27)
        for mid in self.blue_ids:
            m = sim.mechs.get(mid)
            if m is not None and m.alive and m.spec.name == "light":
                self._episode_stats["light_ticks"] += 1.0
                if m.ecm_on:
                    self._episode_stats["ecm_on_ticks"] += 1.0
                if m.eccm_on:
                    self._episode_stats["eccm_on_ticks"] += 1.0

        # Compute first zone entry for arrival bonus
        # An agent gets arrival bonus if: (1) currently in zone, (2) never been in zone before
        first_zone_entry_this_step: dict[str, bool] = {}
        for aid in self.agents:
            in_zone = in_zone_by_agent.get(aid, False)
            was_ever_in_zone = self._ever_in_zone.get(aid, False)
            if in_zone and not was_ever_in_zone:
                first_zone_entry_this_step[aid] = True
                self._ever_in_zone[aid] = True
            else:
                first_zone_entry_this_step[aid] = False

        # Build reward context and compute rewards via the reward module.
        # This enables curriculum-based reward schedules without modifying step().
        reward_ctx = StepContext(
            agents=self.agents,
            blue_ids=self.blue_ids,
            red_ids=self.red_ids,
            mech_teams={aid: sim.mechs[aid].team for aid in self.agents},
            mech_alive={aid: sim.mechs[aid].alive for aid in self.agents},
            mech_died={aid: sim.mechs[aid].died for aid in self.agents},
            zone_center=(zone_cx, zone_cy),
            zone_radius=zone_r,
            max_xy=max_xy,
            in_zone_by_agent=in_zone_by_agent,
            in_zone_tonnage=in_zone_tonnage,
            blue_tick=blue_tick,
            red_tick=red_tick,
            dist_to_zone_before=dist_to_zone_before,
            dist_to_zone_after=dist_to_zone_after,
            step_damage_dealt=step_damage_dealt,
            step_kills=step_kills,
            step_assists=step_assists,
            step_deaths=step_deaths,
            # Paint assists: use same data as step_assists since "assist" events track
            # when a paint-lock-guided kill happens. Painters get paint_assist_bonus (2.0)
            # in addition to the base assist reward (3.0).
            step_paint_assists=step_assists,
            first_zone_entry_this_step=first_zone_entry_this_step,
        )

        rewards, reward_components = self._reward_computer.compute(reward_ctx)

        # Store component breakdown in infos for training script analysis
        for aid in self.agents:
            infos[aid]["reward_components"] = reward_components[aid].to_dict()

        # Episode end conditions (King of the Hill is the primary win condition).
        hp_after = self.team_hp()
        control = blue_tick - red_tick

        blue_alive = sim.team_alive("blue")
        red_alive = sim.team_alive("red")
        time_up = self.step_count >= self.max_steps

        winner: str | None = None
        reason: str | None = None

        # Zone victory disabled - games run until elimination or time-up
        # Zone control still gives rewards, just doesn't end the game
        if (not blue_alive) or (not red_alive):
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
            # Focus fire concentration: damage on top target / total damage
            if self._damage_by_target:
                total_dmg = sum(self._damage_by_target.values())
                max_target_dmg = max(self._damage_by_target.values())
                self._episode_stats["focus_fire_concentration"] = max_target_dmg / max(total_dmg, 1.0)
            else:
                self._episode_stats["focus_fire_concentration"] = 0.0

            self.last_outcome = {
                "reason": reason,
                "winner": winner or "draw",
                "hp": hp_after,
                "zone_score": dict(self.team_zone_score),
                "zone_score_to_win": float(self.zone_score_to_win),
                "stats": dict(self._episode_stats),
            }

            # NOTE: Terminal win/loss rewards removed. They dominated the learning signal
            # (98%) and didn't provide gradient information. Mission success is now shaped
            # entirely through zone approach/control rewards. Dead agents learn from death
            # penalty (W_DEATH) and the shaping rewards they received while alive.

        # Attach aggregated events to all infos (for now).
        if events:
            for aid in infos:
                infos[aid]["events"] = events
        if self.last_outcome is not None:
            for aid in infos:
                infos[aid]["outcome"] = self.last_outcome

        # Global events key for dashboard/logging (lightweight - just reference, no copy)
        infos["events"] = events if events else []

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
        if world.meta:
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
