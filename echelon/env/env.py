from __future__ import annotations

import math

import numpy as np

from ..actions import ACTION_DIM as ACTION_DIM_CONST
from ..constants import PACK_SIZE
from ..config import EnvConfig, MechClassConfig
from ..gen.objective import capture_zone_params, clear_capture_zone, sample_capture_zone
from ..gen.recipe import build_recipe
from ..gen.transforms import apply_transform_solids, list_transforms, opposite_corner, transform_corner
from ..gen.validator import ConnectivityValidator
from ..sim.los import has_los
from ..sim.mech import MechState
from ..sim.sim import Sim
from ..sim.world import VoxelWorld


def default_mech_classes() -> dict[str, MechClassConfig]:
    return {
        "light": MechClassConfig(
            name="light",
            size_voxels=(1.5, 1.5, 2.0),
            max_speed=6.0,
            max_yaw_rate=1.5, # Fast but catchable
            max_jet_accel=16.0,
            hp=80.0,
            leg_hp=40.0,
            heat_cap=80.0,
            heat_dissipation=12.0,
        ),
        "medium": MechClassConfig(
            name="medium",
            size_voxels=(2.5, 2.5, 3.0),
            max_speed=4.5,
            max_yaw_rate=1.0, # Moderate
            max_jet_accel=0.0,
            hp=120.0,
            leg_hp=60.0,
            heat_cap=100.0,
            heat_dissipation=11.0,
        ),
        "heavy": MechClassConfig(
            name="heavy",
            size_voxels=(3.5, 3.5, 4.0),
            max_speed=3.3,
            max_yaw_rate=0.6, # Slow turret
            max_jet_accel=0.0,
            hp=200.0,
            leg_hp=100.0,
            heat_cap=130.0,
            heat_dissipation=10.0,
        ),
    }


def _team_ids(num_packs: int) -> tuple[list[str], list[str]]:
    total = num_packs * PACK_SIZE
    blue = [f"blue_{i}" for i in range(total)]
    red = [f"red_{i}" for i in range(total)]
    return blue, red


def _roster_for_index(i: int) -> str:
    # Pack structure (10 mechs): 2 Heavy, 5 Medium, 3 Light
    idx_in_pack = i % PACK_SIZE
    if idx_in_pack < 2:
        return "heavy"
    elif idx_in_pack < 7: # 2+5 = 7
        return "medium"
    else:
        return "light"


class EchelonEnv:
    """
    Minimal multi-agent env with a PettingZoo-parallel-like API:

      obs, infos = env.reset(seed)
      obs, rewards, terminations, truncations, infos = env.step(actions)

    actions: continuous float32 vectors of shape (9,):
      [forward, strafe, vertical, yaw_rate, fire_laser, vent, fire_missile, paint, fire_kinetic]

    If `config.comm_dim > 0`, actions have shape (9 + comm_dim,) and the tail
    `comm_dim` floats are treated as a pack-scoped communication message (seen by
    packmates on the next decision tick).

    Observation-control action fields (always present):
    - sort mode (3 floats): argmax selects {closest, biggest, most_damaged}
    - filter (1 float): >0 selects hostile-only contacts
    """

    BASE_ACTION_DIM = ACTION_DIM_CONST
    ACTION_DIM = ACTION_DIM_CONST
    OBS_SORT_DIM = 3
    OBS_CTRL_DIM = OBS_SORT_DIM + 1  # + hostile-only filter
    OBS_CTRL_START = BASE_ACTION_DIM
    COMM_START = BASE_ACTION_DIM + OBS_CTRL_DIM
    CONTACT_SLOTS = 5
    # rel(3) + rel_vel(3) + yaw(2) + hp/heat(2) + stab/fallen/legged(3)
    # + relation_onehot(3) + class_onehot(3) + painted(1) + visible(1)
    CONTACT_DIM = 21

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
        self._contact_sort_mode: dict[str, int] = {aid: 0 for aid in self.possible_agents}  # 0=closest
        self._contact_filter_hostile: dict[str, bool] = {aid: False for aid in self.possible_agents}

        self.max_steps = int(math.ceil(config.max_episode_seconds / (config.dt_sim * config.decision_repeat)))
        self.step_count = 0
        self._replay: list[dict] | None = None
        self.team_zone_score: dict[str, float] = {"blue": 0.0, "red": 0.0}
        self.zone_score_to_win: float = float(config.max_episode_seconds * 0.5)
        self._max_team_hp: dict[str, float] = {"blue": 0.0, "red": 0.0}

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

        seq = np.random.SeedSequence(episode_seed)
        rng_world, rng_sim, rng_variants = (np.random.default_rng(s) for s in seq.spawn(3))

        world = VoxelWorld.generate(self.config.world, rng_world)

        # Variant choices: choose spawn corner (canonical) and a map reorientation transform.
        blue_canon = str(rng_variants.choice(["BL", "BR", "TL", "TR"]))
        red_canon = opposite_corner(blue_canon)
        transform = str(rng_variants.choice(list_transforms()))
        world.solid = apply_transform_solids(world.solid, transform)
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
        required_clear = int(math.ceil(max(required_x, required_y) + 1.0))
        spawn_clear = max(25, int(min(self.config.world.size_x, self.config.world.size_y) * 0.25), required_clear)
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
                world.set_box_solid(0, 0, 0, spawn_clear, spawn_clear, world.size_z, False)
            elif corner == "BR":
                world.set_box_solid(world.size_x - spawn_clear, 0, 0, world.size_x, spawn_clear, world.size_z, False)
            elif corner == "TL":
                world.set_box_solid(0, world.size_y - spawn_clear, 0, spawn_clear, world.size_y, world.size_z, False)
            elif corner == "TR":
                world.set_box_solid(
                    world.size_x - spawn_clear, world.size_y - spawn_clear, 0, world.size_x, world.size_y, world.size_z, False
                )
            else:
                raise ValueError(f"Unknown corner: {corner!r}")

        clear_corner(spawn_corners["blue"])
        clear_corner(spawn_corners["red"])

        world.meta["capture_zone"] = sample_capture_zone(
            world, rng_variants, spawn_clear=spawn_clear, spawn_corners=spawn_corners
        )
        clear_capture_zone(world, meta=world.meta)

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
            validator.validate_and_fix(
                world.solid,
                spawn_corners=dict(world.meta.get("spawn_corners", spawn_corners)),
                spawn_clear=spawn_clear,
                meta=world.meta,
            )

        world.meta["recipe"] = build_recipe(
            generator_id="legacy_voxel_archetypes",
            generator_version="1",
            seed=episode_seed,
            world_config=self.config.world,
            world_meta=world.meta,
            solids_zyx=world.solid,
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

                yaw = float(math.atan2(zone_cy - y, zone_cx - x))

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

        if self.comm_dim > 0:
            for aid in self._comm_last:
                self._comm_last[aid].fill(0.0)
        for aid in self._contact_sort_mode:
            self._contact_sort_mode[aid] = 0
        for aid in self._contact_filter_hostile:
            self._contact_filter_hostile[aid] = False

        if self.config.record_replay:
            self._replay = []
        else:
            self._replay = None

        obs = self._obs()
        infos = {aid: {} for aid in self.agents}
        return obs, infos

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
        class_onehot = np.zeros(3, dtype=np.float32)
        if cls == "light":
            class_onehot[0] = 1.0
        elif cls == "medium":
            class_onehot[1] = 1.0
        elif cls == "heavy":
            class_onehot[2] = 1.0

        feat[0:3] = rel
        feat[3:6] = rel_vel
        feat[6:8] = np.asarray([yaw_sin, yaw_cos], dtype=np.float32)
        feat[8:10] = np.asarray([hp_norm, heat_norm], dtype=np.float32)
        feat[10:13] = np.asarray([stab_norm, fallen, is_legged], dtype=np.float32)
        feat[13:16] = rel_onehot
        feat[16:19] = class_onehot
        feat[19] = 1.0 if painted_by_pack else 0.0
        feat[20] = 1.0  # visible
        return feat

    def _obs(self) -> dict[str, np.ndarray]:
        sim = self.sim
        world = self.world
        assert sim is not None and world is not None

        radar_range = 14.0
        zone_cx, zone_cy, zone_r = capture_zone_params(world.meta, size_x=world.size_x, size_y=world.size_y)
        max_xy = float(max(world.size_x, world.size_y))
        zone_r_norm = float(np.clip(zone_r / max(1.0, max_xy), 0.0, 1.0))

        denom = max(1, max(len(self.blue_ids), len(self.red_ids)))
        in_zone_counts: dict[str, int] = {"blue": 0, "red": 0}
        for m in sim.mechs.values():
            if not m.alive:
                continue
            dist = float(math.hypot(float(m.pos[0] - zone_cx), float(m.pos[1] - zone_cy)))
            if dist < zone_r:
                in_zone_counts[m.team] += 1
        zone_control = float((in_zone_counts["blue"] - in_zone_counts["red"]) / denom)

        time_frac = float(np.clip(float(sim.time_s) / max(1e-6, float(self.config.max_episode_seconds)), 0.0, 1.0))
        los_cache: dict[tuple[str, str], bool] = {}

        def _cached_los(a: str, b: str, a_pos: np.ndarray, b_pos: np.ndarray) -> bool:
            key = (a, b) if a < b else (b, a)
            hit = los_cache.get(key)
            if hit is not None:
                return hit
            ok = bool(has_los(world, a_pos, b_pos))
            los_cache[key] = ok
            return ok

        reserved = {"friendly": 3, "hostile": 1, "neutral": 1}
        repurpose_priority = ["hostile", "friendly", "neutral"]

        obs: dict[str, np.ndarray] = {}
        for aid in self.agents:
            viewer = sim.mechs[aid]
            if not viewer.alive:
                obs[aid] = np.zeros(self._obs_dim(), dtype=np.float32)
                continue

            pack_ids = self._packmates.get(aid, [])
            sort_mode = int(self._contact_sort_mode.get(aid, 0))
            hostile_only = bool(self._contact_filter_hostile.get(aid, False))

            # Top-K contact table (fixed size).
            contacts: dict[str, list[tuple[float, str, bool]]] = {"friendly": [], "hostile": [], "neutral": []}
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
                    if painted_by_pack or dist <= radar_range:
                        visible = True
                    else:
                        visible = _cached_los(aid, other_id, viewer.pos, other.pos)

                if not visible:
                    continue

                relation = "friendly" if other.team == viewer.team else "hostile"
                if sort_mode == 0:
                    key = (dist,)
                elif sort_mode == 1:
                    cls_rank = {"light": 1, "medium": 2, "heavy": 3}.get(other.spec.name, 0)
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

            # Extra per-agent scalars.
            targeted = 1.0 if self._is_targeted(viewer) else 0.0
            under_fire = 1.0 if viewer.was_hit else 0.0
            painted = 1.0 if viewer.painted_remaining > 0.0 else 0.0
            shutdown = 1.0 if viewer.shutdown else 0.0

            # Self kinematics (needed for jump jets and movement control).
            self_vel = (viewer.vel / 10.0).astype(np.float32, copy=False)

            # Ability cooldowns (normalized); 5s is a safe upper bound for current weapons.
            cd_scale = 5.0
            laser_cd = float(np.clip(viewer.laser_cooldown / cd_scale, 0.0, 1.0))
            missile_cd = float(np.clip(viewer.missile_cooldown / cd_scale, 0.0, 1.0))
            kinetic_cd = float(np.clip(viewer.kinetic_cooldown / cd_scale, 0.0, 1.0))
            painter_cd = float(np.clip(viewer.painter_cooldown / cd_scale, 0.0, 1.0))

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

            parts.append(
                np.asarray(
                    [
                        targeted,
                        under_fire,
                        painted,
                        shutdown,
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
            obs[aid] = np.concatenate(parts).astype(np.float32, copy=False)
        return obs

    def _obs_dim(self) -> int:
        comm_dim = PACK_SIZE * int(max(0, int(getattr(self.config, "comm_dim", 0))))
        # self features = targeted, under_fire, painted, shutdown, self_vel(3), cooldowns(4), in_zone,
        #                vec_to_zone(3), zone_radius, my_control, my_score, enemy_score, time_frac,
        #                obs_sort_onehot(3), hostile_only = 24
        return self.CONTACT_SLOTS * self.CONTACT_DIM + comm_dim + 24

    def _is_targeted(self, mech: MechState) -> bool:
        sim = self.sim
        world = self.world
        assert sim is not None and world is not None

        # Simple “someone has LOS to me and is in range”.
        for other in sim.mechs.values():
            if not other.alive or other.team == mech.team:
                continue
            if float(np.linalg.norm(other.pos - mech.pos)) > 8.0:
                continue
            if has_los(world, other.pos, mech.pos):
                return True
        return False

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
                        f"(base={self.BASE_ACTION_DIM}, obs_ctrl={self.OBS_CTRL_DIM}, comm_dim={self.comm_dim})"
                    )
                a = a.reshape(self.ACTION_DIM)
            act[aid] = a.astype(np.float32, copy=False)

        # Update per-mech observation controls from the chosen action (applies to returned obs).
        for aid in self.agents:
            a = act[aid]
            sort_pref = a[self.OBS_CTRL_START : self.OBS_CTRL_START + self.OBS_SORT_DIM]
            self._contact_sort_mode[aid] = int(np.argmax(sort_pref))
            self._contact_filter_hostile[aid] = bool(float(a[self.OBS_CTRL_START + self.OBS_SORT_DIM]) > 0.0)

        # Baselines for first-principles rewards.
        hp_before = self.team_hp()

        events = sim.step(act, num_substeps=self.config.decision_repeat)

        # Update comm buffers after the sim step (dead mechs do not broadcast).
        if self.comm_dim > 0:
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

        # Reward is derived from first principles (in order):
        # 1) King-of-the-hill territory control (capture zone).
        # 2) Self-preservation (avoid losing team HP).
        # 3) Attrition (remove enemy HP).
        # 4) Efficiency (avoid shutdown due to heat).
        dt_act = float(self.config.dt_sim * self.config.decision_repeat)
        zone_cx, zone_cy, zone_r = capture_zone_params(
            sim.world.meta, size_x=sim.world.size_x, size_y=sim.world.size_y
        )
        denom = max(1, max(len(self.blue_ids), len(self.red_ids)))
        max_blue_hp = max(1.0, float(self._max_team_hp.get("blue", 0.0)))
        max_red_hp = max(1.0, float(self._max_team_hp.get("red", 0.0)))

        in_zone_counts: dict[str, int] = {"blue": 0, "red": 0}
        shutdown_counts: dict[str, int] = {"blue": 0, "red": 0}
        in_zone_by_agent: dict[str, bool] = {}

        for aid in self.agents:
            m = sim.mechs[aid]
            alive = bool(m.alive)
            terminations[aid] = bool(not alive)
            truncations[aid] = False
            infos[aid] = {"events": [], "alive": alive}

            in_zone = False
            if alive:
                dist = float(math.hypot(float(m.pos[0] - zone_cx), float(m.pos[1] - zone_cy)))
                in_zone = dist < zone_r
                if in_zone:
                    in_zone_counts[m.team] += 1
                if m.shutdown:
                    shutdown_counts[m.team] += 1
            in_zone_by_agent[aid] = in_zone

        # Territory scoring: whichever team has more mechs in the zone gains score,
        # proportional to the (normalized) presence advantage.
        control = float((in_zone_counts["blue"] - in_zone_counts["red"]) / denom)
        if control > 0.0:
            self.team_zone_score["blue"] = float(self.team_zone_score["blue"] + control * dt_act)
        elif control < 0.0:
            self.team_zone_score["red"] = float(self.team_zone_score["red"] + (-control) * dt_act)

        hp_after = self.team_hp()

        blue_hp_loss = max(0.0, float(hp_before["blue"] - hp_after["blue"]))
        red_hp_loss = max(0.0, float(hp_before["red"] - hp_after["red"]))
        blue_hp_loss_norm = float(blue_hp_loss / max_blue_hp)
        red_hp_loss_norm = float(red_hp_loss / max_red_hp)

        # Tunable weights (kept intentionally small/simple).
        W_ZONE = 0.05
        W_SELF = 0.70
        W_KILL = 0.50
        W_SHUTDOWN = 0.05
        W_DEATH = 0.10

        r_team = {
            "blue": (W_ZONE * control)
            - (W_SELF * blue_hp_loss_norm)
            + (W_KILL * red_hp_loss_norm)
            - (W_SHUTDOWN * (shutdown_counts["blue"] / denom)),
            "red": (-W_ZONE * control)
            - (W_SELF * red_hp_loss_norm)
            + (W_KILL * blue_hp_loss_norm)
            - (W_SHUTDOWN * (shutdown_counts["red"] / denom)),
        }

        for aid in self.agents:
            m = sim.mechs[aid]
            r = float(r_team[m.team]) if m.alive else 0.0
            if m.died:
                r -= W_DEATH
            rewards[aid] = float(r)

        # Episode end conditions (King of the Hill is the primary win condition).
        blue_alive = sim.team_alive("blue")
        red_alive = sim.team_alive("red")
        time_up = self.step_count >= self.max_steps

        winner: str | None = None
        reason: str | None = None

        if self.team_zone_score["blue"] >= self.zone_score_to_win or self.team_zone_score["red"] >= self.zone_score_to_win:
            winner = "blue" if self.team_zone_score["blue"] > self.team_zone_score["red"] else "red"
            reason = "zone_control"
            for aid in terminations:
                terminations[aid] = True
                truncations[aid] = True
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

        if winner in ("blue", "red"):
            W_WIN = 1.0
            for aid in self.agents:
                m = sim.mechs[aid]
                if not m.alive:
                    continue
                rewards[aid] += W_WIN if m.team == winner else -W_WIN

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
            }

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
                    "heat": float(m.heat),
                    "stability": float(m.stability),
                    "alive": bool(m.alive),
                    "fallen": bool(m.fallen_time > 0.0),
                }
                for mid, m in sim.mechs.items()
            },
            "events": events,
        }

    def get_replay(self) -> dict | None:
        if self._replay is None:
            return None
        
        # Serialize world (sparse list of solid blocks)
        world_data = {}
        if self.world is not None:
            world_data["size"] = [self.world.size_x, self.world.size_y, self.world.size_z]
            world_data["voxel_size_m"] = float(self.world.voxel_size_m)
            if self._last_reset_seed is not None:
                world_data["seed"] = int(self._last_reset_seed)
            if self._spawn_clear is not None:
                world_data["spawn_clear"] = int(self._spawn_clear)
            if getattr(self.world, "meta", None):
                world_data["meta"] = self.world.meta
            idx_zyx = np.argwhere(self.world.solid)
            if idx_zyx.size:
                world_data["walls"] = idx_zyx[:, [2, 1, 0]].tolist()
            else:
                world_data["walls"] = []

        return {"world": world_data, "frames": self._replay}
