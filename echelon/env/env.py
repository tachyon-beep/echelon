from __future__ import annotations

import math

import numpy as np

from ..config import EnvConfig, MechClassConfig
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
            max_yaw_rate=3.0,
            max_jet_accel=6.0,
            hp=80.0,
            heat_cap=80.0,
            heat_dissipation=12.0,
        ),
        "medium": MechClassConfig(
            name="medium",
            size_voxels=(2.5, 2.5, 3.0),
            max_speed=4.5,
            max_yaw_rate=2.5,
            max_jet_accel=5.0,
            hp=120.0,
            heat_cap=100.0,
            heat_dissipation=11.0,
        ),
        "heavy": MechClassConfig(
            name="heavy",
            size_voxels=(3.5, 3.5, 4.0),
            max_speed=3.3,
            max_yaw_rate=2.0,
            max_jet_accel=4.0,
            hp=200.0,
            heat_cap=130.0,
            heat_dissipation=10.0,
        ),
    }


def _team_ids(num_packs: int) -> tuple[list[str], list[str]]:
    total = num_packs * 5
    blue = [f"blue_{i}" for i in range(total)]
    red = [f"red_{i}" for i in range(total)]
    return blue, red


def _roster_for_index(i: int) -> str:
    # Pack structure: Heavy, Med, Med, Light, Light
    idx_in_pack = i % 5
    if idx_in_pack == 0:
        return "heavy"
    elif idx_in_pack < 3:
        return "medium"
    else:
        return "light"


class EchelonEnv:
    """
    Minimal multi-agent env with a PettingZoo-parallel-like API:

      obs, infos = env.reset(seed)
      obs, rewards, terminations, truncations, infos = env.step(actions)

    Actions are continuous float32 vectors of shape (8,):
      [forward, strafe, vertical, yaw_rate, fire_laser, vent, fire_missile, paint]
    """

    ACTION_DIM = 8

    def __init__(self, config: EnvConfig):
        self.config = config
        self.mech_classes = default_mech_classes()

        self.rng = np.random.default_rng(config.seed)
        self.world: VoxelWorld | None = None
        self.sim: Sim | None = None
        self.last_outcome: dict | None = None

        self.blue_ids, self.red_ids = _team_ids(config.num_packs)
        self.possible_agents = [*self.blue_ids, *self.red_ids]
        self.agents = list(self.possible_agents)

        self.max_steps = int(math.ceil(config.max_episode_seconds / (config.dt_sim * config.decision_repeat)))
        self.step_count = 0
        self._replay: list[dict] | None = None

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.last_outcome = None
        world = VoxelWorld.generate(self.config.world, self.rng)

        # Clear spawn regions in corners.
        spawn_clear = max(15, int(self.config.world.size_x * 0.15))
        # Blue: Bottom-Left (0,0)
        world.set_box_solid(0, 0, 0, spawn_clear, spawn_clear, world.size_z, False)
        # Red: Top-Right (size_x, size_y)
        world.set_box_solid(world.size_x - spawn_clear, world.size_y - spawn_clear, 0, world.size_x, world.size_y, world.size_z, False)

        mechs: dict[str, MechState] = {}
        for team, ids in (("blue", self.blue_ids), ("red", self.red_ids)):
            for i, mech_id in enumerate(ids):
                cls_name = _roster_for_index(i)
                spec = self.mech_classes[cls_name]
                hs = np.asarray(spec.size_voxels, dtype=np.float32) * 0.5

                if team == "blue":
                    # Scatter within bottom-left corner
                    # Arrange packs somewhat logically? 
                    # Just simple grid scatter for now
                    cols = 5
                    x = 2.0 + float(i % cols) * 4.0 + float(hs[0])
                    y = 2.0 + float(i // cols) * 4.0 + float(hs[1])
                    yaw = math.pi * 0.25 # Face center (NE)
                else:
                    # Scatter within top-right corner
                    cols = 5
                    x = float(world.size_x) - 2.0 - float(i % cols) * 4.0 - float(hs[0])
                    y = float(world.size_y) - 2.0 - float(i // cols) * 4.0 - float(hs[1])
                    yaw = math.pi * 1.25 # Face center (SW)

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
                    heat=0.0,
                )
                mechs[mech_id] = mech

        sim = Sim(world=world, dt_sim=self.config.dt_sim, rng=self.rng)
        sim.reset(mechs)

        self.world = world
        self.sim = sim
        self.agents = list(self.possible_agents)

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

    def _obs_entity_features(self, viewer: MechState, other: MechState, visible: bool) -> np.ndarray:
        world = self.world
        assert world is not None

        # Static always-on info.
        alive = 1.0 if other.alive else 0.0
        ally = 1.0 if other.team == viewer.team else 0.0

        cls = other.spec.name
        class_onehot = np.zeros(3, dtype=np.float32)
        if cls == "light":
            class_onehot[0] = 1.0
        elif cls == "medium":
            class_onehot[1] = 1.0
        elif cls == "heavy":
            class_onehot[2] = 1.0

        if not visible:
            # In partial mode: keep just (alive, ally, class) for now.
            return np.concatenate(
                [
                    np.zeros(3, dtype=np.float32),  # rel pos
                    np.zeros(3, dtype=np.float32),  # rel vel
                    np.zeros(2, dtype=np.float32),  # yaw sin/cos
                    np.zeros(2, dtype=np.float32),  # hp/heat norms
                    np.asarray([alive, ally], dtype=np.float32),
                    class_onehot,
                    np.zeros(1, dtype=np.float32), # is_painted
                ]
            )

        rel = (other.pos - viewer.pos).astype(np.float32, copy=False)
        max_dim = float(max(world.size_x, world.size_y, world.size_z))
        rel /= max_dim
        rel_vel = (other.vel - viewer.vel).astype(np.float32, copy=False) / 10.0
        yaw_sin = math.sin(other.yaw)
        yaw_cos = math.cos(other.yaw)
        hp_norm = float(np.clip(other.hp / max(1.0, other.spec.hp), 0.0, 1.0))
        heat_norm = float(np.clip(other.heat / max(1.0, other.spec.heat_cap), 0.0, 2.0))
        is_painted = 1.0 if other.painted_remaining > 0.0 else 0.0

        return np.concatenate(
            [
                rel,
                rel_vel,
                np.asarray([yaw_sin, yaw_cos], dtype=np.float32),
                np.asarray([hp_norm, heat_norm], dtype=np.float32),
                np.asarray([alive, ally], dtype=np.float32),
                class_onehot,
                np.asarray([is_painted], dtype=np.float32),
            ]
        )

    def _obs(self) -> dict[str, np.ndarray]:
        sim = self.sim
        world = self.world
        assert sim is not None and world is not None

        radar_range = 14.0
        obs: dict[str, np.ndarray] = {}
        for aid in self.agents:
            viewer = sim.mechs[aid]
            if not viewer.alive:
                obs[aid] = np.zeros(self._obs_dim(), dtype=np.float32)
                continue

            parts: list[np.ndarray] = []
            for other_id in self.possible_agents:
                other = sim.mechs[other_id]
                if self.config.observation_mode == "full":
                    visible = True
                else:
                    if other.team == viewer.team:
                        visible = True
                    elif not other.alive:
                        visible = False
                    else:
                        dist = float(np.linalg.norm(other.pos - viewer.pos))
                        visible = dist <= radar_range or has_los(world, viewer.pos, other.pos)
                parts.append(self._obs_entity_features(viewer, other, visible=visible))

            # Extra per-agent scalars.
            targeted = 1.0 if self._is_targeted(viewer) else 0.0
            under_fire = 1.0 if viewer.was_hit else 0.0
            painted = 1.0 if viewer.painted_remaining > 0.0 else 0.0
            parts.append(np.asarray([targeted, under_fire, painted], dtype=np.float32))
            obs[aid] = np.concatenate(parts).astype(np.float32, copy=False)
        return obs

    def _obs_dim(self) -> int:
        # entity features = 3 rel + 3 rel_vel + 2 yaw + 2 hp/heat + 2 alive/ally + 3 class + 1 painted = 16
        entity_dim = 16
        return len(self.possible_agents) * entity_dim + 3

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
            a = np.asarray(a, dtype=np.float32).reshape(self.ACTION_DIM)
            act[aid] = a

        events = sim.step(act, num_substeps=self.config.decision_repeat)

        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        # Per-step dense shaping (v0): damage, kill, death.
        for aid in self.agents:
            m = sim.mechs[aid]
            if not m.alive:
                r = -1.0 if m.died else 0.0
            else:
                r = 0.0
                r += 0.01 * float(m.dealt_damage)
                r -= 0.01 * float(m.took_damage)
                r += 1.0 * float(m.kills)
                r -= 0.001  # time pressure
            rewards[aid] = float(r)
            terminations[aid] = bool(not m.alive)
            truncations[aid] = False
            infos[aid] = {"events": [], "alive": bool(m.alive)}

        # Episode end conditions.
        blue_alive = sim.team_alive("blue")
        red_alive = sim.team_alive("red")
        time_up = self.step_count >= self.max_steps

        if time_up or (not blue_alive) or (not red_alive):
            if time_up:
                for aid in truncations:
                    truncations[aid] = True
                # HP-based tiebreaker: decide a winner instead of a draw when time runs out.
                hp = self.team_hp()
                if blue_alive and not red_alive:
                    winner = "blue"
                elif red_alive and not blue_alive:
                    winner = "red"
                elif not blue_alive and not red_alive:
                    winner = "draw"
                else:
                    eps = 1e-3
                    if hp["blue"] > hp["red"] + eps:
                        winner = "blue"
                    elif hp["red"] > hp["blue"] + eps:
                        winner = "red"
                    else:
                        winner = "draw"

                if winner in ("blue", "red"):
                    for aid in self.agents:
                        m = sim.mechs[aid]
                        if not m.alive:
                            continue
                        rewards[aid] += 2.0 if m.team == winner else -2.0

                self.last_outcome = {"reason": "time_up", "winner": winner, "hp": hp}
            else:
                winning_team = "blue" if blue_alive else "red"
                for aid in self.agents:
                    m = sim.mechs[aid]
                    if not m.alive:
                        continue
                    rewards[aid] += 2.0 if m.team == winning_team else -2.0
                # Mark episode termination for all agents (even if still alive).
                for aid in terminations:
                    terminations[aid] = True
                self.last_outcome = {"reason": "elimination", "winner": winning_team, "hp": self.team_hp()}

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
        assert sim is not None
        return {
            "t": float(sim.time_s),
            "tick": int(sim.tick),
            "mechs": {
                mid: {
                    "team": m.team,
                    "class": m.spec.name,
                    "pos": [float(x) for x in m.pos],
                    "vel": [float(v) for v in m.vel],
                    "yaw": float(m.yaw),
                    "hp": float(m.hp),
                    "heat": float(m.heat),
                    "alive": bool(m.alive),
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
            solid_voxels = []
            # Iterate to find solid blocks (simple loop for 20^3 is fast enough)
            for z in range(self.world.size_z):
                for y in range(self.world.size_y):
                    for x in range(self.world.size_x):
                        if self.world.solid[z, y, x]:
                            solid_voxels.append([x, y, z])
            world_data["walls"] = solid_voxels

        return {"world": world_data, "frames": self._replay}
