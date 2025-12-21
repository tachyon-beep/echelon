from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    size_x: int = 20
    size_y: int = 20
    size_z: int = 20
    voxel_size_m: float = 1.0
    obstacle_fill: float = 0.06
    ensure_connectivity: bool = True
    connectivity_clearance_z: int = 4
    connectivity_obstacle_inflate_radius: int = 1
    connectivity_wall_cost: float = 80.0
    connectivity_penalty_radius: int = 3
    connectivity_penalty_cost: float = 20.0
    connectivity_carve_width: int = 5


@dataclass(frozen=True)
class MechClassConfig:
    name: str
    size_voxels: tuple[float, float, float]
    max_speed: float
    max_yaw_rate: float
    max_jet_accel: float
    hp: float
    leg_hp: float
    heat_cap: float
    heat_dissipation: float


@dataclass(frozen=True)
class EnvConfig:
    world: WorldConfig = WorldConfig()
    num_packs: int = 1
    dt_sim: float = 0.05
    decision_repeat: int = 5
    max_episode_seconds: float = 60.0
    observation_mode: str = "full"  # "full" | "partial"
    # Optional, pack-scoped communication side-channel.
    # Each mech can emit a message vector as part of its action, and receives the
    # last messages from packmates in its observation (1 decision-tick delay).
    comm_dim: int = 8  # 0 disables
    record_replay: bool = False
    seed: int | None = None
