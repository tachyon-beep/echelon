from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    size_x: int = 20
    size_y: int = 20
    size_z: int = 20
    voxel_size_m: float = 1.0
    obstacle_fill: float = 0.06


@dataclass(frozen=True)
class MechClassConfig:
    name: str
    size_voxels: tuple[float, float, float]
    max_speed: float
    max_yaw_rate: float
    max_jet_accel: float
    hp: float
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
    record_replay: bool = False
    seed: int | None = None
