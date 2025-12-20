from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import MechClassConfig


@dataclass
class MechState:
    mech_id: str
    team: str
    spec: MechClassConfig
    pos: np.ndarray  # float32[3], voxel units, center of AABB
    vel: np.ndarray  # float32[3], voxel units / s
    yaw: float  # radians

    hp: float
    heat: float
    laser_cooldown: float = 0.0
    missile_cooldown: float = 0.0
    painter_cooldown: float = 0.0
    painted_remaining: float = 0.0
    alive: bool = True

    # Per-decision-step scratch.
    took_damage: float = 0.0
    dealt_damage: float = 0.0
    kills: int = 0
    died: bool = False
    was_hit: bool = False

    def reset_step_stats(self) -> None:
        self.took_damage = 0.0
        self.dealt_damage = 0.0
        self.kills = 0
        self.died = False
        self.was_hit = False

    @property
    def half_size(self) -> np.ndarray:
        sx, sy, sz = self.spec.size_voxels
        return np.asarray([sx, sy, sz], dtype=np.float32) * 0.5

    @property
    def shutdown(self) -> bool:
        return self.heat > self.spec.heat_cap
