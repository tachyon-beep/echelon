from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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
    leg_hp: float
    heat: float
    stability: float
    max_stability: float = 100.0
    fallen_time: float = 0.0  # >0 means fallen/stunned

    # Optional per-step intent (set by the env, read by the sim).
    focus_target_id: str | None = None

    # Electronic warfare toggles (set by the env, applied by the sim/obs).
    ecm_on: bool = False
    eccm_on: bool = False

    # Status effects.
    suppressed_time: float = 0.0  # Stability regen penalty while > 0.
    ams_cooldown: float = 0.0  # Anti-missile system cooldown (seconds).

    laser_cooldown: float = 0.0
    missile_cooldown: float = 0.0
    kinetic_cooldown: float = 0.0
    painter_cooldown: float = 0.0
    painted_remaining: float = 0.0
    last_painter_id: str | None = None
    noise_level: float = 0.0  # Acoustic footprint
    alive: bool = True

    # Per-decision-step scratch.
    took_damage: float = 0.0
    dealt_damage: float = 0.0
    kills: int = 0
    died: bool = False
    was_hit: bool = False
    last_damage_dir: np.ndarray | None = None  # Direction to attacker (for evasion learning)

    @property
    def is_legged(self) -> bool:
        return self.leg_hp <= 0.0

    def reset_step_stats(self) -> None:
        self.took_damage = 0.0
        self.dealt_damage = 0.0
        self.kills = 0
        self.died = False
        self.was_hit = False
        self.last_damage_dir = None

    @property
    def half_size(self) -> np.ndarray:
        sx, sy, sz = self.spec.size_voxels
        return np.asarray([sx, sy, sz], dtype=np.float32) * 0.5

    @property
    def shutdown(self) -> bool:
        return self.heat > self.spec.heat_cap
