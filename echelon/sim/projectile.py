from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Projectile:
    shooter_id: str
    target_id: str | None  # None for unguided
    weapon: str  # "missile" | "gauss" | "autocannon" | ...
    pos: np.ndarray  # float32[3]
    vel: np.ndarray  # float32[3]
    speed: float
    damage: float
    stability_damage: float
    max_lifetime: float
    guidance: str = "homing"  # "homing", "ballistic", "linear"
    splash_rad: float = 0.0
    splash_scale: float = 0.0

    # State
    age: float = 0.0
    alive: bool = True
