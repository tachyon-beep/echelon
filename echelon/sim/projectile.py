from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Projectile:
    shooter_id: str
    target_id: str  # For homing
    pos: np.ndarray  # float32[3]
    vel: np.ndarray  # float32[3]
    speed: float
    damage: float
    max_lifetime: float
    
    # State
    age: float = 0.0
    alive: bool = True
