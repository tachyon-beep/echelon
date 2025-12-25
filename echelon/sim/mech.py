from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import MechClassConfig


@dataclass
class ReceivedOrder:
    """An order received from a command mech.

    Orders are contracts with escape clauses. Units can acknowledge,
    report progress, or override with a valid reason.
    """

    order_type: int  # OrderType enum value
    issuer_id: str  # Who gave the order
    target_id: str | None  # For FOCUS_FIRE: which enemy to target
    issued_at: float  # Sim time when order was issued

    # Contract state
    acknowledged: bool = False  # Whether agent has acknowledged receipt
    progress: float = 0.0  # Order completion progress [0, 1]
    override_reason: int = 0  # OrderOverride enum value (0 = NONE = following)

    # Status flags (set by subordinate's status report action)
    is_engaged: bool = False  # Currently in combat
    is_blocked: bool = False  # Cannot proceed (nav failure, obstacle)
    needs_support: bool = False  # Requesting assistance

    def is_expired(self, current_time: float, ttl: float = 10.0) -> bool:
        """Orders expire after TTL seconds."""
        return (current_time - self.issued_at) > ttl

    def is_overridden(self) -> bool:
        """Whether the order is being overridden (not followed)."""
        return self.override_reason != 0

    def is_complete(self) -> bool:
        """Whether the order is complete (progress >= 1.0)."""
        return self.progress >= 1.0


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

    # Command and control: received orders from pack/squad leaders.
    current_order: ReceivedOrder | None = None

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
