"""Combat Suite System - Variable observation architecture for class-differentiated perception.

Each mech class has a different "combat suite" that determines:
- How many contacts they can track
- How much squad awareness they have
- Whether they can issue orders
- What intel streams they receive

This creates role differentiation through information access, not hard-coded behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from torch import Tensor, nn


class SuiteType(Enum):
    """Types of combat suites available."""

    SCOUT_RECON = auto()  # Maximum contacts, minimal squad awareness
    LIGHT_SKIRMISH = auto()  # Good awareness for mobility/flanking
    MEDIUM_ASSAULT = auto()  # Balanced tactical picture
    HEAVY_FIRE_SUPPORT = auto()  # Narrow focus on current target
    PACK_COMMAND = auto()  # Pack-level C2 (light chassis)
    SQUAD_COMMAND = auto()  # Squad-level C2 (medium chassis)


@dataclass(frozen=True)
class CombatSuiteSpec:
    """Defines a mech's sensor/display package.

    This determines what information is available to the policy.
    The cockpit is a game mechanic, not a rendering artifact.
    """

    suite_type: SuiteType

    # === CONTACT TRACKING ===
    visual_contact_slots: int  # Own sensor contacts
    sensor_range_mult: float  # Detection range modifier
    sensor_fidelity: float  # Detail level on contacts (0-1)

    # === SQUAD AWARENESS ===
    squad_position_slots: int  # How many squadmates shown
    squad_detail_level: int  # 0=icon, 1=status, 2=full
    sees_squad_engaged: bool  # See what squadmates are fighting

    # === COMMAND INTEGRATION ===
    receives_orders: bool  # Gets order broadcast
    issues_orders: bool  # Can send orders (command only)
    order_scope: str  # "none" | "pack" | "squad"

    # === SHARED INTEL ===
    receives_painted_targets: bool  # See pack's paint locks
    receives_contact_reports: bool  # See scout's reported contacts
    intel_slots: int  # How many shared contacts shown

    @property
    def is_command(self) -> bool:
        """Whether this suite can issue orders."""
        return self.issues_orders

    @property
    def contact_capacity_norm(self) -> float:
        """Normalized contact capacity (for suite descriptor)."""
        return self.visual_contact_slots / 20.0

    @property
    def squad_detail_norm(self) -> float:
        """Normalized squad detail level."""
        return self.squad_detail_level / 2.0


# === SUITE LIBRARY ===

COMBAT_SUITES: dict[SuiteType, CombatSuiteSpec] = {
    SuiteType.SCOUT_RECON: CombatSuiteSpec(
        suite_type=SuiteType.SCOUT_RECON,
        # Scouts see EVERYTHING in range - their job is finding
        visual_contact_slots=20,
        sensor_range_mult=1.5,
        sensor_fidelity=0.8,  # Good detail for reporting
        # Minimal squad awareness - focused outward
        squad_position_slots=5,  # Know where friendlies are (don't shoot them)
        squad_detail_level=0,  # Icons only
        sees_squad_engaged=False,  # Not their concern
        # Orders: receive only
        receives_orders=True,
        issues_orders=False,
        order_scope="none",
        # Intel: they CREATE intel, less need to receive
        receives_painted_targets=True,
        receives_contact_reports=False,  # They ARE the contact reports
        intel_slots=5,
    ),
    SuiteType.LIGHT_SKIRMISH: CombatSuiteSpec(
        suite_type=SuiteType.LIGHT_SKIRMISH,
        # Good awareness for mobility/flanking
        visual_contact_slots=10,
        sensor_range_mult=1.2,
        sensor_fidelity=0.6,
        # Need to know where squad is for coordination
        squad_position_slots=5,
        squad_detail_level=1,  # Status level
        sees_squad_engaged=False,
        # Orders: receive, see squad picture
        receives_orders=True,
        issues_orders=False,
        order_scope="none",
        # Intel: receives shared targeting
        receives_painted_targets=True,
        receives_contact_reports=True,
        intel_slots=8,
    ),
    SuiteType.MEDIUM_ASSAULT: CombatSuiteSpec(
        suite_type=SuiteType.MEDIUM_ASSAULT,
        # Moderate awareness - balanced
        visual_contact_slots=8,
        sensor_range_mult=1.0,
        sensor_fidelity=0.7,
        # Squad awareness for formation
        squad_position_slots=5,
        squad_detail_level=1,
        sees_squad_engaged=False,
        # Orders
        receives_orders=True,
        issues_orders=False,
        order_scope="none",
        # Intel
        receives_painted_targets=True,
        receives_contact_reports=True,
        intel_slots=6,
    ),
    SuiteType.HEAVY_FIRE_SUPPORT: CombatSuiteSpec(
        suite_type=SuiteType.HEAVY_FIRE_SUPPORT,
        # Narrow focus - kill what's in front of you
        visual_contact_slots=5,
        sensor_range_mult=0.8,  # Shorter range, relies on squad
        sensor_fidelity=1.0,  # But VERY detailed on what it sees
        # Minimal squad awareness
        squad_position_slots=5,
        squad_detail_level=0,  # Icons - don't shoot friendlies
        sees_squad_engaged=False,
        # Orders: just tell me who to kill
        receives_orders=True,
        issues_orders=False,
        order_scope="none",
        # Intel: receives targeting data
        receives_painted_targets=True,
        receives_contact_reports=True,
        intel_slots=3,  # Few slots, but high priority targets
    ),
    SuiteType.PACK_COMMAND: CombatSuiteSpec(
        suite_type=SuiteType.PACK_COMMAND,
        # Pack leader (light chassis) - coordinates 5 subordinates
        visual_contact_slots=8,
        sensor_range_mult=1.0,
        sensor_fidelity=0.6,
        # Full pack awareness
        squad_position_slots=5,  # All packmates
        squad_detail_level=2,  # Full detail
        sees_squad_engaged=True,  # See what everyone is fighting
        # Orders: issues to pack, receives from squad leader
        receives_orders=True,
        issues_orders=True,
        order_scope="pack",
        # Intel: sees pack sensor fusion
        receives_painted_targets=True,
        receives_contact_reports=True,
        intel_slots=10,
    ),
    SuiteType.SQUAD_COMMAND: CombatSuiteSpec(
        suite_type=SuiteType.SQUAD_COMMAND,
        # Squad leader (medium chassis) - coordinates entire squad
        visual_contact_slots=6,
        sensor_range_mult=1.0,
        sensor_fidelity=0.5,
        # FULL squad awareness - this is the job
        squad_position_slots=12,  # All squad members
        squad_detail_level=2,  # Full detail
        sees_squad_engaged=True,  # See what everyone is fighting
        # Orders: issues to anyone, receives higher echelon orders
        receives_orders=True,
        issues_orders=True,
        order_scope="squad",
        # Intel: full picture
        receives_painted_targets=True,
        receives_contact_reports=True,
        intel_slots=12,
    ),
}


def get_suite_for_role(mech_class: str, command_role: str | None) -> CombatSuiteSpec:
    """Get the appropriate combat suite for a mech based on class and command role."""
    if command_role == "squad_leader":
        return COMBAT_SUITES[SuiteType.SQUAD_COMMAND]
    elif command_role == "pack_leader":
        return COMBAT_SUITES[SuiteType.PACK_COMMAND]
    elif mech_class == "scout":
        return COMBAT_SUITES[SuiteType.SCOUT_RECON]
    elif mech_class == "light":
        return COMBAT_SUITES[SuiteType.LIGHT_SKIRMISH]
    elif mech_class == "medium":
        return COMBAT_SUITES[SuiteType.MEDIUM_ASSAULT]
    elif mech_class == "heavy":
        return COMBAT_SUITES[SuiteType.HEAVY_FIRE_SUPPORT]
    else:
        return COMBAT_SUITES[SuiteType.MEDIUM_ASSAULT]  # Fallback


# === SET ENCODER ===


class DeepSetEncoder(nn.Module):
    """Permutation-invariant set encoder with masking.

    Transforms a variable-length set of entities into a fixed-size representation.
    Uses the DeepSet architecture: phi(x) for each entity, then aggregate with rho.

    Reference: Zaheer et al., "Deep Sets" (NeurIPS 2017)
    """

    def __init__(
        self,
        entity_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        aggregation: str = "mean",  # "sum", "max", or "mean"
    ):
        super().__init__()
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.aggregation = aggregation

        # Per-entity transform (applied independently to each entity)
        self.phi = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Post-aggregation transform
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, entities: Tensor, mask: Tensor) -> Tensor:
        """Encode a set of entities into a fixed-size representation.

        Args:
            entities: (batch, max_entities, entity_dim) - padded entity features
            mask: (batch, max_entities) - 1.0 for padding, 0.0 for real entities

        Returns:
            (batch, output_dim) - fixed-size set representation
        """
        # Encode each entity independently
        h = self.phi(entities)  # (batch, max_entities, hidden_dim)

        # Zero out padding (valid_mask is 1.0 where real, 0.0 where padding)
        valid_mask = (1.0 - mask).unsqueeze(-1)  # (batch, max_entities, 1)
        h = h * valid_mask

        # Aggregate across entities
        if self.aggregation == "sum":
            pooled = h.sum(dim=1)
        elif self.aggregation == "max":
            # For max, we need to handle padding specially
            h = h.masked_fill(mask.unsqueeze(-1).bool(), float("-inf"))
            pooled = h.max(dim=1).values
            # Replace -inf with 0 for empty sets
            pooled = pooled.masked_fill(pooled == float("-inf"), 0.0)
        elif self.aggregation == "mean":
            # Mean pooling with proper count handling
            count = valid_mask.sum(dim=1).clamp(min=1.0)
            pooled = h.sum(dim=1) / count
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        result: Tensor = self.rho(pooled)
        return result


# === SUITE DESCRIPTOR ===


def build_suite_descriptor(suite: CombatSuiteSpec) -> list[float]:
    """Build a descriptor vector that tells the policy what kind of mech it is.

    This is critical for shared weights to work - without it, the policy must
    infer role from indirect cues (observation richness, which slots are filled).
    With it, the network can learn role-specific behavior conditioning.

    Returns:
        14-dim vector: suite_type_onehot(6) + capability_scalars(4) + flags(4)
    """
    # Suite type one-hot (6 dims)
    suite_onehot = [0.0] * 6
    suite_onehot[suite.suite_type.value - 1] = 1.0  # Enum values start at 1

    # Capability scalars (4 dims)
    capabilities = [
        suite.contact_capacity_norm,  # Contact capacity
        suite.sensor_range_mult,  # Sensor quality
        suite.sensor_fidelity,  # Classification quality
        suite.squad_detail_norm,  # Squad awareness
    ]

    # Flags (4 dims)
    flags = [
        float(suite.sees_squad_engaged),
        float(suite.issues_orders),
        float(suite.receives_contact_reports),
        float(suite.intel_slots > 5),  # "Has good intel access"
    ]

    return suite_onehot + capabilities + flags


SUITE_DESCRIPTOR_DIM = 14
