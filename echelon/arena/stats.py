"""Match statistics data structures for arena tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TeamStats:
    """Aggregate statistics for one team in a match."""

    # Combat
    kills: int = 0
    deaths: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0

    # Objective
    zone_ticks: int = 0

    # Weapon usage (counts)
    primary_uses: int = 0
    secondary_uses: int = 0
    tertiary_uses: int = 0

    # Utility usage
    vents: int = 0
    smokes: int = 0
    ecm_toggles: int = 0
    paints: int = 0

    # Resource events
    overheats: int = 0
    knockdowns: int = 0

    # Pack leader command stats
    orders_issued: dict[str, int] = field(default_factory=dict)
    orders_acknowledged: int = 0
    orders_overridden: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "kills": self.kills,
            "deaths": self.deaths,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "zone_ticks": self.zone_ticks,
            "primary_uses": self.primary_uses,
            "secondary_uses": self.secondary_uses,
            "tertiary_uses": self.tertiary_uses,
            "vents": self.vents,
            "smokes": self.smokes,
            "ecm_toggles": self.ecm_toggles,
            "paints": self.paints,
            "overheats": self.overheats,
            "knockdowns": self.knockdowns,
            "orders_issued": self.orders_issued,
            "orders_acknowledged": self.orders_acknowledged,
            "orders_overridden": self.orders_overridden,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TeamStats:
        """Deserialize from dict."""
        return cls(
            kills=d.get("kills", 0),
            deaths=d.get("deaths", 0),
            damage_dealt=d.get("damage_dealt", 0.0),
            damage_taken=d.get("damage_taken", 0.0),
            zone_ticks=d.get("zone_ticks", 0),
            primary_uses=d.get("primary_uses", 0),
            secondary_uses=d.get("secondary_uses", 0),
            tertiary_uses=d.get("tertiary_uses", 0),
            vents=d.get("vents", 0),
            smokes=d.get("smokes", 0),
            ecm_toggles=d.get("ecm_toggles", 0),
            paints=d.get("paints", 0),
            overheats=d.get("overheats", 0),
            knockdowns=d.get("knockdowns", 0),
            orders_issued=d.get("orders_issued", {}),
            orders_acknowledged=d.get("orders_acknowledged", 0),
            orders_overridden=d.get("orders_overridden", 0),
        )


@dataclass
class MatchRecord:
    """Complete record of a single match."""

    match_id: str
    timestamp: float  # Unix epoch
    blue_entry_id: str  # League entry ID or "training"
    red_entry_id: str  # League entry ID (e.g., "heuristic")
    winner: str  # "blue" | "red" | "draw"
    duration_steps: int
    termination: str  # "zone" | "elimination" | "timeout"

    blue_stats: TeamStats
    red_stats: TeamStats

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "match_id": self.match_id,
            "timestamp": self.timestamp,
            "blue_entry_id": self.blue_entry_id,
            "red_entry_id": self.red_entry_id,
            "winner": self.winner,
            "duration_steps": self.duration_steps,
            "termination": self.termination,
            "blue_stats": self.blue_stats.to_dict(),
            "red_stats": self.red_stats.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MatchRecord:
        """Deserialize from dict."""
        return cls(
            match_id=d["match_id"],
            timestamp=d["timestamp"],
            blue_entry_id=d["blue_entry_id"],
            red_entry_id=d["red_entry_id"],
            winner=d["winner"],
            duration_steps=d["duration_steps"],
            termination=d["termination"],
            blue_stats=TeamStats.from_dict(d.get("blue_stats", {})),
            red_stats=TeamStats.from_dict(d.get("red_stats", {})),
        )
