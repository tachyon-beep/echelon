# echelon/server/routes/league.py
"""League API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException

if TYPE_CHECKING:
    from echelon.arena.league import League

router = APIRouter(prefix="/api", tags=["league"])

# Module-level state set by init_league_routes
_league: League | None = None


def init_league_routes(league: League) -> None:
    """Initialize routes with league instance."""
    global _league
    _league = league


@router.get("/league")
def get_league() -> dict[str, Any]:
    """Get full league state."""
    if _league is None:
        raise HTTPException(503, "League not initialized")

    return {
        "schema_version": 1,
        "entries": {eid: entry.as_dict() for eid, entry in _league.entries.items()},
    }


@router.get("/league/standings")
def get_standings() -> list[dict[str, Any]]:
    """Get sorted leaderboard."""
    if _league is None:
        raise HTTPException(503, "League not initialized")

    entries = list(_league.entries.values())
    entries.sort(key=lambda e: e.rating.rating, reverse=True)

    return [
        {
            "entry_id": e.entry_id,
            "commander_name": e.commander_name,
            "kind": e.kind,
            "rating": e.rating.rating,
            "rd": e.rating.rd,
            "games": e.games,
            "aggregate_stats": e.aggregate_stats.to_dict(),
        }
        for e in entries
        if e.kind != "retired"
    ]


@router.get("/commanders/{entry_id}")
def get_commander(entry_id: str) -> dict[str, Any]:
    """Get commander profile with full stats."""
    if _league is None:
        raise HTTPException(503, "League not initialized")

    entry = _league.entries.get(entry_id)
    if entry is None:
        raise HTTPException(404, f"Commander '{entry_id}' not found")

    return {
        "entry_id": entry.entry_id,
        "commander_name": entry.commander_name,
        "kind": entry.kind,
        "rating": entry.rating.rating,
        "rd": entry.rating.rd,
        "games": entry.games,
        "rating_history": entry.rating_history,
        "aggregate_stats": entry.aggregate_stats.to_dict(),
    }
