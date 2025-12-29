# echelon/server/routes/matches.py
"""Match API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query

if TYPE_CHECKING:
    from echelon.arena.history import MatchHistory

router = APIRouter(prefix="/api", tags=["matches"])

# Module-level state set by init_match_routes
_history: MatchHistory | None = None


def init_match_routes(history: MatchHistory) -> None:
    """Initialize routes with history instance."""
    global _history
    _history = history


@router.get("/matches")
def get_matches(
    limit: int = Query(50, ge=1, le=200),
    entry_id: str | None = None,
) -> list[dict[str, Any]]:
    """Get recent matches."""
    if _history is None:
        raise HTTPException(503, "Match history not initialized")

    records = _history.list_recent(limit=limit, entry_id=entry_id)
    return [r.to_dict() for r in records]


@router.get("/matches/{match_id}")
def get_match(match_id: str) -> dict[str, Any]:
    """Get single match detail."""
    if _history is None:
        raise HTTPException(503, "Match history not initialized")

    record = _history.load(match_id)
    if record is None:
        raise HTTPException(404, f"Match '{match_id}' not found")

    return record.to_dict()
