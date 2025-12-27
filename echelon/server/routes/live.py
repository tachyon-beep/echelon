# echelon/server/routes/live.py
"""Live training status endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/live", tags=["live"])

# In-memory state for current matchup
_current_matchup: dict[str, Any] | None = None


class MatchupUpdate(BaseModel):
    """Payload for updating current matchup."""

    blue_entry_id: str
    blue_rating: float
    red_entry_id: str
    red_rating: float
    blue_stats_preview: dict[str, Any] | None = None
    red_stats_preview: dict[str, Any] | None = None


@router.get("/matchup")
def get_live_matchup() -> dict[str, Any]:
    """Get current training matchup."""
    if _current_matchup is None:
        return {"active": False}

    return {"active": True, **_current_matchup}


@router.post("/matchup")
def set_live_matchup(update: MatchupUpdate) -> dict[str, str]:
    """Set current training matchup (called by training loop)."""
    global _current_matchup
    _current_matchup = update.model_dump()
    return {"status": "ok"}


@router.delete("/matchup")
def clear_live_matchup() -> dict[str, str]:
    """Clear current matchup (training ended)."""
    global _current_matchup
    _current_matchup = None
    return {"status": "ok"}
