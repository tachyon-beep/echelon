# echelon/server/models.py
"""Pydantic models for API requests/responses."""

from typing import Any

from pydantic import BaseModel


class ChunkPayload(BaseModel):
    """A chunk of replay data pushed from training script."""

    replay_id: str
    world_ref: str
    chunk_index: int
    chunk_count: int
    frames: list[dict[str, Any]]
    meta: dict[str, Any] | None = None


class PushResponse(BaseModel):
    """Response to a chunk push."""

    status: str
    replay_id: str
    chunk_index: int
    clients_notified: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    clients: int
    uptime_s: float


class NavGraphRequest(BaseModel):
    """Request for NavGraph data."""

    world_hash: str


class PathRequest(BaseModel):
    """Request for A* pathfinding."""

    world_hash: str
    start_pos: list[float]
    goal_pos: list[float]
