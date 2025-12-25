# echelon/server/routes/push.py
"""Chunked replay push endpoint."""

import json
import logging

from fastapi import APIRouter, HTTPException, Response

from ..models import ChunkPayload, PushResponse
from ..sse import sse_manager
from ..world_cache import world_cache

logger = logging.getLogger("echelon.server")

router = APIRouter()


@router.post("/push", response_model=PushResponse)
async def push_chunk(chunk: ChunkPayload):
    """
    Receive a replay chunk and broadcast to connected viewers.

    Training script calls this multiple times per replay:
    - chunk_index=0: includes meta, triggers replay_start
    - chunk_index=1..N-1: frame data only
    - chunk_index=N-1 (last): triggers replay_end
    """
    # Validate world exists in cache
    if not world_cache.has(chunk.world_ref):
        raise HTTPException(
            status_code=400,
            detail=f"World {chunk.world_ref} not in cache. PUT /worlds/{{hash}} first.",
        )

    # Broadcast based on chunk position
    if chunk.chunk_index == 0:
        # First chunk: send replay_start + world + chunk
        start_data = {
            "replay_id": chunk.replay_id,
            "world_ref": chunk.world_ref,
            "chunk_count": chunk.chunk_count,
            "meta": chunk.meta or {},
            "world": world_cache.get(chunk.world_ref),
        }
        await sse_manager.broadcast("replay_start", json.dumps(start_data))

    # Always send chunk data
    chunk_data = {
        "replay_id": chunk.replay_id,
        "chunk_index": chunk.chunk_index,
        "frames": chunk.frames,
    }
    notified = await sse_manager.broadcast("replay_chunk", json.dumps(chunk_data))

    # Last chunk: send replay_end
    if chunk.chunk_index == chunk.chunk_count - 1:
        end_data = {"replay_id": chunk.replay_id}
        await sse_manager.broadcast("replay_end", json.dumps(end_data))

    logger.info(
        f"Pushed chunk {chunk.chunk_index + 1}/{chunk.chunk_count} "
        f"for {chunk.replay_id} ({len(chunk.frames)} frames, {notified} clients)"
    )

    return PushResponse(
        status="ok",
        replay_id=chunk.replay_id,
        chunk_index=chunk.chunk_index,
        clients_notified=notified,
    )


@router.head("/worlds/{world_hash}")
async def check_world(world_hash: str):
    """Check if world is cached (for client-side dedup)."""
    if world_cache.has(world_hash):
        return Response(status_code=200)
    return Response(status_code=404)


@router.get("/worlds/{world_hash}")
async def get_world(world_hash: str):
    """Get cached world by hash."""
    world = world_cache.get(world_hash)
    if world is None:
        raise HTTPException(status_code=404, detail=f"World {world_hash} not found")
    return world


@router.put("/worlds/{world_hash}")
async def put_world(world_hash: str, world: dict):
    """Store world with client-computed hash."""
    actual_hash = world_cache.compute_hash(world)
    if actual_hash != world_hash:
        raise HTTPException(
            status_code=400,
            detail=f"Hash mismatch: expected {world_hash}, got {actual_hash}",
        )
    world_cache.put(world, world_hash)
    logger.info(f"Cached world {world_hash} ({len(world_cache)} in cache)")
    return {"status": "ok", "world_hash": world_hash}
