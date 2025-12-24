"""
Echelon Replay Server - SSE-based streaming with maximum robustness.

Features:
- Server-Sent Events (SSE) for push notifications (replaces WebSocket)
- Per-client message queues with backpressure (drop oldest if full)
- Keep-alive pings every 15s to prevent proxy/browser timeouts
- Event IDs for resumption via Last-Event-ID header
- Channel subscriptions for filtered streams
- Graceful shutdown with client cleanup
- Connection health monitoring
"""

from __future__ import annotations

import asyncio
import heapq
import json
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from echelon.nav.graph import NavGraph
from echelon.nav.planner import Planner
from echelon.sim.world import VoxelWorld

# --- Configuration ---


class Settings(BaseSettings):
    # SSE settings
    SSE_QUEUE_SIZE: int = 16  # Max queued events per client
    SSE_KEEPALIVE_S: float = 15.0  # Send keepalive every N seconds
    SSE_CLIENT_TIMEOUT_S: float = 60.0  # Disconnect idle clients after N seconds
    SSE_MAX_CLIENTS: int = 64  # Maximum concurrent SSE clients
    SSE_MAX_BROADCAST_BYTES: int = 25 * 1024 * 1024  # Max replay size
    SSE_MAX_LATEST_CHANNELS: int = 8  # Max channels to cache latest replay for
    SSE_RECENT_EVENTS_MAX: int = 100  # Max recent event IDs to track (not full data)

    # Replay settings
    REPLAYS_LIST_LIMIT: int = 200
    REPLAYS_CACHE_TTL_S: float = 2.0

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8090
    RUNS_DIR: Path = (Path(__file__).resolve().parent / "runs").resolve()

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echelon.server")

# --- Data Models ---


class ReplayPushResponse(BaseModel):
    status: str
    bytes: int
    summary: dict[str, Any]
    channel: str
    event_id: int
    clients_notified: int


class ReplayListEntry(BaseModel):
    name: str
    path: str


# --- SSE Client Management ---


@dataclass
class SSEClient:
    """Represents a connected SSE client."""

    client_id: str
    queue: asyncio.Queue[tuple[int, str, str]]  # (event_id, event_type, data)
    channel: str | None
    connected_at: float
    last_event_id: int = 0
    _cancelled: bool = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


@dataclass
class SSEEvent:
    """An event to be broadcast to clients."""

    event_id: int
    event_type: str  # "replay", "status", etc.
    data: str  # JSON-encoded payload
    channel: str | None  # None = broadcast to all
    timestamp: float = field(default_factory=time.time)


@dataclass
class SSEEventRef:
    """Lightweight reference to a past event (no data stored)."""

    event_id: int
    event_type: str
    channel: str | None
    timestamp: float


class SSEManager:
    """
    Manages SSE client connections with robustness features.

    - Per-client async queues with backpressure (drop oldest if full)
    - Channel subscriptions
    - Event ID tracking for resumption
    - Keep-alive task
    - Graceful shutdown
    """

    def __init__(self) -> None:
        self.clients: dict[str, SSEClient] = {}
        self.lock = asyncio.Lock()
        self._event_counter = 0
        self._event_counter_lock = asyncio.Lock()
        self._shutdown = False

        # Recent event refs for Last-Event-ID tracking (lightweight, no data)
        self._recent_events: list[SSEEventRef] = []

        # Latest replay per channel for new connections (LRU-ordered)
        self._latest_by_channel: OrderedDict[str, SSEEvent] = OrderedDict()

    async def _next_event_id(self) -> int:
        async with self._event_counter_lock:
            self._event_counter += 1
            return self._event_counter

    async def register(self, channel: str | None = None, last_event_id: int = 0) -> SSEClient:
        """Register a new SSE client."""
        async with self.lock:
            if len(self.clients) >= settings.SSE_MAX_CLIENTS:
                # Evict oldest client
                oldest = min(self.clients.values(), key=lambda c: c.connected_at)
                oldest.cancel()
                del self.clients[oldest.client_id]
                logger.warning(f"Evicted oldest client {oldest.client_id} (max clients reached)")

            client_id = str(uuid.uuid4())[:8]
            client = SSEClient(
                client_id=client_id,
                queue=asyncio.Queue(maxsize=settings.SSE_QUEUE_SIZE),
                channel=channel,
                connected_at=time.time(),
                last_event_id=last_event_id,
            )
            self.clients[client_id] = client
            logger.info(
                f"SSE client {client_id} connected (channel={channel}, last_event_id={last_event_id}, total={len(self.clients)})"
            )
            return client

    async def unregister(self, client: SSEClient) -> None:
        """Unregister an SSE client."""
        async with self.lock:
            if client.client_id in self.clients:
                del self.clients[client.client_id]
                logger.info(f"SSE client {client.client_id} disconnected (total={len(self.clients)})")

    async def broadcast(
        self,
        event_type: str,
        data: str,
        channel: str | None = None,
    ) -> tuple[int, int]:
        """
        Broadcast an event to all matching clients.

        Returns (event_id, clients_notified).
        """
        event_id = await self._next_event_id()
        event = SSEEvent(
            event_id=event_id,
            event_type=event_type,
            data=data,
            channel=channel,
        )

        # Store lightweight ref for resumption tracking (no data)
        event_ref = SSEEventRef(
            event_id=event_id,
            event_type=event_type,
            channel=channel,
            timestamp=time.time(),
        )
        async with self.lock:
            self._recent_events.append(event_ref)
            if len(self._recent_events) > settings.SSE_RECENT_EVENTS_MAX:
                self._recent_events = self._recent_events[-settings.SSE_RECENT_EVENTS_MAX :]

            # Update latest for channel with LRU eviction
            effective_channel = channel or "default"
            # Move to end if exists (LRU touch)
            if effective_channel in self._latest_by_channel:
                self._latest_by_channel.move_to_end(effective_channel)
            self._latest_by_channel[effective_channel] = event
            # Evict oldest if over limit
            while len(self._latest_by_channel) > settings.SSE_MAX_LATEST_CHANNELS:
                self._latest_by_channel.popitem(last=False)

        # Broadcast to matching clients
        notified = 0
        async with self.lock:
            targets = list(self.clients.values())

        for client in targets:
            if client.is_cancelled:
                continue

            # Channel filter: client receives if:
            # - Client subscribed to this channel, OR
            # - Client subscribed to None (global), OR
            # - Event has no channel (broadcast to all)
            if channel is not None and client.channel is not None and client.channel != channel:
                continue

            try:
                # Non-blocking put with backpressure
                if client.queue.full():
                    # Drop oldest event (drain one)
                    with suppress(asyncio.QueueEmpty):
                        client.queue.get_nowait()
                client.queue.put_nowait((event_id, event_type, data))
                notified += 1
            except Exception as e:
                logger.warning(f"Failed to queue event for client {client.client_id}: {e}")

        return event_id, notified

    async def get_missed_event_count(self, client: SSEClient) -> int:
        """Get count of events missed since client's last_event_id.

        Note: We only track event refs, not full data. This is used to
        update client.last_event_id to avoid duplicate sends.
        """
        async with self.lock:
            missed = [e for e in self._recent_events if e.event_id > client.last_event_id]
            if missed:
                # Update client to latest event ID to prevent stale resumption
                client.last_event_id = missed[-1].event_id
            return len(missed)

    async def get_latest_for_channel(self, channel: str | None) -> SSEEvent | None:
        """Get the latest event for a channel (for new connections)."""
        async with self.lock:
            effective_channel = channel or "default"
            # Try exact channel first
            if effective_channel in self._latest_by_channel:
                return self._latest_by_channel[effective_channel]
            # Fallback: return most recent from any channel
            if self._latest_by_channel:
                return max(self._latest_by_channel.values(), key=lambda e: e.event_id)
            return None

    @property
    def is_shutdown(self) -> bool:
        """Check if the manager is shutting down."""
        return self._shutdown

    async def shutdown(self) -> None:
        """Gracefully shutdown all client connections."""
        self._shutdown = True
        async with self.lock:
            for client in self.clients.values():
                client.cancel()
            self.clients.clear()
        logger.info("SSE manager shutdown complete")


# --- Replay Manager ---


class ReplayManager:
    """Manages replay storage and file listing."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.replays_list_cache: dict[int, tuple[float, list[dict[str, str]]]] = {}
        self.list_cache_lock = asyncio.Lock()

    def summarize(self, replay: dict[str, Any]) -> dict[str, Any]:
        world = replay.get("world") if isinstance(replay, dict) else None
        frames = replay.get("frames") if isinstance(replay, dict) else None
        out: dict[str, Any] = {
            "frames": len(frames) if isinstance(frames, list) else None,
        }
        if isinstance(world, dict):
            out["seed"] = world.get("seed")
            out["size"] = world.get("size")
        return out

    def encode(self, replay: dict[str, Any]) -> tuple[str, int]:
        msg = {"type": "replay", "data": replay}
        text = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
        return text, len(text.encode("utf-8"))

    async def list_replays_files(self, limit: int) -> list[dict[str, str]]:
        limit = int(max(1, min(int(limit), 50_000)))
        now = time.monotonic()

        async with self.list_cache_lock:
            hit = self.replays_list_cache.get(limit)
            if hit is not None:
                ts, cached = hit
                if now - ts <= settings.REPLAYS_CACHE_TTL_S:
                    return cached

        result = await asyncio.to_thread(self._scan_fs, limit)

        async with self.list_cache_lock:
            self.replays_list_cache[limit] = (now, result)
            if len(self.replays_list_cache) > 16:
                self.replays_list_cache.clear()

        return result

    def _scan_fs(self, limit: int) -> list[dict[str, str]]:
        root = settings.RUNS_DIR
        if not root.exists():
            return []

        # Use (mtime, path_str, Path) tuples to avoid comparing Path objects
        # when mtimes are equal (Path doesn't define ordering in Python 3)
        items: list[tuple[float, str, Path]] = []
        for p in root.rglob("*.json"):
            try:
                resolved = p.resolve()
                if not (resolved.is_relative_to(root) and resolved.is_file()):
                    continue
                mtime = float(resolved.stat().st_mtime)
            except OSError:
                continue

            path_str = str(p)
            if len(items) < limit:
                heapq.heappush(items, (mtime, path_str, p))
            elif mtime > items[0][0]:
                heapq.heapreplace(items, (mtime, path_str, p))

        items.sort(key=lambda t: t[0], reverse=True)
        return [
            {"name": p.relative_to(root).as_posix(), "path": p.relative_to(root).as_posix()}
            for _, _, p in items
        ]


# --- Global State & Lifespan ---

sse_manager = SSEManager()
replay_manager = ReplayManager()
_server_start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Echelon Replay Server starting...")
    settings.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Echelon Replay Server shutting down...")
    await sse_manager.shutdown()


app = FastAPI(lifespan=lifespan, title="Echelon Replay Server")


# --- SSE Endpoint ---


async def sse_event_generator(client: SSEClient):
    """
    Generate SSE events for a client.

    Handles:
    - Initial replay send (latest or missed events)
    - Keep-alive pings
    - Graceful disconnect on client cancel or shutdown
    """
    try:
        # Check for missed events (updates client.last_event_id)
        missed_count = await sse_manager.get_missed_event_count(client)
        if missed_count > 0:
            # We don't store full event data, just log that events were missed
            # Client will receive new events going forward
            logger.debug(f"Client {client.client_id} missed {missed_count} events, resuming from latest")

        if client.last_event_id == 0 or missed_count > 0:
            # New connection - send latest replay for channel
            latest = await sse_manager.get_latest_for_channel(client.channel)
            if latest:
                yield f"id: {latest.event_id}\nevent: {latest.event_type}\ndata: {latest.data}\n\n"
                client.last_event_id = latest.event_id

        # Main event loop with keep-alive
        while not client.is_cancelled and not sse_manager.is_shutdown:
            try:
                # Wait for event with timeout for keep-alive
                event_id, event_type, data = await asyncio.wait_for(
                    client.queue.get(), timeout=settings.SSE_KEEPALIVE_S
                )
                yield f"id: {event_id}\nevent: {event_type}\ndata: {data}\n\n"
                client.last_event_id = event_id
            except TimeoutError:
                # Send keep-alive comment (SSE spec: lines starting with : are comments)
                yield f": keepalive {int(time.time())}\n\n"

    except asyncio.CancelledError:
        logger.debug(f"SSE client {client.client_id} cancelled")
    except Exception as e:
        logger.error(f"SSE client {client.client_id} error: {e}")
    finally:
        await sse_manager.unregister(client)


@app.get("/events")
async def sse_endpoint(
    request: Request,
    channel: str | None = Query(None, description="Channel to subscribe to"),
):
    """
    Server-Sent Events endpoint for real-time replay streaming.

    Query params:
    - channel: Optional channel to subscribe to (default: receive all)

    Headers:
    - Last-Event-ID: Resume from this event ID (auto-set by browser on reconnect)
    """
    # Parse Last-Event-ID header for resumption
    last_event_id = 0
    last_event_header = request.headers.get("Last-Event-ID", "")
    if last_event_header:
        with suppress(ValueError):
            last_event_id = int(last_event_header)

    client = await sse_manager.register(channel=channel, last_event_id=last_event_id)

    return StreamingResponse(
        sse_event_generator(client),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# --- HTTP Endpoints ---


@app.get("/", response_class=HTMLResponse)
async def get_viewer():
    """Serves the viewer.html."""
    html_path = Path(__file__).parent / "viewer.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("viewer.html not found", status_code=404)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "clients": len(sse_manager.clients),
        "uptime_s": time.time() - _server_start_time,
    }


@app.get("/replays", response_model=list[ReplayListEntry])
async def list_replays(limit: int = settings.REPLAYS_LIST_LIMIT):
    """List available replay files."""
    return await replay_manager.list_replays_files(limit)


@app.get("/replays/{path:path}")
async def get_replay(path: str):
    """Fetch a specific replay file by path."""
    try:
        requested = Path(path)
        if requested.is_absolute():
            raise HTTPException(status_code=400, detail="Invalid path")

        resolved = (settings.RUNS_DIR / requested).resolve()

        if not resolved.is_relative_to(settings.RUNS_DIR):
            raise HTTPException(status_code=403, detail="Access denied")
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        if resolved.suffix != ".json":
            raise HTTPException(status_code=400, detail="Invalid file type")

        return await asyncio.to_thread(lambda: json.loads(resolved.read_text(encoding="utf-8")))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving replay {path}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/push", response_model=ReplayPushResponse)
async def push_replay(
    replay: dict[str, Any],
    channel: str | None = Query(None, description="Channel ID (default: derived from run metadata)"),
):
    """
    Push a new replay to connected clients.

    The replay is broadcast to:
    - All global listeners (channel=None)
    - All listeners subscribed to the specified channel
    """
    # Determine channel
    if not channel:
        run_info = replay.get("run", {})
        run_dir = run_info.get("run_dir")
        if run_dir:
            channel = Path(run_dir).name
        else:
            channel = "default"

    # Encode
    try:
        text, nbytes = replay_manager.encode(replay)
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}") from e

    summary = replay_manager.summarize(replay)

    if nbytes > settings.SSE_MAX_BROADCAST_BYTES:
        logger.warning(f"Replay too large ({nbytes} bytes) for channel {channel}. Dropping.")
        return ReplayPushResponse(
            status="too_large", bytes=nbytes, summary=summary, channel=channel, event_id=0, clients_notified=0
        )

    # Broadcast via SSE
    event_id, clients_notified = await sse_manager.broadcast(
        event_type="replay",
        data=text,
        channel=channel,
    )

    logger.info(
        f"Pushed replay to '{channel}' ({nbytes / 1024:.1f} KB, event_id={event_id}, clients={clients_notified})"
    )
    return ReplayPushResponse(
        status="ok",
        bytes=nbytes,
        summary=summary,
        channel=channel,
        event_id=event_id,
        clients_notified=clients_notified,
    )


# --- Nav Endpoints ---


def _build_world_from_data(world_data: dict[str, Any]) -> tuple[VoxelWorld, int]:
    """Build a VoxelWorld from world data dict.

    Returns (world, clearance_z).
    Raises HTTPException if world is too large.
    """
    sx, sy, sz = world_data["size"]
    if sx > 256 or sy > 256 or sz > 64:
        raise HTTPException(status_code=400, detail="World size too large")

    voxels = np.zeros((sz, sy, sx), dtype=np.uint8)
    for w in world_data.get("walls", []):
        if len(w) >= 3:
            x, y, z = w[:3]
            t = w[3] if len(w) >= 4 else 1
            if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
                voxels[z, y, x] = t

    world = VoxelWorld(voxels=voxels, voxel_size_m=world_data.get("voxel_size_m", 5.0))
    clearance = world_data.get("meta", {}).get("validator", {}).get("clearance_z", 4)
    return world, clearance


@app.post("/nav/build")
async def build_nav_graph(world_data: dict[str, Any]):
    """Computes a NavGraph for the provided world data."""
    try:
        world, clearance = _build_world_from_data(world_data)
        graph = await asyncio.to_thread(NavGraph.build, world, clearance_z=clearance)
        return graph.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to build NavGraph: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class PathRequest(BaseModel):
    world: dict[str, Any]
    start_pos: list[float]
    goal_pos: list[float]


@app.post("/nav/path")
async def plan_path(req: PathRequest):
    """Computes an A* path through the nav graph."""
    try:
        world, clearance = _build_world_from_data(req.world)
        graph = await asyncio.to_thread(NavGraph.build, world, clearance_z=clearance)
        planner = Planner(graph)

        start_node = graph.get_nearest_node(tuple(req.start_pos))  # type: ignore
        goal_node = graph.get_nearest_node(tuple(req.goal_pos))  # type: ignore

        if not start_node or not goal_node:
            return {"found": False, "error": "Start or goal position not on nav graph"}

        path_ids, stats = planner.find_path(start_node, goal_node)
        path_pos = [list(graph.nodes[nid].pos) for nid in path_ids]

        return {
            "found": stats.found,
            "path": path_pos,
            "node_ids": [list(nid) for nid in path_ids],
            "stats": {"length": stats.length, "cost": stats.cost, "visited": stats.visited_count},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to plan path: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Main ---

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Echelon Replay Server")
    parser.add_argument("--port", type=int, default=settings.PORT)
    parser.add_argument("--host", type=str, default=settings.HOST)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
