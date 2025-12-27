# Replay Server Redesign - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace broken monolithic server.py with chunked streaming architecture that handles 10-50 MB replays without crashing.

**Architecture:** Modular FastAPI server in `echelon/server/` with chunked replay protocol. Training script sends ~100-frame chunks. Viewer assembles incrementally. NavGraph cached by world hash.

**Tech Stack:** FastAPI, SSE, asyncio, Three.js

---

## Task 1: Server Package Skeleton

**Files:**
- Create: `echelon/server/__init__.py`
- Create: `echelon/server/config.py`
- Create: `echelon/server/models.py`

**Step 1: Create server package directory**

```bash
mkdir -p echelon/server/routes
```

**Step 2: Write config.py**

```python
# echelon/server/config.py
"""Server configuration with sensible defaults for LAN use."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server settings, overridable via environment variables."""

    # SSE
    SSE_MAX_CLIENTS: int = 16
    SSE_KEEPALIVE_S: float = 15.0
    SSE_BROADCAST_TIMEOUT_S: float = 1.0

    # Caches
    WORLD_CACHE_MAX: int = 32
    NAV_CACHE_MAX: int = 8

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8090
    RUNS_DIR: Path = Path("runs")

    model_config = SettingsConfigDict(env_prefix="ECHELON_", env_file=".env", extra="ignore")


settings = Settings()
```

**Step 3: Write models.py**

```python
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


class PathRequest(BaseModel):
    """Request for A* pathfinding."""

    world_hash: str
    start_pos: list[float]
    goal_pos: list[float]
```

**Step 4: Write __init__.py (app factory)**

```python
# echelon/server/__init__.py
"""Echelon Replay Server - chunked streaming for DRL training visualization."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echelon.server")

_server_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _server_start_time
    _server_start_time = time.time()
    logger.info(f"Echelon Replay Server starting on {settings.HOST}:{settings.PORT}")
    settings.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Echelon Replay Server shutting down...")
    # SSE cleanup will be added in Task 2


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(lifespan=lifespan, title="Echelon Replay Server")

    @app.get("/", response_class=HTMLResponse)
    async def get_viewer():
        """Serve the viewer HTML."""
        html_path = Path(__file__).parent.parent.parent / "viewer.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>viewer.html not found</h1>", status_code=404)

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        from .models import HealthResponse

        return HealthResponse(
            status="ok",
            clients=0,  # Will be updated in Task 2
            uptime_s=time.time() - _server_start_time,
        )

    return app


app = create_app()
```

**Step 5: Verify it runs**

```bash
cd /home/john/echelon && uv run python -c "from echelon.server import app; print('OK')"
```

Expected: `OK`

**Step 6: Commit**

```bash
git add echelon/server/
git commit -m "feat(server): add package skeleton with config and models"
```

---

## Task 2: SSE Manager

**Files:**
- Create: `echelon/server/sse.py`

**Step 1: Write sse.py**

```python
# echelon/server/sse.py
"""Simplified SSE client management with timeout-based disconnect."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from .config import settings

logger = logging.getLogger("echelon.server")


@dataclass
class SSEClient:
    """A connected SSE client."""

    client_id: str
    queue: asyncio.Queue[tuple[str, str, str]]  # (event_id, event_type, data)
    connected_at: float = field(default_factory=time.time)
    _cancelled: bool = field(default=False, repr=False)

    def cancel(self) -> None:
        """Signal this client to disconnect."""
        self._cancelled = True
        # Put poison pill to wake blocked queue.get()
        try:
            self.queue.put_nowait(("", "_shutdown", ""))
        except asyncio.QueueFull:
            pass

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


class SSEManager:
    """
    Manages SSE client connections.

    - Hard limit on clients (evict oldest if exceeded)
    - Timeout-based broadcast (disconnect slow clients)
    - Clean shutdown
    """

    def __init__(self) -> None:
        self._clients: dict[str, SSEClient] = {}
        self._lock = asyncio.Lock()
        self._event_counter = 0
        self._shutdown = False

        # Latest event per replay_id for new connections
        self._latest_replay: dict[str, tuple[str, str]] | None = None

    async def register(self) -> SSEClient:
        """Register a new SSE client."""
        async with self._lock:
            # Evict oldest if at capacity
            if len(self._clients) >= settings.SSE_MAX_CLIENTS:
                oldest = min(self._clients.values(), key=lambda c: c.connected_at)
                oldest.cancel()
                del self._clients[oldest.client_id]
                logger.warning(f"Evicted oldest client {oldest.client_id}")

            client_id = uuid.uuid4().hex[:8]
            client = SSEClient(
                client_id=client_id,
                queue=asyncio.Queue(maxsize=8),
            )
            self._clients[client_id] = client
            logger.info(f"SSE client {client_id} connected (total={len(self._clients)})")
            return client

    async def unregister(self, client: SSEClient) -> None:
        """Unregister an SSE client."""
        async with self._lock:
            if client.client_id in self._clients:
                del self._clients[client.client_id]
                logger.info(f"SSE client {client.client_id} disconnected (total={len(self._clients)})")

    async def broadcast(self, event_type: str, data: str) -> int:
        """
        Broadcast event to all clients.

        Returns number of clients notified.
        Disconnects clients that fail to receive within timeout.
        """
        self._event_counter += 1
        event_id = str(self._event_counter)

        async with self._lock:
            clients = list(self._clients.values())

        notified = 0
        failed: list[str] = []

        for client in clients:
            if client.is_cancelled:
                continue
            try:
                await asyncio.wait_for(
                    client.queue.put((event_id, event_type, data)),
                    timeout=settings.SSE_BROADCAST_TIMEOUT_S,
                )
                notified += 1
            except (asyncio.TimeoutError, asyncio.QueueFull):
                logger.warning(f"Client {client.client_id} too slow, disconnecting")
                client.cancel()
                failed.append(client.client_id)

        # Clean up failed clients
        if failed:
            async with self._lock:
                for client_id in failed:
                    self._clients.pop(client_id, None)

        return notified

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown

    async def shutdown(self) -> None:
        """Gracefully disconnect all clients."""
        self._shutdown = True
        async with self._lock:
            for client in self._clients.values():
                client.cancel()
            self._clients.clear()
        logger.info("SSE manager shutdown complete")


async def sse_event_generator(client: SSEClient, manager: SSEManager) -> AsyncIterator[str]:
    """Generate SSE events for a client."""
    try:
        while not client.is_cancelled and not manager.is_shutdown:
            try:
                event_id, event_type, data = await asyncio.wait_for(
                    client.queue.get(),
                    timeout=settings.SSE_KEEPALIVE_S,
                )
                # Check for shutdown poison pill
                if event_type == "_shutdown":
                    break
                yield f"id: {event_id}\nevent: {event_type}\ndata: {data}\n\n"
            except asyncio.TimeoutError:
                # Send keepalive
                yield f": keepalive {int(time.time())}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        await manager.unregister(client)


# Global instance
sse_manager = SSEManager()
```

**Step 2: Update __init__.py to integrate SSE shutdown**

In `echelon/server/__init__.py`, update the lifespan:

```python
# Add to lifespan, before yield:
from .sse import sse_manager

# After yield (shutdown):
await sse_manager.shutdown()
```

**Step 3: Verify import**

```bash
cd /home/john/echelon && uv run python -c "from echelon.server.sse import sse_manager; print('OK')"
```

**Step 4: Commit**

```bash
git add echelon/server/
git commit -m "feat(server): add SSE manager with timeout-based disconnect"
```

---

## Task 3: World Cache

**Files:**
- Create: `echelon/server/world_cache.py`

**Step 1: Write world_cache.py**

```python
# echelon/server/world_cache.py
"""LRU cache for world data, keyed by content hash."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any

from .config import settings

logger = logging.getLogger("echelon.server")


class WorldCache:
    """Thread-safe LRU cache for world data."""

    def __init__(self, max_size: int | None = None) -> None:
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_size = max_size or settings.WORLD_CACHE_MAX

    @staticmethod
    def compute_hash(world: dict[str, Any]) -> str:
        """Compute deterministic hash of world data."""
        canonical = json.dumps(world, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def get(self, world_hash: str) -> dict[str, Any] | None:
        """Get world by hash, updating LRU order."""
        if world_hash in self._cache:
            self._cache.move_to_end(world_hash)
            return self._cache[world_hash]
        return None

    def put(self, world: dict[str, Any], world_hash: str | None = None) -> str:
        """Store world, return its hash."""
        if world_hash is None:
            world_hash = self.compute_hash(world)

        if world_hash in self._cache:
            self._cache.move_to_end(world_hash)
        else:
            self._cache[world_hash] = world
            while len(self._cache) > self._max_size:
                evicted_hash, _ = self._cache.popitem(last=False)
                logger.debug(f"Evicted world {evicted_hash} from cache")

        return world_hash

    def has(self, world_hash: str) -> bool:
        """Check if world exists in cache."""
        return world_hash in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# Global instance
world_cache = WorldCache()
```

**Step 2: Verify**

```bash
cd /home/john/echelon && uv run python -c "
from echelon.server.world_cache import world_cache, WorldCache
w = {'size': [10, 10, 5], 'walls': []}
h = world_cache.put(w)
assert world_cache.has(h)
assert world_cache.get(h) == w
print('OK')
"
```

**Step 3: Commit**

```bash
git add echelon/server/world_cache.py
git commit -m "feat(server): add LRU world cache"
```

---

## Task 4: NavGraph Cache

**Files:**
- Create: `echelon/server/nav_cache.py`

**Step 1: Write nav_cache.py**

```python
# echelon/server/nav_cache.py
"""LRU cache for NavGraph instances, keyed by world hash."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any

import numpy as np

from echelon.nav.graph import NavGraph
from echelon.sim.world import VoxelWorld

from .config import settings

logger = logging.getLogger("echelon.server")


class NavGraphCache:
    """Async-safe LRU cache for NavGraph instances."""

    def __init__(self, max_size: int | None = None) -> None:
        self._cache: OrderedDict[str, NavGraph] = OrderedDict()
        self._max_size = max_size or settings.NAV_CACHE_MAX
        self._lock = asyncio.Lock()
        self._building: dict[str, asyncio.Event] = {}

    async def get_or_build(
        self,
        world_hash: str,
        world_data: dict[str, Any],
    ) -> NavGraph:
        """Get cached NavGraph or build it."""
        # Fast path: already cached
        async with self._lock:
            if world_hash in self._cache:
                self._cache.move_to_end(world_hash)
                return self._cache[world_hash]

            # Check if another task is building
            if world_hash in self._building:
                event = self._building[world_hash]
        else:
            # We'll build it - create event for others to wait on
            async with self._lock:
                if world_hash in self._cache:
                    self._cache.move_to_end(world_hash)
                    return self._cache[world_hash]
                event = asyncio.Event()
                self._building[world_hash] = event

            try:
                # Build outside lock (expensive)
                graph = await self._build_graph(world_data)

                async with self._lock:
                    self._cache[world_hash] = graph
                    while len(self._cache) > self._max_size:
                        evicted_hash, _ = self._cache.popitem(last=False)
                        logger.debug(f"Evicted NavGraph {evicted_hash}")
                    return graph
            finally:
                async with self._lock:
                    self._building.pop(world_hash, None)
                    event.set()

        # Wait for other task to finish building
        await event.wait()
        async with self._lock:
            return self._cache[world_hash]

    async def _build_graph(self, world_data: dict[str, Any]) -> NavGraph:
        """Build NavGraph in thread pool."""
        sx, sy, sz = world_data["size"]
        voxels = np.zeros((sz, sy, sx), dtype=np.uint8)

        for w in world_data.get("walls", []):
            if len(w) >= 3:
                x, y, z = w[:3]
                t = w[3] if len(w) >= 4 else 1
                if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
                    voxels[z, y, x] = t

        world = VoxelWorld(voxels=voxels, voxel_size_m=world_data.get("voxel_size_m", 5.0))
        clearance = world_data.get("meta", {}).get("validator", {}).get("clearance_z", 4)

        return await asyncio.to_thread(NavGraph.build, world, clearance_z=clearance)

    def __len__(self) -> int:
        return len(self._cache)


# Global instance
nav_cache = NavGraphCache()
```

**Step 2: Verify**

```bash
cd /home/john/echelon && uv run python -c "from echelon.server.nav_cache import nav_cache; print('OK')"
```

**Step 3: Commit**

```bash
git add echelon/server/nav_cache.py
git commit -m "feat(server): add async NavGraph cache with dedup"
```

---

## Task 5: Routes - Stream (SSE Endpoint)

**Files:**
- Create: `echelon/server/routes/__init__.py`
- Create: `echelon/server/routes/stream.py`

**Step 1: Create routes package**

```python
# echelon/server/routes/__init__.py
"""API route modules."""
```

**Step 2: Write stream.py**

```python
# echelon/server/routes/stream.py
"""SSE streaming endpoint."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..sse import sse_event_generator, sse_manager

router = APIRouter()


@router.get("/events")
async def sse_endpoint():
    """
    Server-Sent Events endpoint for live replay streaming.

    Events:
    - replay_start: New replay beginning
    - replay_chunk: Frame data chunk
    - replay_end: Replay complete
    """
    client = await sse_manager.register()

    return StreamingResponse(
        sse_event_generator(client, sse_manager),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

**Step 3: Commit**

```bash
git add echelon/server/routes/
git commit -m "feat(server): add SSE stream endpoint"
```

---

## Task 6: Routes - Push (Chunked Replay)

**Files:**
- Create: `echelon/server/routes/push.py`

**Step 1: Write push.py**

```python
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
```

**Step 2: Commit**

```bash
git add echelon/server/routes/push.py
git commit -m "feat(server): add chunked push endpoint with world cache"
```

---

## Task 7: Routes - Nav (Cached Pathfinding)

**Files:**
- Create: `echelon/server/routes/nav.py`

**Step 1: Write nav.py**

```python
# echelon/server/routes/nav.py
"""Navigation graph and pathfinding endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from echelon.nav.planner import Planner

from ..models import PathRequest
from ..nav_cache import nav_cache
from ..world_cache import world_cache

logger = logging.getLogger("echelon.server")

router = APIRouter(prefix="/nav")


@router.post("/graph")
async def get_nav_graph(request: dict):
    """Get or build NavGraph for a world (cached by hash)."""
    world_hash = request.get("world_hash")
    if not world_hash:
        raise HTTPException(status_code=400, detail="world_hash required")

    world_data = world_cache.get(world_hash)
    if world_data is None:
        raise HTTPException(status_code=404, detail=f"World {world_hash} not in cache")

    try:
        graph = await nav_cache.get_or_build(world_hash, world_data)
        return graph.to_dict()
    except Exception as e:
        logger.error(f"Failed to build NavGraph: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/path")
async def plan_path(req: PathRequest):
    """Compute A* path using cached NavGraph."""
    world_data = world_cache.get(req.world_hash)
    if world_data is None:
        raise HTTPException(status_code=404, detail=f"World {req.world_hash} not in cache")

    try:
        graph = await nav_cache.get_or_build(req.world_hash, world_data)
        planner = Planner(graph)

        start_node = graph.get_nearest_node(tuple(req.start_pos))
        goal_node = graph.get_nearest_node(tuple(req.goal_pos))

        if not start_node or not goal_node:
            return {"found": False, "error": "Start or goal not on nav graph"}

        path_ids, stats = planner.find_path(start_node, goal_node)
        path_pos = [list(graph.nodes[nid].pos) for nid in path_ids]

        return {
            "found": stats.found,
            "path": path_pos,
            "node_ids": [list(nid) for nid in path_ids],
            "stats": {
                "length": stats.length,
                "cost": stats.cost,
                "visited": stats.visited_count,
            },
        }
    except Exception as e:
        logger.error(f"Pathfinding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
```

**Step 2: Commit**

```bash
git add echelon/server/routes/nav.py
git commit -m "feat(server): add cached nav graph and pathfinding endpoints"
```

---

## Task 8: Wire Up Routes

**Files:**
- Modify: `echelon/server/__init__.py`

**Step 1: Update __init__.py to include all routes**

Replace `create_app()` function:

```python
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from .models import HealthResponse
    from .routes import nav, push, stream
    from .sse import sse_manager

    app = FastAPI(lifespan=lifespan, title="Echelon Replay Server")

    # Include route modules
    app.include_router(stream.router)
    app.include_router(push.router)
    app.include_router(nav.router)

    @app.get("/", response_class=HTMLResponse)
    async def get_viewer():
        """Serve the viewer HTML."""
        html_path = Path(__file__).parent.parent.parent / "viewer.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>viewer.html not found</h1>", status_code=404)

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            clients=sse_manager.client_count,
            uptime_s=time.time() - _server_start_time,
        )

    return app
```

**Step 2: Add GzipMiddleware for compressed pushes**

Add at top of `__init__.py`:

```python
import gzip
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class GzipRequestMiddleware(BaseHTTPMiddleware):
    """Decompress gzip-encoded request bodies."""

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-encoding") == "gzip":
            body = await request.body()
            try:
                request._body = gzip.decompress(body)
            except Exception:
                pass  # Let it fail later with better error
        return await call_next(request)
```

And in `create_app()`:

```python
app.add_middleware(GzipRequestMiddleware)
```

**Step 3: Verify server starts**

```bash
cd /home/john/echelon && timeout 3 uv run python -m uvicorn echelon.server:app --host 127.0.0.1 --port 8091 || true
```

Expected: Server starts, then timeout kills it.

**Step 4: Commit**

```bash
git add echelon/server/
git commit -m "feat(server): wire up all routes and add gzip middleware"
```

---

## Task 9: Add Server Entry Point

**Files:**
- Create: `echelon/server/__main__.py`

**Step 1: Write __main__.py**

```python
# echelon/server/__main__.py
"""Entry point: python -m echelon.server"""

import argparse

import uvicorn

from .config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Echelon Replay Server")
    parser.add_argument("--host", type=str, default=settings.HOST)
    parser.add_argument("--port", type=int, default=settings.PORT)
    args = parser.parse_args()

    uvicorn.run("echelon.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

**Step 2: Verify**

```bash
cd /home/john/echelon && timeout 3 uv run python -m echelon.server --port 8091 || true
```

**Step 3: Commit**

```bash
git add echelon/server/__main__.py
git commit -m "feat(server): add __main__.py entry point"
```

---

## Task 10: Update Training Script

**Files:**
- Modify: `scripts/train_ppo.py` (lines 178-230)

**Step 1: Replace push_replay function**

Find and replace the existing `push_replay` function with:

```python
def push_replay(url: str, replay: dict, chunk_size: int = 100) -> None:
    """Push replay in chunks to server.

    Sends world once (deduplicated by hash), then streams frames in chunks.
    Each chunk is ~1-2 MB, keeping memory bounded on both ends.
    """
    if not url:
        return

    import gzip
    import hashlib
    import uuid

    import requests

    try:
        base_url = url.rsplit("/push", 1)[0]

        # 1. Push world if not cached
        world = replay.get("world")
        if not world:
            logger.warning("Replay has no world data, skipping push")
            return

        canonical = json.dumps(world, sort_keys=True, separators=(",", ":"))
        world_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]

        # Check server cache
        try:
            head_resp = requests.head(f"{base_url}/worlds/{world_hash}", timeout=2)
            world_cached = head_resp.status_code == 200
        except Exception:
            world_cached = False

        if not world_cached:
            world_data = gzip.compress(json.dumps(world).encode())
            requests.put(
                f"{base_url}/worlds/{world_hash}",
                data=world_data,
                headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
                timeout=10,
            )

        # 2. Push chunks
        frames = replay.get("frames", [])
        if not frames:
            return

        replay_id = uuid.uuid4().hex[:12]
        chunk_count = (len(frames) + chunk_size - 1) // chunk_size

        for i in range(chunk_count):
            chunk = {
                "replay_id": replay_id,
                "world_ref": world_hash,
                "chunk_index": i,
                "chunk_count": chunk_count,
                "frames": frames[i * chunk_size : (i + 1) * chunk_size],
            }
            if i == 0:
                chunk["meta"] = {
                    "episode": replay.get("episode"),
                    "update": replay.get("update"),
                }

            requests.post(url, json=chunk, timeout=10)

    except requests.RequestException as e:
        logger.warning(f"Failed to push replay chunk: {e}")
    except Exception as e:
        logger.error(f"Unexpected error pushing replay: {e}")
```

**Step 2: Remove old _pushed_worlds global if present**

Search for `_pushed_worlds` and remove it.

**Step 3: Verify lint**

```bash
cd /home/john/echelon && uv run ruff check scripts/train_ppo.py
```

**Step 4: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(train): update push_replay to use chunked protocol"
```

---

## Task 11: Rewrite Viewer HTML

**Files:**
- Rewrite: `viewer.html`

**Step 1: Write simplified viewer**

This is a complete rewrite. Key changes:
- Remove: file upload, server file list, channel selector
- Add: chunked replay assembly, loading progress
- Simplify: connection handling with exponential backoff

The new viewer should be ~600 lines instead of 1269.

Create the new viewer with these sections:
1. Minimal HTML structure (connection indicator, viewport, controls, progress)
2. Three.js setup (same camera, lights, grid)
3. SSE connection with reconnect backoff
4. Chunked replay assembly (buffer chunks, assemble when complete)
5. Scene building (keep existing mesh/material code)
6. Playback controls (simplified)
7. Nav graph toggle (use new /nav/graph endpoint with world_hash)

**Step 2: Key JavaScript changes**

```javascript
// Chunked replay state
let currentReplay = null;  // {id, world, chunks: Map<index, frames>, chunkCount, complete}
let reconnectDelay = 1000;

function connectEventStream() {
    const eventSource = new EventSource('/events');

    eventSource.onopen = () => {
        updateStatus('connected');
        reconnectDelay = 1000;  // Reset on success
    };

    eventSource.addEventListener('replay_start', (e) => {
        const data = JSON.parse(e.data);
        currentReplay = {
            id: data.replay_id,
            world: data.world,
            worldHash: data.world_ref,
            chunks: new Map(),
            chunkCount: data.chunk_count,
            complete: false,
            meta: data.meta,
        };
        setupWorld(data.world);
        updateProgress(0, data.chunk_count);
    });

    eventSource.addEventListener('replay_chunk', (e) => {
        const data = JSON.parse(e.data);
        if (currentReplay && data.replay_id === currentReplay.id) {
            currentReplay.chunks.set(data.chunk_index, data.frames);
            updateProgress(currentReplay.chunks.size, currentReplay.chunkCount);
            // Start playback as soon as first chunk arrives
            if (data.chunk_index === 0) {
                assembleAndPlay();
            }
        }
    });

    eventSource.addEventListener('replay_end', (e) => {
        const data = JSON.parse(e.data);
        if (currentReplay && data.replay_id === currentReplay.id) {
            currentReplay.complete = true;
            assembleAndPlay();  // Final assembly
        }
    });

    eventSource.onerror = () => {
        updateStatus('disconnected');
        setTimeout(connectEventStream, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 10000);
    };
}

function assembleAndPlay() {
    if (!currentReplay) return;

    // Assemble frames in order
    const frames = [];
    for (let i = 0; i < currentReplay.chunkCount; i++) {
        const chunk = currentReplay.chunks.get(i);
        if (chunk) frames.push(...chunk);
        else break;  // Gap - stop here
    }

    if (frames.length > 0) {
        replayData = { world: currentReplay.world, frames };
        if (!isPlaying) {
            setupReplay();
            isPlaying = true;
        }
        updateMaxTime();
    }
}
```

**Step 3: Commit**

```bash
git add viewer.html
git commit -m "feat(viewer): rewrite for chunked streaming, strip unused UI"
```

---

## Task 12: Delete Old server.py

**Files:**
- Delete: `server.py`

**Step 1: Remove old server**

```bash
rm server.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "chore: remove old monolithic server.py"
```

---

## Task 13: Integration Test

**Step 1: Start new server**

```bash
cd /home/john/echelon && uv run python -m echelon.server --port 8090 &
SERVER_PID=$!
sleep 2
```

**Step 2: Test health endpoint**

```bash
curl http://127.0.0.1:8090/health
```

Expected: `{"status":"ok","clients":0,"uptime_s":...}`

**Step 3: Test world push**

```bash
curl -X PUT http://127.0.0.1:8090/worlds/testhash123456 \
  -H "Content-Type: application/json" \
  -d '{"size":[10,10,5],"walls":[]}'
```

Expected: Error (hash mismatch) or success with correct hash

**Step 4: Stop server**

```bash
kill $SERVER_PID
```

**Step 5: Run smoke test with new server**

```bash
cd /home/john/echelon
uv run python -m echelon.server --port 8090 &
sleep 2
uv run python scripts/smoke.py --episodes 1 --packs-per-team 1 --size 40 --mode full
kill %1
```

**Step 6: Final commit**

```bash
git add -A
git commit -m "test: verify new server works with smoke test"
```

---

## Success Criteria

- [ ] Server starts without errors: `python -m echelon.server`
- [ ] Health endpoint responds: `GET /health`
- [ ] World caching works: `PUT /worlds/{hash}` + `HEAD /worlds/{hash}`
- [ ] Chunked push works: `POST /push` with chunk payload
- [ ] SSE streams events: `GET /events` receives replay_start/chunk/end
- [ ] Viewer shows live replays
- [ ] NavGraph cached (second /nav/path call is fast)
- [ ] No memory growth over 10+ replays
- [ ] Clean shutdown (no hang)
