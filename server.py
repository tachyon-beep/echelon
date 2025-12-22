from __future__ import annotations

import asyncio
import heapq
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Configuration ---

class Settings(BaseSettings):
    ECHELON_WS_SEND_TIMEOUT_S: float = 2.0
    ECHELON_WS_MAX_BROADCAST_BYTES: int = 25 * 1024 * 1024
    ECHELON_WS_MAX_CLIENTS: int = 32
    ECHELON_REPLAYS_LIST_LIMIT: int = 200
    ECHELON_REPLAYS_CACHE_TTL_S: float = 2.0
    ECHELON_HOST: str = "0.0.0.0"
    ECHELON_PORT: int = 8090
    RUNS_DIR: Path = (Path(__file__).resolve().parent / "runs").resolve()

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("echelon.server")

# --- Data Models ---

class ReplayPushResponse(BaseModel):
    status: str
    bytes: int
    summary: Dict[str, Any]
    channel: str

class ReplayListEntry(BaseModel):
    name: str
    path: str

# --- Managers ---

class ConnectionManager:
    def __init__(self):
        # Global listeners (legacy behavior: receive everything)
        self.global_connections: Set[WebSocket] = set()
        # Channel-specific listeners
        self.channel_connections: Dict[str, Set[WebSocket]] = {}
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: Optional[str] = None):
        await websocket.accept()
        async with self.lock:
            if channel:
                if channel not in self.channel_connections:
                    self.channel_connections[channel] = set()
                self.channel_connections[channel].add(websocket)
                logger.info(f"Client connected to channel: {channel}")
            else:
                self.global_connections.add(websocket)
                logger.info("Client connected to global stream")

    async def disconnect(self, websocket: WebSocket, channel: Optional[str] = None):
        async with self.lock:
            if channel:
                if channel in self.channel_connections:
                    self.channel_connections[channel].discard(websocket)
                    if not self.channel_connections[channel]:
                        del self.channel_connections[channel]
            else:
                self.global_connections.discard(websocket)

    async def broadcast(self, message: str, channel: Optional[str] = None):
        """
        Broadcasts message to:
        1. All global connections.
        2. If channel is specified, all connections subscribed to that channel.
        """
        targets = set()
        
        async with self.lock:
            # 1. Global listeners
            targets.update(self.global_connections)
            
            # 2. Channel listeners
            if channel and channel in self.channel_connections:
                targets.update(self.channel_connections[channel])
        
        if not targets:
            return

        # Fan-out with timeout
        # We don't want one slow client to block the training loop's push request (indirectly via broadcast task)
        # But here we are in a background task usually, so it's okay-ish.
        # Still, we use wait_for to be safe.
        
        dead_sockets = []
        for connection in targets:
            try:
                await asyncio.wait_for(connection.send_text(message), timeout=settings.ECHELON_WS_SEND_TIMEOUT_S)
            except (WebSocketDisconnect, asyncio.TimeoutError, Exception) as e:
                # logger.warning(f"Failed to send to client: {e}")
                dead_sockets.append(connection)

        if dead_sockets:
            await self._cleanup_dead(dead_sockets, channel)

    async def _cleanup_dead(self, dead: List[WebSocket], channel: Optional[str]):
        async with self.lock:
            for ws in dead:
                self.global_connections.discard(ws)
                if channel and channel in self.channel_connections:
                    self.channel_connections[channel].discard(ws)

class ReplayManager:
    def __init__(self):
        self.latest_replays: Dict[str, Tuple[str, int]] = {} # channel -> (json_text, size)
        self.lock = asyncio.Lock()
        
        # Cache for file listing
        self.replays_list_cache: Dict[int, Tuple[float, List[Dict[str, str]]]] = {}
        self.list_cache_lock = asyncio.Lock()

    def summarize(self, replay: Dict[str, Any]) -> Dict[str, Any]:
        world = replay.get("world") if isinstance(replay, dict) else None
        frames = replay.get("frames") if isinstance(replay, dict) else None
        out: Dict[str, Any] = {
            "frames": int(len(frames)) if isinstance(frames, list) else None,
        }
        if isinstance(world, dict):
            out["seed"] = world.get("seed")
            out["size"] = world.get("size")
        return out

    def encode(self, replay: Dict[str, Any]) -> Tuple[str, int]:
        msg = {"type": "replay", "data": replay}
        text = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
        return text, len(text.encode("utf-8"))

    async def set_latest(self, channel: str, text: str, size: int):
        async with self.lock:
            self.latest_replays[channel] = (text, size)
            # Also set 'default' if it's the most recent global activity?
            # For now, let's keep 'default' as a special key if we want.
            # But the viewer logic usually just waits for the next push.
            # If a viewer connects late, we want to send them the latest from their channel.
            
    async def get_latest(self, channel: str) -> Optional[str]:
        async with self.lock:
            if channel in self.latest_replays:
                return self.latest_replays[channel][0]
            # Fallback: if requesting "default" or None, maybe return the most recently updated one?
            # For now, simple strict key lookup.
            return None

    async def get_any_latest(self) -> Optional[str]:
        """Returns the most recently pushed replay from any channel."""
        async with self.lock:
            if not self.latest_replays:
                return None
            # This is a bit arbitrary since we don't store timestamps, 
            # but usually the last one inserted is at the end of the dict in modern Python.
            key = list(self.latest_replays.keys())[-1]
            return self.latest_replays[key][0]

    async def list_replays_files(self, limit: int) -> List[Dict[str, str]]:
        limit = int(max(1, min(int(limit), 50_000)))
        now = time.monotonic()
        
        async with self.list_cache_lock:
            hit = self.replays_list_cache.get(limit)
            if hit is not None:
                ts, cached = hit
                if now - ts <= settings.ECHELON_REPLAYS_CACHE_TTL_S:
                    return cached

        # Run in threadpool
        return await asyncio.to_thread(self._scan_fs, limit, now)

    def _scan_fs(self, limit: int, now: float) -> List[Dict[str, str]]:
        root = settings.RUNS_DIR
        if not root.exists():
            return []
        
        items: List[Tuple[float, Path]] = []
        for p in root.rglob("*.json"):
            try:
                resolved = p.resolve()
                if not (resolved.is_relative_to(root) and resolved.is_file()):
                    continue
                mtime = float(resolved.stat().st_mtime)
            except OSError:
                continue
            
            if len(items) < limit:
                heapq.heappush(items, (mtime, p))
            elif mtime > items[0][0]:
                heapq.heapreplace(items, (mtime, p))
        
        items.sort(key=lambda t: t[0], reverse=True)
        result = [{"name": p.relative_to(root).as_posix(), "path": p.relative_to(root).as_posix()} for _, p in items]
        
        # Update cache (async update is fine, but we are in a thread here, so we can't easily touch the async lock)
        # We'll just return and let the caller update.
        return result


# --- Global State & Lifespan ---

conn_manager = ConnectionManager()
replay_manager = ReplayManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Server starting up...")
    settings.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown
    logger.info("Server shutting down...")

app = FastAPI(lifespan=lifespan, title="Echelon Replay Server")

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_viewer():
    html_path = Path(__file__).parent / "viewer.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("viewer.html not found", status_code=404)

@app.get("/replays", response_model=List[ReplayListEntry])
async def list_replays(limit: int = settings.ECHELON_REPLAYS_LIST_LIMIT):
    files = await replay_manager.list_replays_files(limit)
    # Update cache
    async with replay_manager.list_cache_lock:
        replay_manager.replays_list_cache[limit] = (time.monotonic(), files)
        if len(replay_manager.replays_list_cache) > 16:
            replay_manager.replays_list_cache.clear()
    return files

@app.get("/replays/{path:path}")
async def get_replay(path: str):
    # Security check
    try:
        requested = Path(path)
        if requested.is_absolute():
             raise HTTPException(status_code=400, detail="Invalid path")
        
        # Handle cases where client might send 'runs/foo/bar.json' or just 'foo/bar.json'
        # We want to be relative to settings.RUNS_DIR
        resolved = (settings.RUNS_DIR / requested).resolve()
        
        if not resolved.is_relative_to(settings.RUNS_DIR):
            raise HTTPException(status_code=403, detail="Access denied")
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        if resolved.suffix != ".json":
            raise HTTPException(status_code=400, detail="Invalid file type")
            
        return await asyncio.to_thread(lambda: json.loads(resolved.read_text(encoding="utf-8")))
    except Exception as e:
        logger.error(f"Error serving replay {path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/push", response_model=ReplayPushResponse)
async def push_replay(
    replay: Dict[str, Any], 
    channel: Optional[str] = Query(None, description="Specific channel ID. If None, derived from run metadata.")
):
    # Determine channel
    if not channel:
        run_info = replay.get("run", {})
        # Try run_dir basename
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
        return {"status": "error", "bytes": 0, "summary": {}, "channel": channel}

    summary = replay_manager.summarize(replay)

    if nbytes > settings.ECHELON_WS_MAX_BROADCAST_BYTES:
        logger.warning(f"Replay too large ({nbytes} bytes) for channel {channel}. Dropping broadcast.")
        return {"status": "too_large", "bytes": nbytes, "summary": summary, "channel": channel}

    # Store latest
    await replay_manager.set_latest(channel, text, nbytes)
    
    # Broadcast (background task to avoid blocking response?)
    # For simplicity and flow control, we await it here. It uses wait_for internally.
    asyncio.create_task(conn_manager.broadcast(text, channel))
    
    logger.info(f"Pushed replay to channel '{channel}' ({nbytes/1024:.1f} KB). Summary: {summary}")
    return {"status": "ok", "bytes": nbytes, "summary": summary, "channel": channel}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, channel: Optional[str] = None):
    await conn_manager.connect(websocket, channel)
    
    # Send initial state
    # If channel specified, get that. If not, get 'default' or ANY latest.
    payload = None
    if channel:
        payload = await replay_manager.get_latest(channel)
    else:
        # Global listener -> send the absolute latest from any channel to give immediate feedback
        payload = await replay_manager.get_any_latest()
        
    if payload:
        try:
            await websocket.send_text(payload)
        except Exception:
            pass # Disconnect will be handled in loop

    try:
        while True:
            # Keep alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await conn_manager.disconnect(websocket, channel)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await conn_manager.disconnect(websocket, channel)

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=settings.ECHELON_PORT)
    parser.add_argument("--host", type=str, default=settings.ECHELON_HOST)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)