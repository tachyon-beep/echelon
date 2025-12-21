from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()

# Directory that replay files are allowed to be served from.
RUNS_DIR = (Path(__file__).resolve().parent / "runs").resolve()

# Global state to hold the last replay
latest_replay: dict[str, Any] = {}
clients: list[WebSocket] = []

def _resolve_replay_path(path: str) -> Path | None:
    requested = Path(path)
    if requested.is_absolute():
        return None
    if requested.parts and requested.parts[0] == RUNS_DIR.name:
        requested = Path(*requested.parts[1:])
    resolved = (RUNS_DIR / requested).resolve()
    if not resolved.is_relative_to(RUNS_DIR):
        return None
    return resolved

@app.get("/")
async def get():
    html_path = Path(__file__).parent / "viewer.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("viewer.html not found", status_code=404)

@app.get("/replays")
async def list_replays():
    # Find all json files in runs/ directory
    root = RUNS_DIR
    if not root.exists():
        return []
    files: list[Path] = []
    for p in root.glob("**/*.json"):
        resolved = p.resolve()
        if resolved.is_relative_to(root) and resolved.is_file():
            files.append(p)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.relative_to(root).as_posix(), "path": p.relative_to(root).as_posix()} for p in files]

@app.get("/replays/{path:path}")
async def get_replay(path: str):
    file_path = _resolve_replay_path(path)
    # Security check: ensure we are inside runs/ and not following symlinks outside.
    if file_path is None:
        return {"error": "Invalid path"}
    if file_path.suffix != ".json":
        return {"error": "Invalid replay file"}
    if not file_path.exists() or not file_path.is_file():
        return {"error": "File not found"}
    return json.loads(file_path.read_text())

@app.post("/push")
async def push_replay(replay: dict[str, Any]):
    global latest_replay
    latest_replay = replay
    print(f"Received push. Broadcasting to {len(clients)} clients.")
    # Broadcast to all connected clients
    dead_clients = []
    for client in clients:
        try:
            await client.send_json({"type": "replay", "data": latest_replay})
        except Exception as e:
            print(f"Failed to send to client: {e}")
            dead_clients.append(client)
    
    for dead in dead_clients:
        if dead in clients:
            clients.remove(dead)
            
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"New WebSocket client connected. Total: {len(clients) + 1}")
    clients.append(websocket)
    # Send current replay immediately if exists
    if latest_replay:
        try:
            await websocket.send_json({"type": "replay", "data": latest_replay})
        except:
            pass
    
    try:
        while True:
            # Keep connection open, ignore incoming messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
        if websocket in clients:
            clients.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in clients:
            clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
