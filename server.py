from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()

# Global state to hold the last replay
latest_replay: dict[str, Any] = {}
clients: list[WebSocket] = []

@app.get("/")
async def get():
    html_path = Path(__file__).parent / "viewer.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("viewer.html not found", status_code=404)

@app.get("/replays")
async def list_replays():
    # Find all json files in runs/ directory
    root = Path("runs")
    if not root.exists():
        return []
    files = sorted(root.glob("**/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": str(p), "path": str(p)} for p in files]

@app.get("/replays/{path:path}")
async def get_replay(path: str):
    file_path = Path(path)
    # Security check: ensure we are inside the project
    if ".." in str(file_path) or file_path.is_absolute():
        return {"error": "Invalid path"}
    if not file_path.exists():
        return {"error": "File not found"}
    return json.loads(file_path.read_text())

@app.post("/push")
async def push_replay(replay: dict[str, Any]):
    global latest_replay
    latest_replay = replay
    # Broadcast to all connected clients
    for client in clients:
        try:
            await client.send_json({"type": "replay", "data": latest_replay})
        except:
            pass
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    # Send current replay immediately if exists
    if latest_replay:
        await websocket.send_json({"type": "replay", "data": latest_replay})
    
    try:
        while True:
            # Keep connection open, ignore incoming messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
