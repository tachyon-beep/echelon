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
    from .sse import sse_manager

    await sse_manager.shutdown()


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
        from .sse import sse_manager

        return HealthResponse(
            status="ok",
            clients=sse_manager.client_count,
            uptime_s=time.time() - _server_start_time,
        )

    return app


app = create_app()
