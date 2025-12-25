# echelon/server/__init__.py
"""Echelon Replay Server - chunked streaming for DRL training visualization."""

from __future__ import annotations

import contextlib
import gzip
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echelon.server")

_server_start_time: float = 0.0


class GzipRequestMiddleware(BaseHTTPMiddleware):
    """Decompress gzip-encoded request bodies."""

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-encoding") == "gzip":
            body = await request.body()
            with contextlib.suppress(Exception):
                request._body = gzip.decompress(body)
        return await call_next(request)


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
    from .models import HealthResponse
    from .routes import nav, push, stream
    from .sse import sse_manager

    app = FastAPI(lifespan=lifespan, title="Echelon Replay Server")

    # Add gzip decompression middleware
    app.add_middleware(GzipRequestMiddleware)

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


app = create_app()
