# echelon/server/__main__.py
"""Entry point: python -m echelon.server"""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn

from .config import settings

if TYPE_CHECKING:
    from fastapi import FastAPI


async def run_server(app: FastAPI, host: str, port: int) -> None:
    """Run the server with proper shutdown handling."""
    from .sse import sse_manager

    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)

    # Override uvicorn's default signal handling
    loop = asyncio.get_running_loop()

    def handle_exit():
        # Cancel SSE clients first so connections can close
        sse_manager._shutdown = True
        for client in list(sse_manager._clients.values()):
            client.cancel()
        sse_manager._clients.clear()
        # Then tell uvicorn to shutdown
        server.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_exit)

    await server.serve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Echelon Replay Server")
    parser.add_argument("--host", type=str, default=settings.HOST)
    parser.add_argument("--port", type=int, default=settings.PORT)
    parser.add_argument(
        "--league",
        type=Path,
        default=Path("runs/arena/league.json"),
        help="Path to league.json for arena API endpoints",
    )
    parser.add_argument(
        "--matches",
        type=Path,
        default=None,
        help="Path to matches directory for match history API",
    )
    args = parser.parse_args()

    from . import create_app

    app = create_app(league_path=args.league, matches_path=args.matches)
    asyncio.run(run_server(app, args.host, args.port))


if __name__ == "__main__":
    main()
