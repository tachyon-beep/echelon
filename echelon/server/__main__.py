# echelon/server/__main__.py
"""Entry point: python -m echelon.server"""

import argparse
import asyncio
import signal

import uvicorn

from .config import settings


async def run_server(host: str, port: int) -> None:
    """Run the server with proper shutdown handling."""
    from . import app
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
    args = parser.parse_args()

    asyncio.run(run_server(args.host, args.port))


if __name__ == "__main__":
    main()
