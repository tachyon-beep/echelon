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
