# echelon/server/sse.py
"""Simplified SSE client management with timeout-based disconnect."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(("", "_shutdown", ""))

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
            except (TimeoutError, asyncio.QueueFull):
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
            except TimeoutError:
                # Send keepalive
                yield f": keepalive {int(time.time())}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        await manager.unregister(client)


# Global instance
sse_manager = SSEManager()
