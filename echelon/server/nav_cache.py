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
        """Get cached NavGraph or build it. Deduplicates concurrent builds."""
        # Check if already cached
        async with self._lock:
            if world_hash in self._cache:
                self._cache.move_to_end(world_hash)
                return self._cache[world_hash]

            # Check if another task is already building
            if world_hash in self._building:
                event = self._building[world_hash]
                is_builder = False
            else:
                event = asyncio.Event()
                self._building[world_hash] = event
                is_builder = True

        # If we're not the builder, wait for the builder to finish
        if not is_builder:
            await event.wait()
            # Retry the entire operation to handle builder failure correctly
            return await self.get_or_build(world_hash, world_data)

        # We're the builder - build outside lock
        try:
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
