# echelon/server/world_cache.py
"""LRU cache for world data, keyed by content hash."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any

from .config import settings

logger = logging.getLogger("echelon.server")


class WorldCache:
    """Thread-safe LRU cache for world data."""

    def __init__(self, max_size: int | None = None) -> None:
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_size = max_size or settings.WORLD_CACHE_MAX

    @staticmethod
    def compute_hash(world: dict[str, Any]) -> str:
        """Compute deterministic hash of world data."""
        canonical = json.dumps(world, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def get(self, world_hash: str) -> dict[str, Any] | None:
        """Get world by hash, updating LRU order."""
        if world_hash in self._cache:
            self._cache.move_to_end(world_hash)
            return self._cache[world_hash]
        return None

    def put(self, world: dict[str, Any], world_hash: str | None = None) -> str:
        """Store world, return its hash."""
        if world_hash is None:
            world_hash = self.compute_hash(world)

        if world_hash in self._cache:
            self._cache.move_to_end(world_hash)
        else:
            self._cache[world_hash] = world
            while len(self._cache) > self._max_size:
                evicted_hash, _ = self._cache.popitem(last=False)
                logger.debug(f"Evicted world {evicted_hash} from cache")

        return world_hash

    def has(self, world_hash: str) -> bool:
        """Check if world exists in cache."""
        return world_hash in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# Global instance
world_cache = WorldCache()
