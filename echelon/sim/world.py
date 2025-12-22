from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import WorldConfig


class _SolidMask:
    def __init__(self, world: VoxelWorld) -> None:
        self._world = world

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._world.voxels.shape

    @property
    def dtype(self) -> type[np.bool_]:
        return np.bool_

    @property
    def ndim(self) -> int:
        return int(self._world.voxels.ndim)

    @property
    def size(self) -> int:
        return int(self._world.voxels.size)

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        out = (self._world.voxels == self._world.SOLID) | (self._world.voxels == self._world.KILLED_HULL)
        if dtype is None:
            return out
        return out.astype(dtype, copy=False)

    def __getitem__(self, key: Any) -> Any:
        return (self._world.voxels[key] == self._world.SOLID) | (self._world.voxels[key] == self._world.KILLED_HULL)

    def __setitem__(self, key: Any, value: Any) -> None:
        if np.isscalar(value):
            self._world.voxels[key] = self._world.SOLID if bool(value) else self._world.AIR
            return

        arr = np.asarray(value)
        if arr.dtype == np.bool_ or arr.dtype == bool:
            self._world.voxels[key] = np.where(arr, self._world.SOLID, self._world.AIR).astype(
                np.uint8, copy=False
            )
            return

        self._world.voxels[key] = arr.astype(np.uint8, copy=False)

    def copy(self) -> np.ndarray:
        return np.asarray(self).copy()


@dataclass
class VoxelWorld:
    # Voxel types
    AIR = 0
    SOLID = 1
    LAVA = 2
    WATER = 3
    HOT_DEBRIS = 4
    KILLED_HULL = 5
    DIRT = 6

    voxels: np.ndarray  # uint8[sz, sy, sx] (z-major)
    voxel_size_m: float = 5.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def solid(self) -> Any:
        # Backward-compatible "solid mask" view.
        # Historically, callers treated `world.solid` as a mutable boolean ndarray.
        # Now that terrain is typed (uint8 voxels), we provide a lightweight proxy
        # that supports numpy reads and in-place boolean writes.
        return _SolidMask(self)

    @solid.setter
    def solid(self, value: Any) -> None:
        arr = np.asarray(value)
        if arr.shape != self.voxels.shape:
            raise ValueError(f"solid has shape {arr.shape}, expected {self.voxels.shape}")
        if arr.dtype == np.bool_ or arr.dtype == bool:
            self.voxels[:] = np.where(arr, self.SOLID, self.AIR).astype(np.uint8, copy=False)
            return
        self.voxels[:] = arr.astype(np.uint8, copy=False)

    @property
    def size_z(self) -> int:
        return int(self.voxels.shape[0])

    @property
    def size_y(self) -> int:
        return int(self.voxels.shape[1])

    @property
    def size_x(self) -> int:
        return int(self.voxels.shape[2])

    def in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        return 0 <= ix < self.size_x and 0 <= iy < self.size_y and 0 <= iz < self.size_z

    def get_voxel(self, ix: int, iy: int, iz: int) -> int:
        if not self.in_bounds(ix, iy, iz):
            return self.SOLID
        return int(self.voxels[iz, iy, ix])

    def is_solid_index(self, ix: int, iy: int, iz: int) -> bool:
        v = self.get_voxel(ix, iy, iz)
        return v == self.SOLID or v == self.KILLED_HULL

    def set_box_solid(
        self,
        min_ix: int,
        min_iy: int,
        min_iz: int,
        max_ix_excl: int,
        max_iy_excl: int,
        max_iz_excl: int,
        value: bool | int,
    ) -> None:
        min_ix = max(min_ix, 0)
        min_iy = max(min_iy, 0)
        min_iz = max(min_iz, 0)
        max_ix_excl = min(max_ix_excl, self.size_x)
        max_iy_excl = min(max_iy_excl, self.size_y)
        max_iz_excl = min(max_iz_excl, self.size_z)
        if min_ix >= max_ix_excl or min_iy >= max_iy_excl or min_iz >= max_iz_excl:
            return
        
        val = int(value) if isinstance(value, (int, np.integer)) else (self.SOLID if value else self.AIR)
        self.voxels[min_iz:max_iz_excl, min_iy:max_iy_excl, min_ix:max_ix_excl] = val

    def ensure_ground_layer(self) -> None:
        if self.size_z <= 0:
            return
        layer = self.voxels[0]
        layer[layer == self.AIR] = self.DIRT

    def aabb_collides(self, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
        # Only SOLID voxels (walls) cause physics collisions.
        # Below ground (z < 0) is always solid.
        if aabb_min[2] < 0:
            return True

        min_ix = int(np.floor(aabb_min[0]))
        min_iy = int(np.floor(aabb_min[1]))
        min_iz = int(np.floor(aabb_min[2]))
        max_ix = int(np.ceil(aabb_max[0]))
        max_iy = int(np.ceil(aabb_max[1]))
        max_iz = int(np.ceil(aabb_max[2]))

        # Clamp search region to world bounds.
        s_min_ix = max(min_ix, 0)
        s_min_iy = max(min_iy, 0)
        s_min_iz = max(min_iz, 0)
        s_max_ix = min(max_ix, self.size_x)
        s_max_iy = min(max_iy, self.size_y)
        s_max_iz = min(max_iz, self.size_z)

        if s_min_ix >= s_max_ix or s_min_iy >= s_max_iy or s_min_iz >= s_max_iz:
            # Entirely out of bounds (and not below ground).
            # We treat horizontal OOB as non-colliding here; the Sim handles map boundaries.
            return False

        region = self.voxels[s_min_iz:s_max_iz, s_min_iy:s_max_iy, s_min_ix:s_max_ix]
        return bool(np.any((region == self.SOLID) | (region == self.KILLED_HULL)))

    @classmethod
    def generate(cls, config: WorldConfig, rng: np.random.Generator) -> VoxelWorld:
        # Defer to the modular layout generator (v2 architecture)
        from ..gen.layout import generate_layout
        return generate_layout(config, rng)
