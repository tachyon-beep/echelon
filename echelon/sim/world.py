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
    GLASS = 7      # Blocks movement, not LOS. Low HP.
    REINFORCED = 8 # Indestructible SOLID.
    FOLIAGE = 9    # Blocks LOS, not movement. 0 HP.

    MATERIAL_PROPS = {
        AIR: {"hp": 0, "collides": False, "blocks_los": False},
        SOLID: {"hp": 100, "collides": True, "blocks_los": True},
        LAVA: {"hp": 0, "collides": False, "blocks_los": False},
        WATER: {"hp": 0, "collides": False, "blocks_los": False},
        HOT_DEBRIS: {"hp": 50, "collides": True, "blocks_los": True},
        KILLED_HULL: {"hp": 200, "collides": True, "blocks_los": True},
        DIRT: {"hp": 0, "collides": False, "blocks_los": False},
        GLASS: {"hp": 10, "collides": True, "blocks_los": False},
        REINFORCED: {"hp": 1000, "collides": True, "blocks_los": True},
        FOLIAGE: {"hp": 1, "collides": False, "blocks_los": True},
    }

    voxels: np.ndarray  # uint8[sz, sy, sx] (z-major)
    voxel_hp: np.ndarray | None = None # float32[sz, sy, sx]
    voxel_size_m: float = 5.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.voxel_hp is None:
            self.voxel_hp = np.zeros(self.voxels.shape, dtype=np.float32)
            for v_id, props in self.MATERIAL_PROPS.items():
                self.voxel_hp[self.voxels == v_id] = props["hp"]

    @property
    def solid(self) -> Any:
        # Backward-compatible "solid mask" view.
        return _SolidMask(self)

    @solid.setter
    def solid(self, value: Any) -> None:
        arr = np.asarray(value)
        if arr.shape != self.voxels.shape:
            raise ValueError(f"solid has shape {arr.shape}, expected {self.voxels.shape}")
        if arr.dtype == np.bool_ or arr.dtype == bool:
            self.voxels[:] = np.where(arr, self.SOLID, self.AIR).astype(np.uint8, copy=False)
        else:
            self.voxels[:] = arr.astype(np.uint8, copy=False)
        
        # Sync HP
        if self.voxel_hp is not None:
            for v_id, props in self.MATERIAL_PROPS.items():
                self.voxel_hp[self.voxels == v_id] = props["hp"]

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
        return self.MATERIAL_PROPS.get(v, {}).get("collides", False)

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
        min_ix = max(min_ix, 0); min_iy = max(min_iy, 0); min_iz = max(min_iz, 0)
        max_ix_excl = min(max_ix_excl, self.size_x)
        max_iy_excl = min(max_iy_excl, self.size_y)
        max_iz_excl = min(max_iz_excl, self.size_z)
        if min_ix >= max_ix_excl or min_iy >= max_iy_excl or min_iz >= max_iz_excl:
            return
        
        val = int(value) if isinstance(value, (int, np.integer)) else (self.SOLID if value else self.AIR)
        self.voxels[min_iz:max_iz_excl, min_iy:max_iy_excl, min_ix:max_ix_excl] = val
        if self.voxel_hp is not None:
            # Sync HP for the region
            hp_val = self.MATERIAL_PROPS.get(val, {}).get("hp", 0)
            self.voxel_hp[min_iz:max_iz_excl, min_iy:max_iy_excl, min_ix:max_ix_excl] = hp_val

    def damage_voxel(self, ix: int, iy: int, iz: int, amount: float) -> bool:
        """Subtracts HP from a voxel. Returns True if destroyed."""
        if not self.in_bounds(ix, iy, iz) or self.voxel_hp is None:
            return False
        
        v_id = self.voxels[iz, iy, ix]
        if v_id == self.AIR or v_id == self.REINFORCED:
            return False
            
        self.voxel_hp[iz, iy, ix] -= amount
        if self.voxel_hp[iz, iy, ix] <= 0:
            # Voxel destroyed!
            self.voxels[iz, iy, ix] = self.AIR
            self.voxel_hp[iz, iy, ix] = 0
            return True
        return False

    def ensure_ground_layer(self) -> None:
        if self.size_z <= 0:
            return
        layer = self.voxels[0]
        mask = (layer == self.AIR)
        layer[mask] = self.DIRT
        if self.voxel_hp is not None:
            self.voxel_hp[0][mask] = 0

    def aabb_collides(self, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
        if aabb_min[2] < 0:
            return True

        min_ix, min_iy, min_iz = np.floor(aabb_min).astype(int)
        max_ix, max_iy, max_iz = np.ceil(aabb_max).astype(int)

        s_min_ix, s_min_iy, s_min_iz = max(min_ix, 0), max(min_iy, 0), max(min_iz, 0)
        s_max_ix = min(max_ix, self.size_x)
        s_max_iy = min(max_iy, self.size_y)
        s_max_iz = min(max_iz, self.size_z)

        if s_min_ix >= s_max_ix or s_min_iy >= s_max_iy or s_min_iz >= s_max_iz:
            return False

        region = self.voxels[s_min_iz:s_max_iz, s_min_iy:s_max_iy, s_min_ix:s_max_ix]
        # Check against MATERIAL_PROPS collides flag
        # Optimization: pre-calculate collision bitmask? 
        # For now, just check all non-AIR and non-FOLIAGE etc.
        for v_id, props in self.MATERIAL_PROPS.items():
            if props["collides"]:
                if np.any(region == v_id):
                    return True
        return False

    @classmethod
    def generate(cls, config: WorldConfig, rng: np.random.Generator) -> VoxelWorld:
        # Defer to the modular layout generator (v2 architecture)
        from ..gen.layout import generate_layout
        return generate_layout(config, rng)
