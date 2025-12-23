from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numba

from .world import VoxelWorld


@dataclass(frozen=True)
class RaycastHit:
    blocked: bool
    blocked_voxel: tuple[int, int, int] | None


def _raycast_dda_pure(
    voxels: np.ndarray,
    blocks_los_lut: np.ndarray,
    start_x: float, start_y: float, start_z: float,
    end_x_f: float, end_y_f: float, end_z_f: float,
    include_end: bool,
) -> Tuple[bool, int, int, int]:
    """
    Pure-Python DDA raycast core.

    Returns: (blocked, hit_x, hit_y, hit_z)
             If not blocked, hit coords are -1.
    """
    dx = end_x_f - start_x
    dy = end_y_f - start_y
    dz = end_z_f - start_z

    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length <= 1e-9:
        return (False, -1, -1, -1)

    # Unit direction
    ux, uy, uz = dx/length, dy/length, dz/length

    # Current voxel coordinates
    x = int(math.floor(start_x))
    y = int(math.floor(start_y))
    z = int(math.floor(start_z))
    end_x = int(math.floor(end_x_f))
    end_y = int(math.floor(end_y_f))
    end_z = int(math.floor(end_z_f))

    step_x = 1 if ux > 0 else (-1 if ux < 0 else 0)
    step_y = 1 if uy > 0 else (-1 if uy < 0 else 0)
    step_z = 1 if uz > 0 else (-1 if uz < 0 else 0)

    t_delta_x = abs(1.0 / ux) if ux != 0 else 1e30
    t_delta_y = abs(1.0 / uy) if uy != 0 else 1e30
    t_delta_z = abs(1.0 / uz) if uz != 0 else 1e30

    if step_x > 0:
        t_max_x = (x + 1.0 - start_x) * t_delta_x
    elif step_x < 0:
        t_max_x = (start_x - x) * t_delta_x
    else:
        t_max_x = 1e30

    if step_y > 0:
        t_max_y = (y + 1.0 - start_y) * t_delta_y
    elif step_y < 0:
        t_max_y = (start_y - y) * t_delta_y
    else:
        t_max_y = 1e30

    if step_z > 0:
        t_max_z = (z + 1.0 - start_z) * t_delta_z
    elif step_z < 0:
        t_max_z = (start_z - z) * t_delta_z
    else:
        t_max_z = 1e30

    # Cache world dimensions
    sz, sy, sx = voxels.shape

    # Loop limit
    max_steps = int(length * 2) + sx + sy + sz + 2

    for _ in range(max_steps):
        if x == end_x and y == end_y and z == end_z:
            if not include_end:
                return (False, -1, -1, -1)
            # Check end voxel if requested
            if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
                if blocks_los_lut[voxels[z, y, x]]:
                    return (True, x, y, z)
            return (False, -1, -1, -1)

        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                x += step_x
                t_max_x += t_delta_x
            else:
                z += step_z
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                y += step_y
                t_max_y += t_delta_y
            else:
                z += step_z
                t_max_z += t_delta_z

        # Out of bounds check
        if not (0 <= x < sx and 0 <= y < sy and 0 <= z < sz):
            if x == end_x and y == end_y and z == end_z:
                return (False, -1, -1, -1)
            return (True, -1, -1, -1)

        # Ignore end voxel check if not requested
        if (not include_end) and x == end_x and y == end_y and z == end_z:
            return (False, -1, -1, -1)

        # Check if this voxel type blocks LOS
        if blocks_los_lut[voxels[z, y, x]]:
            return (True, x, y, z)

    return (True, -1, -1, -1)


@numba.njit(cache=True)
def _raycast_dda_numba(
    voxels: np.ndarray,
    blocks_los_lut: np.ndarray,
    start_x: float, start_y: float, start_z: float,
    end_x_f: float, end_y_f: float, end_z_f: float,
    include_end: bool,
) -> Tuple[bool, int, int, int]:
    """
    Numba JIT-compiled DDA raycast core.

    Returns: (blocked, hit_x, hit_y, hit_z)
             If not blocked, hit coords are -1.
    """
    dx = end_x_f - start_x
    dy = end_y_f - start_y
    dz = end_z_f - start_z

    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length <= 1e-9:
        return (False, -1, -1, -1)

    # Unit direction
    ux, uy, uz = dx/length, dy/length, dz/length

    # Current voxel coordinates
    x = int(math.floor(start_x))
    y = int(math.floor(start_y))
    z = int(math.floor(start_z))
    end_x = int(math.floor(end_x_f))
    end_y = int(math.floor(end_y_f))
    end_z = int(math.floor(end_z_f))

    step_x = 1 if ux > 0 else (-1 if ux < 0 else 0)
    step_y = 1 if uy > 0 else (-1 if uy < 0 else 0)
    step_z = 1 if uz > 0 else (-1 if uz < 0 else 0)

    t_delta_x = abs(1.0 / ux) if ux != 0 else 1e30
    t_delta_y = abs(1.0 / uy) if uy != 0 else 1e30
    t_delta_z = abs(1.0 / uz) if uz != 0 else 1e30

    if step_x > 0:
        t_max_x = (x + 1.0 - start_x) * t_delta_x
    elif step_x < 0:
        t_max_x = (start_x - x) * t_delta_x
    else:
        t_max_x = 1e30

    if step_y > 0:
        t_max_y = (y + 1.0 - start_y) * t_delta_y
    elif step_y < 0:
        t_max_y = (start_y - y) * t_delta_y
    else:
        t_max_y = 1e30

    if step_z > 0:
        t_max_z = (z + 1.0 - start_z) * t_delta_z
    elif step_z < 0:
        t_max_z = (start_z - z) * t_delta_z
    else:
        t_max_z = 1e30

    # Cache world dimensions
    sz = voxels.shape[0]
    sy = voxels.shape[1]
    sx = voxels.shape[2]

    # Loop limit
    max_steps = int(length * 2) + sx + sy + sz + 2

    for _ in range(max_steps):
        if x == end_x and y == end_y and z == end_z:
            if not include_end:
                return (False, -1, -1, -1)
            # Check end voxel if requested
            if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
                if blocks_los_lut[voxels[z, y, x]]:
                    return (True, x, y, z)
            return (False, -1, -1, -1)

        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                x += step_x
                t_max_x += t_delta_x
            else:
                z += step_z
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                y += step_y
                t_max_y += t_delta_y
            else:
                z += step_z
                t_max_z += t_delta_z

        # Out of bounds check
        if not (0 <= x < sx and 0 <= y < sy and 0 <= z < sz):
            if x == end_x and y == end_y and z == end_z:
                return (False, -1, -1, -1)
            return (True, -1, -1, -1)

        # Ignore end voxel check if not requested
        if (not include_end) and x == end_x and y == end_y and z == end_z:
            return (False, -1, -1, -1)

        # Check if this voxel type blocks LOS
        if blocks_los_lut[voxels[z, y, x]]:
            return (True, x, y, z)

    return (True, -1, -1, -1)


def raycast_voxels(
    world: VoxelWorld,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    *,
    include_end: bool = False,
) -> RaycastHit:
    """
    Fast voxel traversal (3D DDA) from start to end.
    Uses Numba JIT-compiled core for performance.
    """
    blocks_los_lut = world.blocks_los_lut()

    blocked, hx, hy, hz = _raycast_dda_numba(
        world.voxels,
        blocks_los_lut,
        float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2]),
        float(end_xyz[0]), float(end_xyz[1]), float(end_xyz[2]),
        include_end,
    )

    blocked_voxel = (hx, hy, hz) if blocked and hx >= 0 else None
    return RaycastHit(blocked=blocked, blocked_voxel=blocked_voxel)


def has_los(world: VoxelWorld, start_xyz: np.ndarray, end_xyz: np.ndarray) -> bool:
    return not raycast_voxels(world, start_xyz, end_xyz).blocked
