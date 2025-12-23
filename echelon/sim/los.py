from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .world import VoxelWorld


@dataclass(frozen=True)
class RaycastHit:
    blocked: bool
    blocked_voxel: tuple[int, int, int] | None


def raycast_voxels(
    world: VoxelWorld,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    *,
    include_end: bool = False,
) -> RaycastHit:
    """
    Fast voxel traversal (3D DDA) from start to end.
    Optimized for speed in pure Python.
    """
    # Use float64 for DDA to avoid precision drift
    start_x, start_y, start_z = float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2])
    end_x_f, end_y_f, end_z_f = float(end_xyz[0]), float(end_xyz[1]), float(end_xyz[2])
    
    dx = end_x_f - start_x
    dy = end_y_f - start_y
    dz = end_z_f - start_z
    
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length <= 1e-9:
        return RaycastHit(blocked=False, blocked_voxel=None)

    # Unit direction
    ux, uy, uz = dx/length, dy/length, dz/length

    # Current voxel coordinates
    x, y, z = int(math.floor(start_x)), int(math.floor(start_y)), int(math.floor(start_z))
    end_x, end_y, end_z = int(math.floor(end_x_f)), int(math.floor(end_y_f)), int(math.floor(end_z_f))

    step_x = 1 if ux > 0 else (-1 if ux < 0 else 0)
    step_y = 1 if uy > 0 else (-1 if uy < 0 else 0)
    step_z = 1 if uz > 0 else (-1 if uz < 0 else 0)

    t_delta_x = abs(1.0 / ux) if ux != 0 else 1e30
    t_delta_y = abs(1.0 / uy) if uy != 0 else 1e30
    t_delta_z = abs(1.0 / uz) if uz != 0 else 1e30

    if step_x > 0: t_max_x = (x + 1.0 - start_x) * t_delta_x
    elif step_x < 0: t_max_x = (start_x - x) * t_delta_x
    else: t_max_x = 1e30

    if step_y > 0: t_max_y = (y + 1.0 - start_y) * t_delta_y
    elif step_y < 0: t_max_y = (start_y - y) * t_delta_y
    else: t_max_y = 1e30

    if step_z > 0: t_max_z = (z + 1.0 - start_z) * t_delta_z
    elif step_z < 0: t_max_z = (start_z - z) * t_delta_z
    else: t_max_z = 1e30

    # Cache world properties
    voxels = world.voxels
    sz, sy, sx = voxels.shape
    # Use a pre-calculated blocks_los LUT for faster access
    blocks_los_lut = world.blocks_los_lut()

    # Pre-calculate termination voxel
    # Loop limit to avoid infinite loops
    max_steps = int(length * 2) + sx + sy + sz + 2
    
    for _ in range(max_steps):
        if x == end_x and y == end_y and z == end_z:
            if not include_end:
                return RaycastHit(blocked=False, blocked_voxel=None)
            # Check end voxel if requested
            if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
                if blocks_los_lut[voxels[z, y, x]]:
                    return RaycastHit(blocked=True, blocked_voxel=(x, y, z))
            return RaycastHit(blocked=False, blocked_voxel=None)

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
            # If we reached the end voxel but it's out of bounds, we're done
            if x == end_x and y == end_y and z == end_z:
                return RaycastHit(blocked=False, blocked_voxel=None)
            return RaycastHit(blocked=True, blocked_voxel=None)

        # Ignore end voxel check if not requested
        if (not include_end) and x == end_x and y == end_y and z == end_z:
            return RaycastHit(blocked=False, blocked_voxel=None)

        # Check if this voxel type blocks LOS
        if blocks_los_lut[voxels[z, y, x]]:
            return RaycastHit(blocked=True, blocked_voxel=(x, y, z))

    return RaycastHit(blocked=True, blocked_voxel=None)


def has_los(world: VoxelWorld, start_xyz: np.ndarray, end_xyz: np.ndarray) -> bool:
    return not raycast_voxels(world, start_xyz, end_xyz).blocked
