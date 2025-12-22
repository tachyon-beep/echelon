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

    Notes:
    - The start voxel is not treated as a blocker; only intervening voxels.
    - The end voxel is ignored by default, but can be included via include_end=True.
    - Out-of-bounds is treated as blocked.
    """
    start = np.asarray(start_xyz, dtype=np.float64)
    end = np.asarray(end_xyz, dtype=np.float64)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return RaycastHit(blocked=False, blocked_voxel=None)

    direction /= length

    x = int(math.floor(start[0]))
    y = int(math.floor(start[1]))
    z = int(math.floor(start[2]))
    end_x = int(math.floor(end[0]))
    end_y = int(math.floor(end[1]))
    end_z = int(math.floor(end[2]))

    step_x = 1 if direction[0] > 0 else (-1 if direction[0] < 0 else 0)
    step_y = 1 if direction[1] > 0 else (-1 if direction[1] < 0 else 0)
    step_z = 1 if direction[2] > 0 else (-1 if direction[2] < 0 else 0)

    inv_x = 1.0 / direction[0] if step_x != 0 else math.inf
    inv_y = 1.0 / direction[1] if step_y != 0 else math.inf
    inv_z = 1.0 / direction[2] if step_z != 0 else math.inf
    t_delta_x = abs(inv_x)
    t_delta_y = abs(inv_y)
    t_delta_z = abs(inv_z)

    if step_x > 0:
        next_boundary_x = (x + 1.0)
        t_max_x = (next_boundary_x - start[0]) * abs(inv_x)
    elif step_x < 0:
        next_boundary_x = (x * 1.0)
        t_max_x = (start[0] - next_boundary_x) * abs(inv_x)
    else:
        t_max_x = math.inf

    if step_y > 0:
        next_boundary_y = (y + 1.0)
        t_max_y = (next_boundary_y - start[1]) * abs(inv_y)
    elif step_y < 0:
        next_boundary_y = (y * 1.0)
        t_max_y = (start[1] - next_boundary_y) * abs(inv_y)
    else:
        t_max_y = math.inf

    if step_z > 0:
        next_boundary_z = (z + 1.0)
        t_max_z = (next_boundary_z - start[2]) * abs(inv_z)
    elif step_z < 0:
        next_boundary_z = (z * 1.0)
        t_max_z = (start[2] - next_boundary_z) * abs(inv_z)
    else:
        t_max_z = math.inf

    start_voxel = (x, y, z)
    end_voxel = (end_x, end_y, end_z)

    # Bound the number of steps: segment length in voxels can't exceed this much.
    max_steps = int(length * 4) + world.size_x + world.size_y + world.size_z + 8
    for _ in range(max_steps):
        if (x, y, z) == end_voxel:
            return RaycastHit(blocked=False, blocked_voxel=None)

        # Step to next voxel boundary.
        if t_max_x <= t_max_y and t_max_x <= t_max_z:
            x += step_x
            t_max_x += t_delta_x
        elif t_max_y <= t_max_z:
            y += step_y
            t_max_y += t_delta_y
        else:
            z += step_z
            t_max_z += t_delta_z

        current = (x, y, z)
        if current == start_voxel:
            continue
        if (not include_end) and current == end_voxel:
            continue
        
        # Check if this voxel type blocks LOS
        v_id = world.get_voxel(x, y, z)
        if world.MATERIAL_PROPS.get(v_id, {}).get("blocks_los", False):
            return RaycastHit(blocked=True, blocked_voxel=current)

    # If we somehow failed to reach the end, treat as blocked for safety.
    return RaycastHit(blocked=True, blocked_voxel=None)


def has_los(world: VoxelWorld, start_xyz: np.ndarray, end_xyz: np.ndarray) -> bool:
    return not raycast_voxels(world, start_xyz, end_xyz).blocked
