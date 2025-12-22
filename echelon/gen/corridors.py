from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..sim.world import VoxelWorld
from .objective import capture_zone_params


def _corner_anchor_xy(corner: str, spawn_clear: int, *, size_x: int, size_y: int) -> tuple[float, float]:
    half = float(max(1, int(spawn_clear // 2)))
    sx = float(size_x)
    sy = float(size_y)
    if corner == "BL":
        return half, half
    if corner == "BR":
        return sx - 1.0 - half, half
    if corner == "TL":
        return half, sy - 1.0 - half
    if corner == "TR":
        return sx - 1.0 - half, sy - 1.0 - half
    raise ValueError(f"Unknown corner: {corner!r}")


def _carve_line(world: VoxelWorld, x0: float, y0: float, x1: float, y1: float, *, width_vox: float) -> None:
    dist = float(math.hypot(x1 - x0, y1 - y0))
    steps = max(1, int(dist * 1.5))
    half_w = float(width_vox) * 0.5
    # Standing room: assume default 4 voxels if not known
    clearance_z = int(world.meta.get("validator", {}).get("clearance_z", 4))
    for i in range(steps + 1):
        t = float(i) / float(steps)
        px = x0 + (x1 - x0) * t
        py = y0 + (y1 - y0) * t
        min_x = int(math.floor(px - half_w))
        max_x = int(math.floor(px + half_w)) + 1
        min_y = int(math.floor(py - half_w))
        max_y = int(math.floor(py + half_w)) + 1
        # Clear only standing room
        world.voxels[0 : clearance_z + 1, min_y:max_y, min_x:max_x] = 0 # AIR


def carve_macro_corridors(
    world: VoxelWorld,
    *,
    spawn_corners: dict[str, str],
    spawn_clear: int,
    meta: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """
    Carve a simple "fightability skeleton" (roads/corridors) into an existing boolean voxel map.

    This is intentionally cheap and deterministic (seed-driven), and is designed to complement
    the ConnectivityValidator rather than replace it.
    """
    sx = int(world.size_x)
    sy = int(world.size_y)

    cx, cy, zone_r = capture_zone_params(meta, size_x=sx, size_y=sy)

    min_dim = float(min(sx, sy))
    width = int(max(4, min(8, round(min_dim * 0.06))))
    width = int(min(width, max(2, min(sx, sy) // 4)))
    width_f = float(width)

    # Objective ring (a readable "laning" structure near the hill).
    ring_r = int(round(float(zone_r) + width_f * 1.25))
    ring_r = int(np.clip(ring_r, 3, max(3, int(min_dim * 0.35))))

    x0 = float(np.clip(cx - ring_r, 1.0, float(sx - 2)))
    x1 = float(np.clip(cx + ring_r, 1.0, float(sx - 2)))
    y0 = float(np.clip(cy - ring_r, 1.0, float(sy - 2)))
    y1 = float(np.clip(cy + ring_r, 1.0, float(sy - 2)))

    _carve_line(world, x0, y0, x1, y0, width_vox=width_f)
    _carve_line(world, x0, y1, x1, y1, width_vox=width_f)
    _carve_line(world, x0, y0, x0, y1, width_vox=width_f)
    _carve_line(world, x1, y0, x1, y1, width_vox=width_f)

    corridors: list[dict[str, Any]] = meta.setdefault("corridors", [])
    corridors.append({"kind": "objective_ring", "width": float(width_f), "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]})

    # Spawn-to-objective lanes (two per team: direct + offset flank).
    for team in ("blue", "red"):
        corner = str(spawn_corners.get(team, "BL"))
        ax, ay = _corner_anchor_xy(corner, spawn_clear, size_x=sx, size_y=sy)
        _carve_line(world, ax, ay, float(cx), float(cy), width_vox=width_f)
        corridors.append({"kind": f"{team}_lane_main", "width": float(width_f), "points": [[ax, ay], [float(cx), float(cy)]]})

        vx = float(cx - ax)
        vy = float(cy - ay)
        norm = float(math.hypot(vx, vy))
        if norm > 1e-6:
            px = -vy / norm
            py = vx / norm
        else:
            px, py = 1.0, 0.0

        # Deterministic but varied offset; opposite sign for teams by default.
        sign = -1.0 if team == "blue" else 1.0
        sign *= -1.0 if bool(rng.integers(0, 2) == 0) else 1.0
        offset = float(rng.uniform(1.5, 2.5)) * width_f * sign

        bx = float(np.clip(ax + px * offset, 1.0, float(sx - 2)))
        by = float(np.clip(ay + py * offset, 1.0, float(sy - 2)))
        tx = float(np.clip(cx + px * offset, 1.0, float(sx - 2)))
        ty = float(np.clip(cy + py * offset, 1.0, float(sy - 2)))
        _carve_line(world, bx, by, tx, ty, width_vox=width_f)
        corridors.append({"kind": f"{team}_lane_flank", "width": float(width_f), "points": [[bx, by], [tx, ty]]})

