from __future__ import annotations

from typing import Any

import numpy as np

from ..sim.world import VoxelWorld

DEFAULT_CAPTURE_ZONE_RADIUS = 15.0


def capture_zone_params(meta: dict[str, Any], *, size_x: int, size_y: int) -> tuple[float, float, float]:
    cz = meta.get("capture_zone")
    if not isinstance(cz, dict):
        raise ValueError("world meta has no valid `capture_zone` dict")
    center = cz.get("center")
    radius = cz.get("radius")
    if (
        not isinstance(center, (list, tuple))
        or len(center) < 2
        or not isinstance(center[0], (int, float))
        or not isinstance(center[1], (int, float))
        or not isinstance(radius, (int, float))
    ):
        raise ValueError("world meta has no valid `capture_zone.center` or `capture_zone.radius`")
    cx = float(center[0])
    cy = float(center[1])
    r = float(radius)
    if not (0.0 <= cx <= float(size_x) and 0.0 <= cy <= float(size_y) and r > 0.0):
        raise ValueError("world meta has out-of-bounds capture zone")
    return cx, cy, r


def capture_zone_anchor(meta: dict[str, Any], *, size_x: int, size_y: int) -> tuple[int, int]:
    cx, cy, _ = capture_zone_params(meta, size_x=size_x, size_y=size_y)
    x = int(round(cx))
    y = int(round(cy))
    x = int(np.clip(x, 0, max(0, size_x - 1)))
    y = int(np.clip(y, 0, max(0, size_y - 1)))
    return (y, x)


def _circle_intersects_aabb(
    cx: float, cy: float, r: float, *, x0: float, x1: float, y0: float, y1: float
) -> bool:
    closest_x = float(np.clip(cx, x0, x1))
    closest_y = float(np.clip(cy, y0, y1))
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy <= r * r


def _spawn_clear_aabb(corner: str, spawn_clear: int, *, size_x: int, size_y: int) -> tuple[float, float, float, float]:
    sc = float(max(0, spawn_clear))
    sx = float(size_x)
    sy = float(size_y)
    if corner == "BL":
        return 0.0, sc, 0.0, sc
    if corner == "BR":
        return sx - sc, sx, 0.0, sc
    if corner == "TL":
        return 0.0, sc, sy - sc, sy
    if corner == "TR":
        return sx - sc, sx, sy - sc, sy
    raise ValueError(f"Unknown corner: {corner!r}")


def _center_clear(world: VoxelWorld, cx: float, cy: float) -> bool:
    # Require a small patch of ground around the center to be free of walls.
    ix = int(round(cx))
    iy = int(round(cy))
    if ix < 1 or ix >= world.size_x - 1:
        return False
    if iy < 1 or iy >= world.size_y - 1:
        return False
    if world.size_z <= 0:
        return True
    return not bool(np.any(world.solid[0, iy - 1 : iy + 2, ix - 1 : ix + 2]))


def sample_capture_zone(
    world: VoxelWorld, rng: np.random.Generator, *, spawn_clear: int, spawn_corners: dict[str, str]
) -> dict[str, Any]:
    # Old fixed radius was 15.0; new radius is ~half of that, with +/- 50% variation.
    base_radius = DEFAULT_CAPTURE_ZONE_RADIUS * 0.5
    radius_scale = float(rng.uniform(0.5, 1.5))
    r = float(base_radius * radius_scale)

    # Keep the zone fully inside the map.
    margin = 2.0
    max_r = float(min(world.size_x, world.size_y)) * 0.5 - margin
    if max_r < 1.0:
        max_r = 1.0
    r = float(min(r, max_r))

    # Choose a quadrant and sample a random point within it.
    mid_x = world.size_x / 2.0
    mid_y = world.size_y / 2.0
    left = bool(rng.integers(0, 2) == 0)
    bottom = bool(rng.integers(0, 2) == 0)

    x_min_global = r + margin
    x_max_global = float(world.size_x) - r - margin
    y_min_global = r + margin
    y_max_global = float(world.size_y) - r - margin

    def _range_for_quadrant(min_global: float, max_global: float, mid: float, choose_low: bool) -> tuple[float, float]:
        if choose_low:
            lo, hi = min_global, min(max_global, mid)
        else:
            lo, hi = max(min_global, mid), max_global
        if lo > hi:
            lo, hi = min_global, max_global
        return lo, hi

    x_lo, x_hi = _range_for_quadrant(x_min_global, x_max_global, mid_x, left)
    y_lo, y_hi = _range_for_quadrant(y_min_global, y_max_global, mid_y, bottom)

    def _in_spawn_clear(cx: float, cy: float, r: float) -> bool:
        for corner in (spawn_corners.get("blue"), spawn_corners.get("red")):
            if not corner:
                continue
            x0, x1, y0, y1 = _spawn_clear_aabb(str(corner), spawn_clear, size_x=world.size_x, size_y=world.size_y)
            if _circle_intersects_aabb(cx, cy, r, x0=x0, x1=x1, y0=y0, y1=y1):
                return True
        return False

    cx = mid_x
    cy = mid_y
    for attempt in range(256):
        if x_hi > x_lo:
            cx = float(rng.uniform(x_lo, x_hi))
        else:
            cx = float(x_lo)
        if y_hi > y_lo:
            cy = float(rng.uniform(y_lo, y_hi))
        else:
            cy = float(y_lo)

        if not _center_clear(world, cx, cy):
            continue

        # Avoid placing the objective inside spawn clear regions when possible, but allow fallback
        # (e.g., small maps where spawn_clear covers most of the arena).
        if spawn_clear > 0 and attempt < 200 and _in_spawn_clear(cx, cy, r):
            continue

        break

    corner = ("B" if bottom else "T") + ("L" if left else "R")
    return {"center": [cx, cy], "radius": r, "corner": corner}


def clear_capture_zone(world: VoxelWorld, *, meta: dict[str, Any]) -> int:
    cx, cy, r = capture_zone_params(meta, size_x=world.size_x, size_y=world.size_y)
    # Clear any voxel whose XY cell intersects the capture circle, approximated by expanding
    # the radius by half a cell diagonal.
    r_eff = float(r) + float(np.sqrt(2.0) * 0.5)
    r2 = r_eff * r_eff

    xs = (np.arange(world.size_x, dtype=np.float32) + 0.5)[None, :]
    ys = (np.arange(world.size_y, dtype=np.float32) + 0.5)[:, None]
    mask2d = (xs - float(cx)) ** 2 + (ys - float(cy)) ** 2 <= r2

    before = int(np.count_nonzero(world.solid[:, mask2d]))
    world.solid[:, mask2d] = False
    after = int(np.count_nonzero(world.solid[:, mask2d]))
    return before - after
