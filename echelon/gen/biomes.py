from __future__ import annotations

import math
from typing import Protocol, Dict, List, Tuple, Optional

import numpy as np

from ..sim.world import VoxelWorld


class BiomeBrush(Protocol):
    def __call__(
        self,
        world: VoxelWorld,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
        rng: np.random.Generator,
    ) -> None: ...


def _safe_integers(rng: np.random.Generator, low: int, high: int) -> int:
    """Helper to avoid ValueError if low >= high on small quadrants."""
    if low >= high:
        return int(low)
    return int(rng.integers(low, high))


def _random_box(
    rng: np.random.Generator, min_x: int, min_y: int, max_x: int, max_y: int, w_range: tuple[int, int], h_range: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    if max_x - min_x < w_range[0] or max_y - min_y < h_range[0]:
        return None
    w = _safe_integers(rng, w_range[0], min(w_range[1], max_x - min_x) + 1)
    h = _safe_integers(rng, h_range[0], min(h_range[1], max_y - min_y) + 1)
    x = _safe_integers(rng, min_x, max_x - w + 1)
    y = _safe_integers(rng, min_y, max_y - h + 1)
    return x, y, w, h


def fill_urban_residential(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Dense courtyard blocks with an alley grid.
    Signature: Hollow rectangles (courtyards).
    """
    area = (max_x - min_x) * (max_y - min_y)
    density_target = 0.35  # High density
    filled = 0
    attempts = 0
    
    while filled / max(1, area) < density_target and attempts < 100:
        attempts += 1
        res = _random_box(rng, min_x, min_y, max_x, max_y, (6, 10), (6, 10))
        if not res: continue
        x, y, w, h = res

        # Align to grid partially
        x = x - (x % 2)
        y = y - (y % 2)

        height = _safe_integers(rng, 4, 9)
        world.set_box_solid(x, y, 0, x+w, y+h, height, True)
        
        if w > 4 and h > 4:
            world.set_box_solid(x+2, y+2, 0, x+w-2, y+h-2, height, False)
            world.set_box_solid(x+2, y+2, 0, x+w-2, y+h-2, 1, VoxelWorld.GLASS)
            
        filled += w * h * 0.75 # Approx


def fill_industrial_refinery(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Tanks, pipe corridors, thin walls.
    Signature: Cylinders (approx) and scatter.
    """
    area = (max_x - min_x) * (max_y - min_y)
    density_target = 0.25
    filled = 0
    attempts = 0
    
    while filled / max(1, area) < density_target and attempts < 200:
        attempts += 1
        res = _random_box(rng, min_x, min_y, max_x, max_y, (3, 6), (3, 6))
        if not res: continue
        x, y, w, h = res
        
        height = _safe_integers(rng, 4, 12)
        world.set_box_solid(x, y, 0, x+w, y+h, height, True)
        filled += w*h

        if rng.random() < 0.3:
            res_p = _random_box(rng, min_x, min_y, max_x, max_y, (1, 2), (8, 15))
            if res_p:
                px, py, pw, ph = res_p
                p_height = _safe_integers(rng, 2, 5)
                world.set_box_solid(px, py, 0, px+pw, py+ph, p_height, True)


def fill_civic_government(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Large monolithic buildings, plazas, low walls.
    """
    area = (max_x - min_x) * (max_y - min_y)
    density_target = 0.20
    filled = 0
    attempts = 0
    
    while filled / max(1, area) < density_target and attempts < 50:
        attempts += 1
        res = _random_box(rng, min_x, min_y, max_x, max_y, (8, 15), (8, 15))
        if not res: continue
        x, y, w, h = res
        
        height = _safe_integers(rng, 6, 10)
        world.set_box_solid(x, y, 0, x+w, y+h, height, True)
        filled += w*h
        
        # Plaza walls
        margin = 2
        wx, wy = x - margin, y - margin
        ww, wh = w + margin*2, h + margin*2
        wx = max(min_x, wx); wy = max(min_y, wy)
        ww = min(ww, max_x - wx); wh = min(wh, max_y - wy)
        
        if ww > 0 and wh > 0:
            world.set_box_solid(wx, wy, 0, wx+ww, wy+wh, 1, True)
            world.set_box_solid(wx+1, wy+1, 0, wx+ww-1, wy+wh-1, 1, False)
            world.set_box_solid(x, y, 0, x+w, y+h, height, True)


def fill_forest_park(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Scattered vertical columns (trees) and rock outcrops.
    """
    area = (max_x - min_x) * (max_y - min_y)
    num_trees = int(area * 0.05)
    
    for _ in range(num_trees):
        tx = _safe_integers(rng, min_x, max_x)
        ty = _safe_integers(rng, min_y, max_y)
        ts = _safe_integers(rng, 1, 3) 
        th = _safe_integers(rng, 3, 8)
        world.set_box_solid(tx, ty, 0, tx+ts, ty+ts, th, True)
        if th >= 4:
            world.set_box_solid(tx - 1, ty - 1, 3, tx + ts + 1, ty + ts + 1, 5, VoxelWorld.FOLIAGE)
        
    num_rocks = int(area * 0.01)
    for _ in range(num_rocks):
        res = _random_box(rng, min_x, min_y, max_x, max_y, (3, 5), (3, 5))
        if res:
            rx, ry, rw, rh = res
            rheight = _safe_integers(rng, 2, 4)
            world.set_box_solid(rx, ry, 0, rx+rw, ry+rh, rheight, True)


def fill_barrens(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Sparse, open, minimal cover.
    """
    area = (max_x - min_x) * (max_y - min_y)
    num_rocks = int(area * 0.02)
    for _ in range(num_rocks):
        res = _random_box(rng, min_x, min_y, max_x, max_y, (2, 4), (2, 4))
        if res:
            rx, ry, rw, rh = res
            rheight = _safe_integers(rng, 1, 3)
            world.set_box_solid(rx, ry, 0, rx+rw, ry+rh, rheight, True)


def fill_arcology_spire(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Brutalist verticality. Platforms at Z=5, Z=10, Z=15.
    """
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
    cw, ch = (max_x - min_x) // 3, (max_y - min_y) // 3
    world.set_box_solid(cx - cw//2, cy - ch//2, 0, cx + cw//2, cy + ch//2, world.size_z - 2, True)

    platforms = []
    for _ in range(_safe_integers(rng, 4, 8)):
        z_level = int(rng.choice([5, 10, 15]))
        if z_level >= world.size_z - 2: continue
        
        res = _random_box(rng, min_x, min_y, max_x, max_y, (8, 12), (8, 12))
        if res:
            px, py, pw, ph = res
            world.set_box_solid(px, py, z_level, px+pw, py+ph, z_level + 1, True)
            clearance = int(world.meta.get("validator", {}).get("clearance_z", 4))
            world.set_box_solid(px, py, z_level + 1, px+pw, py+ph, z_level + 1 + clearance, False)
            platforms.append((px + pw//2, py + ph//2, z_level))

    if platforms:
        p0 = platforms[0]
        _build_stairs(world, min_x + 2, min_y + 2, -1, p0[0], p0[1], p0[2], width=3)
        for i in range(len(platforms) - 1):
            pA = platforms[i]
            pB = platforms[i+1]
            _build_stairs(world, pA[0], pA[1], pA[2], pB[0], pB[1], pB[2], width=3)

    for _ in range(_safe_integers(rng, 2, 4)):
        start_x, start_y = min_x, _safe_integers(rng, min_y, max_y)
        end_x, end_y = max_x, _safe_integers(rng, min_y, max_y)
        z = _safe_integers(rng, 2, world.size_z - 5)
        _carve_worm(world, start_x, start_y, z, end_x, end_y, z, radius=2)


def fill_sunken_canal(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Industrial logistics. Low-level water canal with high-level bridges.
    """
    is_horiz = rng.random() > 0.5
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
    cw = _safe_integers(rng, 8, 14)
    
    if is_horiz:
        world.set_box_solid(min_x, cy - cw//2, 0, max_x, cy + cw//2, 1, VoxelWorld.WATER)
        world.set_box_solid(min_x, cy - cw//2, 1, max_x, cy + cw//2, world.size_z, False)
        for _ in range(_safe_integers(rng, 10, 20)):
            bx = _safe_integers(rng, min_x, max_x - 4)
            by = _safe_integers(rng, min_y, max_y - 4)
            if abs(by - cy) < cw//2 + 2: continue 
            bh = _safe_integers(rng, 4, 10)
            world.set_box_solid(bx, by, 0, bx + 4, by + 4, bh, True)
        for _ in range(_safe_integers(rng, 2, 4)):
            bx = _safe_integers(rng, min_x + 5, max_x - 5)
            z = _safe_integers(rng, 4, 8)
            world.set_box_solid(bx, cy - cw//2 - 2, z, bx + 4, cy + cw//2 + 2, z + 1, True)
    else:
        world.set_box_solid(cx - cw//2, min_y, 0, cx + cw//2, max_y, 1, VoxelWorld.WATER)
        world.set_box_solid(cx - cw//2, min_y, 1, cx + cw//2, max_y, world.size_z, False)
        for _ in range(_safe_integers(rng, 10, 20)):
            by = _safe_integers(rng, min_y, max_y - 4)
            bx = _safe_integers(rng, min_x, max_x - 4)
            if abs(bx - cx) < cw//2 + 2: continue
            bh = _safe_integers(rng, 4, 10)
            world.set_box_solid(bx, by, 0, bx + 4, by + 4, bh, True)
        for _ in range(_safe_integers(rng, 2, 4)):
            by = _safe_integers(rng, min_y + 5, max_y - 5)
            z = _safe_integers(rng, 4, 8)
            world.set_box_solid(cx - cw//2 - 2, by, z, cx + cw//2 + 2, by + 4, z + 1, True)


def fill_geo_front(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Subterranean maze. Enclosed tunnels and lava pools.
    """
    ceiling_z = _safe_integers(rng, 10, world.size_z - 2)
    world.set_box_solid(min_x, min_y, 0, max_x, max_y, ceiling_z, True)
    mid_y = (min_y + max_y) // 2
    mid_x = (min_x + max_x) // 2
    _carve_worm(world, min_x, mid_y, 2, max_x, mid_y, 2, radius=3)
    _carve_worm(world, mid_x, min_y, 2, mid_x, max_y, 2, radius=3)
    for _ in range(_safe_integers(rng, 5, 10)):
        x0, y0 = _safe_integers(rng, min_x, max_x), _safe_integers(rng, min_y, max_y)
        x1, y1 = _safe_integers(rng, min_x, max_x), _safe_integers(rng, min_y, max_y)
        z0 = _safe_integers(rng, 1, ceiling_z - 3)
        z1 = _safe_integers(rng, 1, ceiling_z - 3)
        _carve_worm(world, x0, y0, z0, x1, y1, z1, radius=2)
    for _ in range(_safe_integers(rng, 3, 6)):
        lx, ly = _safe_integers(rng, min_x, max_x - 4), _safe_integers(rng, min_y, max_y - 4)
        world.set_box_solid(lx, ly, 0, lx + 4, ly + 4, 1, VoxelWorld.LAVA)


def fill_glass_desert(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Crystalline structures made of translucent GLASS.
    Signature: Tall crystalline pillars and low glass walls.
    """
    area = (max_x - min_x) * (max_y - min_y)
    num_crystals = _safe_integers(rng, 10, 20)
    for _ in range(num_crystals):
        res = _random_box(rng, min_x, min_y, max_x, max_y, (2, 5), (2, 5))
        if res:
            x, y, w, h = res
            z = _safe_integers(rng, 5, 12)
            world.set_box_solid(x, y, 0, x+w, y+h, z, VoxelWorld.GLASS)
            
    num_walls = _safe_integers(rng, 5, 10)
    for _ in range(num_walls):
        res = _random_box(rng, min_x, min_y, max_x, max_y, (1, 8), (1, 8))
        if res:
            x, y, w, h = res
            world.set_box_solid(x, y, 0, x+w, y+h, 1, VoxelWorld.GLASS)


def fill_command_outpost(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Heavily fortified bunkers made of REINFORCED voxels.
    """
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
    # Central Bunker
    bw, bh, bz = 12, 12, 6
    world.set_box_solid(cx - bw//2, cy - bh//2, 0, cx + bw//2, cy + bh//2, bz, VoxelWorld.REINFORCED)
    # Interior rooms (SOLID, destructible)
    world.set_box_solid(cx - bw//2 + 2, cy - bh//2 + 2, 0, cx + bw//2 - 2, cy + bh//2 - 2, bz - 1, VoxelWorld.SOLID)
    # Entrance (AIR)
    world.set_box_solid(cx - bw//2, cy - 2, 0, cx - bw//2 + 3, cy + 2, 3, VoxelWorld.AIR)
    
    # Perimeter fences (REINFORCED low walls)
    margin = 8
    world.set_box_solid(cx - bw//2 - margin, cy - bh//2 - margin, 0, cx + bw//2 + margin, cy + bh//2 + margin, 1, VoxelWorld.REINFORCED)
    world.set_box_solid(cx - bw//2 - margin + 1, cy - bh//2 - margin + 1, 0, cx + bw//2 + margin - 1, cy + bh//2 + margin - 1, 1, VoxelWorld.AIR)


def fill_dense_jungle(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Complete LOS blockage via heavy FOLIAGE canopies.
    """
    area = (max_x - min_x) * (max_y - min_y)
    # High density of thin trees
    num_trees = int(area * 0.15)
    for _ in range(num_trees):
        tx = _safe_integers(rng, min_x, max_x)
        ty = _safe_integers(rng, min_y, max_y)
        th = _safe_integers(rng, 4, 8)
        world.set_box_solid(tx, ty, 0, tx + 1, ty + 1, th, VoxelWorld.SOLID)
        # Large broad canopies
        world.set_box_solid(tx - 2, ty - 2, 3, tx + 3, ty + 3, 6, VoxelWorld.FOLIAGE)


def fill_mining_pit(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Circular terraced excavation using the Staircase logic.
    """
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
    max_r = min(max_x - min_x, max_y - min_y) // 2 - 2
    
    # Fill whole quadrant with solid first
    world.set_box_solid(min_x, min_y, 0, max_x, max_y, 10, VoxelWorld.SOLID)
    
    # Carve terraces
    for r in range(max_r, 4, -4):
        z = 10 - (max_r - r) // 2
        # Carve ring
        for angle in range(0, 360, 5):
            rad = math.radians(angle)
            px = int(cx + r * math.cos(rad))
            py = int(cy + r * math.sin(rad))
            world.set_box_solid(px - 2, py - 2, z, px + 3, py + 3, 10, VoxelWorld.AIR)
            
    # Central pit floor
    world.set_box_solid(cx - 4, cy - 4, 0, cx + 5, cy + 5, 10, VoxelWorld.AIR)
    
    # Switchback ramp
    _build_stairs(world, cx + max_r, cy, 10, cx, cy, 0, width=4)


def _build_stairs(world: VoxelWorld, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int, width: int) -> None:
    """Builds a solid ramp of voxels with AIR headroom."""
    dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    if dist < 1e-6: return
    steps = max(1, int(dist * 1.5))
    clearance = int(world.meta.get("validator", {}).get("clearance_z", 4))
    half_w = width / 2.0
    for i in range(steps + 1):
        t = i / steps
        px, py = x0 + (x1-x0)*t, y0 + (y1-y0)*t
        pz = z0 + (z1-z0)*t
        ix, iy, iz = int(px), int(py), int(pz)
        if iz >= 0:
            world.set_box_solid(int(px - half_w), int(py - half_w), iz, 
                               int(px + half_w + 1), int(py + half_w + 1), iz + 1, True)
        world.set_box_solid(int(px - half_w), int(py - half_w), iz + 1,
                           int(px + half_w + 1), int(py + half_w + 1), iz + 1 + clearance, False)


def _carve_worm(world: VoxelWorld, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int, radius: int) -> None:
    """Carves a 3D tube of AIR."""
    dist = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
    steps = max(1, int(dist * 2))
    for i in range(steps + 1):
        t = i / steps
        px, py, pz = x0 + (x1-x0)*t, y0 + (y1-y0)*t, z0 + (z1-z0)*t
        ix, iy, iz = int(px), int(py), int(pz)
        world.set_box_solid(ix - radius, iy - radius, iz - radius, ix + radius + 1, iy + radius + 1, iz + radius + 1, False)


def fill_volcanic_ridge(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Elevated ridge with lava rivers.
    """
    is_horiz = rng.random() > 0.5
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
    
    # 1. The Ridge (Raised Solid spine)
    if is_horiz:
        world.set_box_solid(min_x, cy - 4, 0, max_x, cy + 4, 4, VoxelWorld.SOLID)
        # 2. Lava rivers flowing off the ridge
        for _ in range(_safe_integers(rng, 2, 4)):
            lx = _safe_integers(rng, min_x, max_x - 3)
            world.set_box_solid(lx, cy - 6, 0, lx + 3, cy + 6, 1, VoxelWorld.LAVA)
    else:
        world.set_box_solid(cx - 4, min_y, 0, cx + 4, max_y, 4, VoxelWorld.SOLID)
        for _ in range(_safe_integers(rng, 2, 4)):
            ly = _safe_integers(rng, min_y, max_y - 3)
            world.set_box_solid(cx - 6, ly, 0, cx + 6, ly + 3, 1, VoxelWorld.LAVA)


def fill_hydroponic_lab(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Glass-enclosed greenhouses with foliage.
    """
    for _ in range(_safe_integers(rng, 2, 4)):
        res = _random_box(rng, min_x, min_y, max_x, max_y, (10, 15), (10, 15))
        if res:
            hx, hy, hw, hh = res
            hz = _safe_integers(rng, 5, 8)
            # Walls (REINFORCED corners, GLASS panels)
            world.set_box_solid(hx, hy, 0, hx+hw, hy+hh, hz, VoxelWorld.GLASS)
            world.set_box_solid(hx, hy, 0, hx+1, hy+1, hz, VoxelWorld.REINFORCED)
            world.set_box_solid(hx+hw-1, hy, 0, hx+hw, hy+1, hz, VoxelWorld.REINFORCED)
            # Foliage rows inside
            for row in range(hy + 2, hy + hh - 2, 3):
                world.set_box_solid(hx + 2, row, 0, hx + hw - 2, row + 1, 2, VoxelWorld.FOLIAGE)


def fill_server_farm(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Dense grid of reinforced racks and glass data lines.
    """
    step = 5
    for y in range(min_y + 1, max_y - step, step):
        for x in range(min_x + 1, max_x - step, step):
            if rng.random() > 0.3:
                world.set_box_solid(x, y, 0, x+2, y+2, _safe_integers(rng, 6, 12), VoxelWorld.REINFORCED)
                # Glass conduit overhead
                if rng.random() > 0.5:
                    world.set_box_solid(x, y, 4, x+step, y+1, 5, VoxelWorld.GLASS)


def fill_orbital_relay(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Massive reinforced dishes and high-altitude planks.
    """
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
    # Dish base
    r = _safe_integers(rng, 8, 12)
    for angle in range(0, 360, 10):
        rad = math.radians(angle)
        px, py = int(cx + r * math.cos(rad)), int(cy + r * math.sin(rad))
        world.set_box_solid(px-1, py-1, 0, px+2, py+2, _safe_integers(rng, 8, 15), VoxelWorld.REINFORCED)
    
    # High-altitude bridges
    for _ in range(_safe_integers(rng, 2, 4)):
        bx = _safe_integers(rng, min_x, max_x - 10)
        bz = _safe_integers(rng, 10, 15)
        world.set_box_solid(bx, cy - 1, bz, bx + 10, cy + 1, bz + 1, VoxelWorld.SOLID)


CATALOG: dict[str, BiomeBrush] = {
    "urban_residential": fill_urban_residential,
    "industrial_refinery": fill_industrial_refinery,
    "civic_government": fill_civic_government,
    "forest_park": fill_forest_park,
    "barrens": fill_barrens,
    "arcology_spire": fill_arcology_spire,
    "sunken_canal": fill_sunken_canal,
    "geo_front": fill_geo_front,
    "glass_desert": fill_glass_desert,
    "command_outpost": fill_command_outpost,
    "dense_jungle": fill_dense_jungle,
    "mining_pit": fill_mining_pit,
    "volcanic_ridge": fill_volcanic_ridge,
    "hydroponic_lab": fill_hydroponic_lab,
    "server_farm": fill_server_farm,
    "orbital_relay": fill_orbital_relay,
}

def get_biome_brush(name: str) -> BiomeBrush:
    return CATALOG.get(name, fill_barrens)