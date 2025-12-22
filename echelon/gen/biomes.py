from __future__ import annotations

from typing import Callable, Protocol

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


def _random_box(
    rng: np.random.Generator, min_x: int, min_y: int, max_x: int, max_y: int, w_range: tuple[int, int], h_range: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    if max_x - min_x < w_range[0] or max_y - min_y < h_range[0]:
        return None
    w = rng.integers(w_range[0], min(w_range[1], max_x - min_x) + 1)
    h = rng.integers(h_range[0], min(h_range[1], max_y - min_y) + 1)
    x = rng.integers(min_x, max_x - w + 1)
    y = rng.integers(min_y, max_y - h + 1)
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
    
    # Grid alignment helps the urban feel
    grid_sz = 12
    
    while filled / area < density_target and attempts < 100:
        attempts += 1
        x, y, w, h = _random_box(rng, min_x, min_y, max_x, max_y, (6, 10), (6, 10)) or (0,0,0,0)
        if w == 0: continue

        # Align to grid partially
        x = x - (x % 2)
        y = y - (y % 2)

        # Ensure we don't block existing corridors (simple check: if center is clear, maybe ok? 
        # Rely on the mask check in set_box_solid which is not yet implemented, 
        # so for now we rely on the carver running *after* or a check here).
        # Strategy: Place indiscriminately, then the skeleton carver clears the roads again.
        
        height = rng.integers(4, 9)
        
        # Courtyard shape: Solid box, then clear center
        world.set_box_solid(x, y, 0, x+w, y+h, height, True)
        
        # Clear center (courtyard)
        if w > 4 and h > 4:
            world.set_box_solid(x+2, y+2, 0, x+w-2, y+h-2, height, False)
            # Add a low wall or floor in courtyard?
            # world.set_box_solid(x+2, y+2, 0, x+w-2, y+h-2, 1, True) 
            
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
    
    while filled / area < density_target and attempts < 200:
        attempts += 1
        
        # Tanks (Squares for now)
        x, y, w, h = _random_box(rng, min_x, min_y, max_x, max_y, (3, 6), (3, 6)) or (0,0,0,0)
        if w == 0: continue
        
        height = rng.integers(4, 12)
        world.set_box_solid(x, y, 0, x+w, y+h, height, True)
        filled += w*h

        # Pipe runs (thin long walls)
        if rng.random() < 0.3:
            px, py, pw, ph = _random_box(rng, min_x, min_y, max_x, max_y, (1, 2), (8, 15)) or (0,0,0,0)
            if pw > 0:
                p_height = rng.integers(2, 5)
                # Floating pipe? No, solid for now.
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
    
    while filled / area < density_target and attempts < 50:
        attempts += 1
        
        # Large Block
        x, y, w, h = _random_box(rng, min_x, min_y, max_x, max_y, (8, 15), (8, 15)) or (0,0,0,0)
        if w == 0: continue
        
        height = rng.integers(6, 10)
        world.set_box_solid(x, y, 0, x+w, y+h, height, True)
        filled += w*h
        
        # Plaza walls (low cover ribbons around it)
        margin = 2
        wx, wy = x - margin, y - margin
        ww, wh = w + margin*2, h + margin*2
        # Ensure bounds
        wx = max(min_x, wx); wy = max(min_y, wy)
        ww = min(ww, max_x - wx); wh = min(wh, max_y - wy)
        
        if ww > 0 and wh > 0:
            # Draw box
            world.set_box_solid(wx, wy, 0, wx+ww, wy+wh, 1, True)
            # Clear inner to leave just the rim + building
            world.set_box_solid(wx+1, wy+1, 0, wx+ww-1, wy+wh-1, 1, False)
            # Restore building
            world.set_box_solid(x, y, 0, x+w, y+h, height, True)


def fill_forest_park(
    world: VoxelWorld, min_x: int, min_y: int, max_x: int, max_y: int, rng: np.random.Generator
) -> None:
    """
    Scattered vertical columns (trees) and rock outcrops.
    """
    area = (max_x - min_x) * (max_y - min_y)
    num_trees = int(area * 0.05) # 5% coverage
    
    for _ in range(num_trees):
        tx = rng.integers(min_x, max_x)
        ty = rng.integers(min_y, max_y)
        ts = rng.integers(1, 3) # 1x1 or 2x2 trunk
        th = rng.integers(3, 8)
        world.set_box_solid(tx, ty, 0, tx+ts, ty+ts, th, True)
        
    # Rock outcrops
    num_rocks = int(area * 0.01)
    for _ in range(num_rocks):
        rx, ry, rw, rh = _random_box(rng, min_x, min_y, max_x, max_y, (3, 5), (3, 5)) or (0,0,0,0)
        if rw > 0:
            rheight = rng.integers(2, 4)
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
        rx, ry, rw, rh = _random_box(rng, min_x, min_y, max_x, max_y, (2, 4), (2, 4)) or (0,0,0,0)
        if rw > 0:
            rheight = rng.integers(1, 3)
            world.set_box_solid(rx, ry, 0, rx+rw, ry+rh, rheight, True)


CATALOG: dict[str, BiomeBrush] = {
    "urban_residential": fill_urban_residential,
    "industrial_refinery": fill_industrial_refinery,
    "civic_government": fill_civic_government,
    "forest_park": fill_forest_park,
    "barrens": fill_barrens,
}

def get_biome_brush(name: str) -> BiomeBrush:
    return CATALOG.get(name, fill_barrens)
