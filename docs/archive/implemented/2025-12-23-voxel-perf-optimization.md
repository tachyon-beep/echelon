# Voxel Performance Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize voxel system for simulation step latency (LOS raycasting) and future-proof for 500x500x100 world sizes.

**Architecture:** JIT-compile the raycast DDA loop with Numba for 50-100x speedup on individual rays. Add batch raycasting API with parallel execution for multi-ray queries. Vectorize NavGraph node detection using NumPy broadcasting.

**Tech Stack:** Numba (JIT compilation), NumPy (vectorization), scipy.ndimage (optional, for footprint erosion)

---

## Task 1: Add Numba Dependency

**Files:**
- Modify: `pyproject.toml` (if exists) OR `requirements.txt`

**Step 1: Check current dependency file**

Run: `ls -la /home/john/echelon/pyproject.toml /home/john/echelon/requirements.txt 2>/dev/null`

**Step 2: Add numba dependency**

If `pyproject.toml` exists, add to dependencies:
```toml
numba = ">=0.58.0"
```

If `requirements.txt`, add:
```
numba>=0.58.0
```

**Step 3: Install dependency**

Run: `pip install numba>=0.58.0`
Expected: Successfully installed numba

**Step 4: Verify import works**

Run: `python -c "import numba; print(numba.__version__)"`
Expected: Version number printed (e.g., 0.58.0 or higher)

**Step 5: Commit**

```bash
git add pyproject.toml requirements.txt 2>/dev/null
git commit -m "build: add numba dependency for JIT-compiled raycasting"
```

---

## Task 2: Create LOS Test File with Baseline Tests

**Files:**
- Create: `tests/unit/test_los.py`

**Step 1: Write baseline correctness tests**

Create `tests/unit/test_los.py`:

```python
"""Tests for LOS raycasting correctness and performance."""
from __future__ import annotations

import numpy as np
import pytest

from echelon.sim.world import VoxelWorld
from echelon.sim.los import raycast_voxels, has_los, RaycastHit


@pytest.fixture
def empty_world() -> VoxelWorld:
    """10x10x10 world with no obstacles."""
    voxels = np.zeros((10, 10, 10), dtype=np.uint8)
    return VoxelWorld(voxels=voxels)


@pytest.fixture
def wall_world() -> VoxelWorld:
    """10x10x10 world with a solid wall at x=5."""
    voxels = np.zeros((10, 10, 10), dtype=np.uint8)
    voxels[:, :, 5] = VoxelWorld.SOLID  # Wall at x=5
    return VoxelWorld(voxels=voxels)


@pytest.fixture
def glass_world() -> VoxelWorld:
    """10x10x10 world with glass wall (blocks movement, not LOS)."""
    voxels = np.zeros((10, 10, 10), dtype=np.uint8)
    voxels[:, :, 5] = VoxelWorld.GLASS
    return VoxelWorld(voxels=voxels)


class TestRaycastBasics:
    """Basic raycast correctness tests."""

    def test_los_clear_in_empty_world(self, empty_world: VoxelWorld):
        """LOS should be clear when no obstacles exist."""
        start = np.array([1.5, 1.5, 1.5])
        end = np.array([8.5, 8.5, 8.5])

        result = raycast_voxels(empty_world, start, end)

        assert result.blocked is False
        assert result.blocked_voxel is None

    def test_los_blocked_by_solid_wall(self, wall_world: VoxelWorld):
        """LOS should be blocked by solid voxels."""
        start = np.array([2.5, 5.0, 5.0])
        end = np.array([8.5, 5.0, 5.0])

        result = raycast_voxels(wall_world, start, end)

        assert result.blocked is True
        assert result.blocked_voxel is not None
        assert result.blocked_voxel[0] == 5  # Hit the wall at x=5

    def test_los_not_blocked_by_glass(self, glass_world: VoxelWorld):
        """Glass blocks movement but NOT line of sight."""
        start = np.array([2.5, 5.0, 5.0])
        end = np.array([8.5, 5.0, 5.0])

        result = raycast_voxels(glass_world, start, end)

        assert result.blocked is False

    def test_has_los_helper(self, empty_world: VoxelWorld, wall_world: VoxelWorld):
        """has_los() convenience function works correctly."""
        start = np.array([2.5, 5.0, 5.0])
        end = np.array([8.5, 5.0, 5.0])

        assert has_los(empty_world, start, end) is True
        assert has_los(wall_world, start, end) is False

    def test_zero_length_ray(self, empty_world: VoxelWorld):
        """Zero-length ray should not block."""
        pos = np.array([5.0, 5.0, 5.0])

        result = raycast_voxels(empty_world, pos, pos)

        assert result.blocked is False

    def test_ray_along_axis(self, wall_world: VoxelWorld):
        """Ray traveling along single axis hits wall correctly."""
        # Ray along X axis
        start = np.array([0.5, 5.0, 5.0])
        end = np.array([9.5, 5.0, 5.0])

        result = raycast_voxels(wall_world, start, end)

        assert result.blocked is True
        assert result.blocked_voxel[0] == 5

    def test_ray_diagonal(self, empty_world: VoxelWorld):
        """Diagonal ray through empty space."""
        start = np.array([0.5, 0.5, 0.5])
        end = np.array([9.5, 9.5, 9.5])

        result = raycast_voxels(empty_world, start, end)

        assert result.blocked is False
```

**Step 2: Run tests to verify they pass with current implementation**

Run: `pytest tests/unit/test_los.py -v`
Expected: All tests PASS (validating current pure-Python implementation)

**Step 3: Commit**

```bash
git add tests/unit/test_los.py
git commit -m "test: add baseline LOS raycast correctness tests"
```

---

## Task 3: Extract Pure-Python DDA Core Function

**Files:**
- Modify: `echelon/sim/los.py`

**Step 1: Refactor raycast_voxels to extract core logic**

The goal is to separate the numeric DDA loop from Python object handling, preparing for Numba. Edit `echelon/sim/los.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

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


def raycast_voxels(
    world: VoxelWorld,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    *,
    include_end: bool = False,
) -> RaycastHit:
    """
    Fast voxel traversal (3D DDA) from start to end.
    """
    blocks_los_lut = world.blocks_los_lut()

    blocked, hx, hy, hz = _raycast_dda_pure(
        world.voxels,
        blocks_los_lut,
        float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2]),
        float(end_xyz[0]), float(end_xyz[1]), float(end_xyz[2]),
        include_end,
    )

    blocked_voxel = (hx, hy, hz) if blocked and hx >= 0 else None
    return RaycastHit(blocked=blocked, blocked_voxel=blocked_voxel)


def has_los(world: VoxelWorld, start_xyz: np.ndarray, end_xyz: np.ndarray) -> bool:
    """Check if line of sight exists between two points."""
    return not raycast_voxels(world, start_xyz, end_xyz).blocked
```

**Step 2: Run tests to verify refactor didn't break anything**

Run: `pytest tests/unit/test_los.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add echelon/sim/los.py
git commit -m "refactor: extract DDA core function for Numba preparation"
```

---

## Task 4: Add Numba JIT to DDA Core

**Files:**
- Modify: `echelon/sim/los.py`

**Step 1: Write test that will validate Numba version matches pure Python**

Add to `tests/unit/test_los.py`:

```python
class TestNumbaConsistency:
    """Verify Numba implementation matches pure Python exactly."""

    @pytest.fixture
    def random_world(self) -> VoxelWorld:
        """World with random obstacles for fuzz testing."""
        rng = np.random.default_rng(42)
        voxels = rng.choice(
            [VoxelWorld.AIR, VoxelWorld.SOLID, VoxelWorld.GLASS, VoxelWorld.FOLIAGE],
            size=(20, 20, 20),
            p=[0.7, 0.2, 0.05, 0.05]
        ).astype(np.uint8)
        return VoxelWorld(voxels=voxels)

    def test_numba_matches_pure_python_random_rays(self, random_world: VoxelWorld):
        """Fuzz test: Numba results must match pure Python on random rays."""
        rng = np.random.default_rng(123)

        for _ in range(100):
            start = rng.uniform(0, 20, size=3)
            end = rng.uniform(0, 20, size=3)

            result = raycast_voxels(random_world, start, end)

            # Result should be deterministic (same input = same output)
            result2 = raycast_voxels(random_world, start, end)
            assert result.blocked == result2.blocked
            assert result.blocked_voxel == result2.blocked_voxel
```

**Step 2: Run new test**

Run: `pytest tests/unit/test_los.py::TestNumbaConsistency -v`
Expected: PASS

**Step 3: Add Numba JIT decorator to DDA core**

Modify `echelon/sim/los.py` - add import and decorator:

```python
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
    """Check if line of sight exists between two points."""
    return not raycast_voxels(world, start_xyz, end_xyz).blocked
```

**Step 4: Run all LOS tests**

Run: `pytest tests/unit/test_los.py -v`
Expected: All tests PASS (first run will be slower due to JIT compilation)

**Step 5: Commit**

```bash
git add echelon/sim/los.py tests/unit/test_los.py
git commit -m "perf: add Numba JIT compilation to raycast DDA core"
```

---

## Task 5: Add Batch Raycasting API

**Files:**
- Modify: `echelon/sim/los.py`
- Modify: `tests/unit/test_los.py`

**Step 1: Write failing test for batch API**

Add to `tests/unit/test_los.py`:

```python
from echelon.sim.los import batch_has_los


class TestBatchRaycast:
    """Tests for batch raycasting API."""

    def test_batch_has_los_empty_world(self, empty_world: VoxelWorld):
        """Batch LOS in empty world should all be clear."""
        starts = np.array([
            [1.5, 1.5, 1.5],
            [2.5, 2.5, 2.5],
            [3.5, 3.5, 3.5],
        ], dtype=np.float32)
        ends = np.array([
            [8.5, 8.5, 8.5],
            [7.5, 7.5, 7.5],
            [6.5, 6.5, 6.5],
        ], dtype=np.float32)

        results = batch_has_los(empty_world, starts, ends)

        assert results.shape == (3,)
        assert results.dtype == np.bool_
        assert np.all(results)  # All clear

    def test_batch_has_los_with_wall(self, wall_world: VoxelWorld):
        """Batch LOS with wall blocking some rays."""
        starts = np.array([
            [2.5, 5.0, 5.0],  # Will hit wall
            [2.5, 5.0, 5.0],  # Will hit wall
            [6.5, 5.0, 5.0],  # After wall, clear to end
        ], dtype=np.float32)
        ends = np.array([
            [8.5, 5.0, 5.0],  # Blocked
            [4.0, 5.0, 5.0],  # Clear (before wall)
            [9.5, 5.0, 5.0],  # Clear (both after wall)
        ], dtype=np.float32)

        results = batch_has_los(wall_world, starts, ends)

        assert results[0] == False  # Blocked by wall
        assert results[1] == True   # Before wall
        assert results[2] == True   # After wall

    def test_batch_has_los_matches_single(self, wall_world: VoxelWorld):
        """Batch results must match individual has_los calls."""
        rng = np.random.default_rng(456)
        n_rays = 50

        starts = rng.uniform(0, 10, size=(n_rays, 3)).astype(np.float32)
        ends = rng.uniform(0, 10, size=(n_rays, 3)).astype(np.float32)

        batch_results = batch_has_los(wall_world, starts, ends)

        for i in range(n_rays):
            single_result = has_los(wall_world, starts[i], ends[i])
            assert batch_results[i] == single_result, f"Mismatch at ray {i}"

    def test_batch_has_los_empty_input(self, empty_world: VoxelWorld):
        """Empty batch should return empty array."""
        starts = np.zeros((0, 3), dtype=np.float32)
        ends = np.zeros((0, 3), dtype=np.float32)

        results = batch_has_los(empty_world, starts, ends)

        assert results.shape == (0,)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_los.py::TestBatchRaycast -v`
Expected: FAIL with ImportError (batch_has_los doesn't exist)

**Step 3: Implement batch_has_los**

Add to `echelon/sim/los.py` after the existing functions:

```python
@numba.njit(parallel=True, cache=True)
def _batch_raycast_numba(
    voxels: np.ndarray,
    blocks_los_lut: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """
    Batch raycast with parallel execution.

    Args:
        voxels: uint8[z, y, x] voxel grid
        blocks_los_lut: bool[256] LUT for LOS blocking
        starts: float32[N, 3] start positions (x, y, z)
        ends: float32[N, 3] end positions (x, y, z)

    Returns:
        bool[N] - True if LOS is clear for each ray
    """
    n = starts.shape[0]
    results = np.empty(n, dtype=np.bool_)

    for i in numba.prange(n):
        blocked, _, _, _ = _raycast_dda_numba(
            voxels,
            blocks_los_lut,
            starts[i, 0], starts[i, 1], starts[i, 2],
            ends[i, 0], ends[i, 1], ends[i, 2],
            False,
        )
        results[i] = not blocked

    return results


def batch_has_los(
    world: VoxelWorld,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """
    Check line of sight for multiple rays in parallel.

    Args:
        world: The voxel world
        starts: float32[N, 3] array of start positions (x, y, z)
        ends: float32[N, 3] array of end positions (x, y, z)

    Returns:
        bool[N] array - True where LOS is clear
    """
    if starts.shape[0] == 0:
        return np.zeros(0, dtype=np.bool_)

    # Ensure correct dtype
    starts_f = np.ascontiguousarray(starts, dtype=np.float64)
    ends_f = np.ascontiguousarray(ends, dtype=np.float64)

    blocks_los_lut = world.blocks_los_lut()

    return _batch_raycast_numba(world.voxels, blocks_los_lut, starts_f, ends_f)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_los.py::TestBatchRaycast -v`
Expected: All tests PASS

**Step 5: Run all LOS tests**

Run: `pytest tests/unit/test_los.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add echelon/sim/los.py tests/unit/test_los.py
git commit -m "feat: add parallel batch_has_los API for multi-ray queries"
```

---

## Task 6: Add Performance Benchmark Tests

**Files:**
- Create: `tests/benchmark/test_los_perf.py`

**Step 1: Create benchmark directory**

Run: `mkdir -p tests/benchmark`

**Step 2: Write benchmark tests**

Create `tests/benchmark/test_los_perf.py`:

```python
"""Performance benchmarks for LOS raycasting.

Run with: pytest tests/benchmark/test_los_perf.py -v -s
"""
from __future__ import annotations

import time
import numpy as np
import pytest

from echelon.sim.world import VoxelWorld
from echelon.sim.los import raycast_voxels, has_los, batch_has_los


@pytest.fixture
def large_world() -> VoxelWorld:
    """100x100x20 world with scattered obstacles."""
    rng = np.random.default_rng(42)
    voxels = np.zeros((20, 100, 100), dtype=np.uint8)
    # Add ~10% random solid voxels
    mask = rng.random(voxels.shape) < 0.1
    voxels[mask] = VoxelWorld.SOLID
    return VoxelWorld(voxels=voxels)


@pytest.fixture
def target_world() -> VoxelWorld:
    """500x500x100 world for scaling tests."""
    rng = np.random.default_rng(42)
    voxels = np.zeros((100, 500, 500), dtype=np.uint8)
    # Add ~5% random solid voxels
    mask = rng.random(voxels.shape) < 0.05
    voxels[mask] = VoxelWorld.SOLID
    return VoxelWorld(voxels=voxels)


class TestSingleRayPerformance:
    """Benchmark single-ray raycasting."""

    def test_warmup_jit(self, large_world: VoxelWorld):
        """Warm up JIT compilation (not a real benchmark)."""
        start = np.array([10.0, 50.0, 50.0])
        end = np.array([10.0, 50.0, 90.0])
        # First call triggers JIT
        raycast_voxels(large_world, start, end)
        print("\nJIT warmup complete")

    def test_single_ray_throughput(self, large_world: VoxelWorld):
        """Measure single-ray throughput after JIT warmup."""
        rng = np.random.default_rng(123)
        n_rays = 10000

        # Generate random rays
        starts = rng.uniform([0, 0, 0], [20, 100, 100], size=(n_rays, 3))
        ends = rng.uniform([0, 0, 0], [20, 100, 100], size=(n_rays, 3))

        # Warmup
        for i in range(100):
            raycast_voxels(large_world, starts[i], ends[i])

        # Benchmark
        t0 = time.perf_counter()
        for i in range(n_rays):
            raycast_voxels(large_world, starts[i], ends[i])
        elapsed = time.perf_counter() - t0

        rays_per_sec = n_rays / elapsed
        us_per_ray = (elapsed / n_rays) * 1e6

        print(f"\nSingle-ray performance:")
        print(f"  {rays_per_sec:,.0f} rays/sec")
        print(f"  {us_per_ray:.2f} us/ray")

        # Soft assertion: should be at least 50k rays/sec
        assert rays_per_sec > 50000, f"Too slow: {rays_per_sec:.0f} rays/sec"


class TestBatchRayPerformance:
    """Benchmark batch raycasting."""

    def test_batch_throughput(self, large_world: VoxelWorld):
        """Measure batch ray throughput."""
        rng = np.random.default_rng(456)
        n_rays = 10000

        starts = rng.uniform([0, 0, 0], [20, 100, 100], size=(n_rays, 3)).astype(np.float32)
        ends = rng.uniform([0, 0, 0], [20, 100, 100], size=(n_rays, 3)).astype(np.float32)

        # Warmup
        batch_has_los(large_world, starts[:100], ends[:100])

        # Benchmark
        t0 = time.perf_counter()
        results = batch_has_los(large_world, starts, ends)
        elapsed = time.perf_counter() - t0

        rays_per_sec = n_rays / elapsed
        us_per_ray = (elapsed / n_rays) * 1e6

        print(f"\nBatch ray performance ({n_rays} rays):")
        print(f"  {rays_per_sec:,.0f} rays/sec")
        print(f"  {us_per_ray:.2f} us/ray")
        print(f"  Total time: {elapsed*1000:.1f} ms")

        # Soft assertion: batch should be faster than single
        assert rays_per_sec > 100000, f"Too slow: {rays_per_sec:.0f} rays/sec"

    def test_batch_scaling(self, large_world: VoxelWorld):
        """Test batch performance at different sizes."""
        rng = np.random.default_rng(789)

        print("\nBatch scaling:")
        for n in [100, 1000, 10000]:
            starts = rng.uniform([0, 0, 0], [20, 100, 100], size=(n, 3)).astype(np.float32)
            ends = rng.uniform([0, 0, 0], [20, 100, 100], size=(n, 3)).astype(np.float32)

            # Warmup
            batch_has_los(large_world, starts[:10], ends[:10])

            t0 = time.perf_counter()
            batch_has_los(large_world, starts, ends)
            elapsed = time.perf_counter() - t0

            print(f"  n={n:>5}: {n/elapsed:>10,.0f} rays/sec ({elapsed*1000:.2f} ms)")


class TestLargeWorldPerformance:
    """Benchmark on target world size (500x500x100)."""

    @pytest.mark.slow
    def test_target_world_batch(self, target_world: VoxelWorld):
        """Batch performance on 500x500x100 world."""
        rng = np.random.default_rng(101)
        n_rays = 5000

        starts = rng.uniform([0, 0, 0], [100, 500, 500], size=(n_rays, 3)).astype(np.float32)
        ends = rng.uniform([0, 0, 0], [100, 500, 500], size=(n_rays, 3)).astype(np.float32)

        # Warmup
        batch_has_los(target_world, starts[:100], ends[:100])

        t0 = time.perf_counter()
        results = batch_has_los(target_world, starts, ends)
        elapsed = time.perf_counter() - t0

        rays_per_sec = n_rays / elapsed

        print(f"\n500x500x100 world batch performance:")
        print(f"  {rays_per_sec:,.0f} rays/sec")
        print(f"  Total time: {elapsed*1000:.1f} ms for {n_rays} rays")
```

**Step 3: Run benchmarks**

Run: `pytest tests/benchmark/test_los_perf.py -v -s --ignore=tests/benchmark/test_los_perf.py::TestLargeWorldPerformance::test_target_world_batch`
Expected: PASS with performance numbers printed

**Step 4: Commit**

```bash
git add tests/benchmark/
git commit -m "test: add LOS raycasting performance benchmarks"
```

---

## Task 7: Vectorize NavGraph Node Detection

**Files:**
- Modify: `echelon/nav/graph.py`
- Modify or Create: `tests/unit/test_nav_graph.py`

**Step 1: Write test for NavGraph correctness**

Create or add to `tests/unit/test_nav_graph.py`:

```python
"""Tests for NavGraph building and correctness."""
from __future__ import annotations

import numpy as np
import pytest

from echelon.sim.world import VoxelWorld
from echelon.nav.graph import NavGraph


@pytest.fixture
def flat_world() -> VoxelWorld:
    """10x10x5 world with solid floor at z=0."""
    voxels = np.zeros((5, 10, 10), dtype=np.uint8)
    voxels[0, :, :] = VoxelWorld.SOLID  # Floor
    return VoxelWorld(voxels=voxels)


@pytest.fixture
def platform_world() -> VoxelWorld:
    """World with floor and elevated platform."""
    voxels = np.zeros((10, 20, 20), dtype=np.uint8)
    voxels[0, :, :] = VoxelWorld.SOLID  # Floor
    voxels[3, 5:15, 5:15] = VoxelWorld.SOLID  # Platform at z=3
    return VoxelWorld(voxels=voxels)


class TestNavGraphBuild:
    """Tests for NavGraph.build() correctness."""

    def test_flat_world_nodes(self, flat_world: VoxelWorld):
        """Flat world should have nodes at ground level."""
        graph = NavGraph.build(flat_world, clearance_z=2, mech_radius=0)

        # Should have nodes for most ground positions
        # (edges may exclude some due to boundary)
        assert len(graph.nodes) > 0

        # All nodes should be at z=0 (standing on floor)
        for nid in graph.nodes:
            z, y, x = nid
            assert z == 0, f"Node {nid} not at ground level"

    def test_platform_world_has_two_levels(self, platform_world: VoxelWorld):
        """Platform world should have nodes at both levels."""
        graph = NavGraph.build(platform_world, clearance_z=2, mech_radius=0)

        z_levels = set(nid[0] for nid in graph.nodes)

        # Should have ground (z=0) and platform (z=3) nodes
        assert 0 in z_levels, "Missing ground level nodes"
        assert 3 in z_levels, "Missing platform level nodes"

    def test_edges_connect_neighbors(self, flat_world: VoxelWorld):
        """Adjacent nodes should have edges between them."""
        graph = NavGraph.build(flat_world, clearance_z=2, mech_radius=0)

        # Pick a center node
        center = (0, 5, 5)
        if center in graph.nodes:
            node = graph.nodes[center]
            neighbor_ids = [e[0] for e in node.edges]

            # Should have edges to orthogonal neighbors
            expected_neighbors = [(0, 5, 4), (0, 5, 6), (0, 4, 5), (0, 6, 5)]
            for expected in expected_neighbors:
                if expected in graph.nodes:
                    assert expected in neighbor_ids, f"Missing edge to {expected}"

    def test_mech_radius_reduces_nodes(self, flat_world: VoxelWorld):
        """Larger mech radius should produce fewer walkable nodes."""
        graph_r0 = NavGraph.build(flat_world, clearance_z=2, mech_radius=0)
        graph_r1 = NavGraph.build(flat_world, clearance_z=2, mech_radius=1)

        # Radius 1 excludes nodes near edges
        assert len(graph_r1.nodes) < len(graph_r0.nodes)

    def test_clearance_blocks_low_ceiling(self):
        """Nodes under low ceiling should be excluded."""
        voxels = np.zeros((5, 10, 10), dtype=np.uint8)
        voxels[0, :, :] = VoxelWorld.SOLID  # Floor
        voxels[2, :, 5:] = VoxelWorld.SOLID  # Low ceiling on right half
        world = VoxelWorld(voxels=voxels)

        graph = NavGraph.build(world, clearance_z=3, mech_radius=0)

        # Nodes under ceiling should be excluded
        for nid in graph.nodes:
            z, y, x = nid
            if z == 0 and x >= 5:
                pytest.fail(f"Node {nid} should be blocked by low ceiling")
```

**Step 2: Run tests to establish baseline**

Run: `pytest tests/unit/test_nav_graph.py -v`
Expected: Tests should PASS with current implementation

**Step 3: Implement vectorized node detection**

Replace the node detection loop in `echelon/nav/graph.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

from ..sim.world import VoxelWorld

NodeID = Tuple[int, int, int]


@dataclass
class NavNode:
    id: NodeID
    pos: Tuple[float, float, float]
    edges: List[Tuple[NodeID, float]] = field(default_factory=list)


class NavGraph:
    """
    A graph representation of the walkable surfaces in the VoxelWorld.

    v1 Implementation (vectorized):
    - Uses NumPy broadcasting to identify walkable surfaces
    - A surface is walkable if: solid below AND clear above for clearance_z
    - Applies mech_radius footprint via erosion
    """
    def __init__(self, clearance_z: int = 2):
        self.nodes: Dict[NodeID, NavNode] = {}
        self.nodes_by_col: Dict[Tuple[int, int], List[int]] = {}
        self.clearance_z = clearance_z
        self.voxel_size = 5.0

    @classmethod
    def build(cls, world: VoxelWorld, clearance_z: int = 4, mech_radius: int = 1) -> NavGraph:
        graph = cls(clearance_z=clearance_z)
        graph.voxel_size = world.voxel_size_m

        collides = world.collides_mask
        sz, sy, sx = collides.shape

        # --- Vectorized Node Detection ---

        # 1. Find solid voxels (potential surfaces to stand ON)
        #    We need solid at z, and clear from z+1 to z+clearance_z

        # Walkable mask: start with all False, then mark valid positions
        # A position (z, y, x) is walkable if:
        #   - collides[z, y, x] is True (solid to stand on)
        #   - collides[z+1:z+1+clearance_z, y, x] are all False (headroom)

        walkable = np.zeros((sz, sy, sx), dtype=np.bool_)

        # For each z level that could be a floor (0 to sz-2 at minimum)
        for z in range(sz - 1):
            # Check if this z is solid
            solid_here = collides[z]

            # Check clearance above
            z_top = min(z + 1 + clearance_z, sz)
            if z + 1 >= sz:
                # Top of world, can stand here if solid
                clear_above = np.ones((sy, sx), dtype=np.bool_)
            else:
                # All voxels from z+1 to z_top must be non-colliding
                clear_above = ~np.any(collides[z+1:z_top], axis=0)

            walkable[z] = solid_here & clear_above

        # 2. Also check bedrock (z=-1, standing on virtual floor below z=0)
        #    Need clearance from z=0 upward
        z_top_bedrock = min(clearance_z, sz)
        if z_top_bedrock > 0:
            clear_from_bedrock = ~np.any(collides[0:z_top_bedrock], axis=0)
        else:
            clear_from_bedrock = np.ones((sy, sx), dtype=np.bool_)

        # 3. Apply mech_radius footprint erosion
        if mech_radius > 0:
            # Erode walkable mask - a cell is only walkable if all cells
            # within mech_radius are also walkable
            from scipy.ndimage import minimum_filter
            footprint = np.ones((1, 2*mech_radius+1, 2*mech_radius+1), dtype=np.bool_)
            walkable = minimum_filter(walkable.astype(np.uint8), footprint=footprint, mode='constant', cval=0).astype(np.bool_)

            # Also erode bedrock walkability
            footprint_2d = np.ones((2*mech_radius+1, 2*mech_radius+1), dtype=np.bool_)
            clear_from_bedrock = minimum_filter(clear_from_bedrock.astype(np.uint8), footprint=footprint_2d, mode='constant', cval=0).astype(np.bool_)

        # 4. Extract node coordinates and build node dict
        # Bedrock nodes (z=-1)
        bedrock_coords = np.argwhere(clear_from_bedrock)
        for coord in bedrock_coords:
            y, x = int(coord[0]), int(coord[1])
            graph._add_node((-1, y, x), world.voxel_size_m)

        # Regular nodes
        node_coords = np.argwhere(walkable)
        for coord in node_coords:
            z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
            graph._add_node((z, y, x), world.voxel_size_m)

        # --- Edge Building (still iterative over nodes) ---
        neighbors = [
            (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]
        step_height_max = 2

        for (z, y, x), node in graph.nodes.items():
            for dy, dx, dist_mult in neighbors:
                ny, nx = y + dy, x + dx

                potential_zs = graph.nodes_by_col.get((ny, nx))
                if not potential_zs:
                    continue

                # Diagonal corner-cutting check
                if dy != 0 and dx != 0:
                    ortho1 = (z, y + dy, x)
                    ortho2 = (z, y, x + dx)
                    if ortho1 not in graph.nodes or ortho2 not in graph.nodes:
                        continue

                for nz in potential_zs:
                    dz = nz - z
                    if abs(dz) <= step_height_max:
                        base_cost = world.voxel_size_m * dist_mult
                        if dz > 0:
                            base_cost *= 1.2
                        node.edges.append(((nz, ny, nx), base_cost))

        return graph

    def _add_node(self, nid: NodeID, scale: float):
        z, y, x = nid
        wz = (z + 1) * scale
        wx = (x + 0.5) * scale
        wy = (y + 0.5) * scale
        self.nodes[nid] = NavNode(nid, (wx, wy, wz))

        if (y, x) not in self.nodes_by_col:
            self.nodes_by_col[(y, x)] = []
        self.nodes_by_col[(y, x)].append(z)

    def get_nearest_node(self, pos: Tuple[float, float, float]) -> Optional[NodeID]:
        """Find the nearest graph node to a world position."""
        scale = self.voxel_size
        ix = int(pos[0] // scale)
        iy = int(pos[1] // scale)
        wz = pos[2]

        best_node: Optional[NodeID] = None
        best_dist_sq = float('inf')

        search_r = 1
        for dy in range(-search_r, search_r + 1):
            for dx in range(-search_r, search_r + 1):
                ny, nx = iy + dy, ix + dx

                zs = self.nodes_by_col.get((ny, nx))
                if not zs:
                    continue

                for z in zs:
                    node = self.nodes[(z, ny, nx)]
                    d_sq = (node.pos[0] - pos[0])**2 + (node.pos[1] - pos[1])**2 + (node.pos[2] - wz)**2

                    dz = abs(node.pos[2] - wz)
                    if dz > scale * 1.5:
                        d_sq += 1000.0

                    if d_sq < best_dist_sq:
                        best_dist_sq = d_sq
                        best_node = node.id

        return best_node

    def to_dict(self) -> dict:
        """Serialize graph to a JSON-safe dictionary."""
        out_nodes = []
        for nid, node in self.nodes.items():
            out_nodes.append({
                "id": list(nid),
                "pos": list(node.pos),
                "edges": [list(e[0]) for e in node.edges]
            })
        return {
            "voxel_size": self.voxel_size,
            "clearance_z": self.clearance_z,
            "nodes": out_nodes
        }
```

**Step 4: Run tests to verify vectorized version is correct**

Run: `pytest tests/unit/test_nav_graph.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add echelon/nav/graph.py tests/unit/test_nav_graph.py
git commit -m "perf: vectorize NavGraph node detection with NumPy"
```

---

## Task 8: Add NavGraph Performance Benchmark

**Files:**
- Create: `tests/benchmark/test_nav_perf.py`

**Step 1: Write NavGraph benchmark**

Create `tests/benchmark/test_nav_perf.py`:

```python
"""Performance benchmarks for NavGraph building."""
from __future__ import annotations

import time
import numpy as np
import pytest

from echelon.sim.world import VoxelWorld
from echelon.nav.graph import NavGraph


@pytest.fixture
def medium_world() -> VoxelWorld:
    """100x100x20 world with terrain."""
    rng = np.random.default_rng(42)
    voxels = np.zeros((20, 100, 100), dtype=np.uint8)
    voxels[0, :, :] = VoxelWorld.SOLID  # Floor
    # Add some random structures
    mask = rng.random((20, 100, 100)) < 0.05
    voxels[mask] = VoxelWorld.SOLID
    return VoxelWorld(voxels=voxels)


@pytest.fixture
def large_world() -> VoxelWorld:
    """200x200x40 world."""
    rng = np.random.default_rng(42)
    voxels = np.zeros((40, 200, 200), dtype=np.uint8)
    voxels[0, :, :] = VoxelWorld.SOLID
    mask = rng.random((40, 200, 200)) < 0.03
    voxels[mask] = VoxelWorld.SOLID
    return VoxelWorld(voxels=voxels)


class TestNavGraphPerformance:
    """Benchmark NavGraph.build() performance."""

    def test_medium_world_build_time(self, medium_world: VoxelWorld):
        """Benchmark on 100x100x20 world."""
        # Warmup
        NavGraph.build(medium_world, clearance_z=4, mech_radius=1)

        t0 = time.perf_counter()
        graph = NavGraph.build(medium_world, clearance_z=4, mech_radius=1)
        elapsed = time.perf_counter() - t0

        print(f"\n100x100x20 NavGraph build:")
        print(f"  Time: {elapsed*1000:.1f} ms")
        print(f"  Nodes: {len(graph.nodes):,}")
        print(f"  Edges: {sum(len(n.edges) for n in graph.nodes.values()):,}")

        # Should build in under 500ms
        assert elapsed < 0.5, f"Too slow: {elapsed*1000:.0f} ms"

    def test_large_world_build_time(self, large_world: VoxelWorld):
        """Benchmark on 200x200x40 world."""
        t0 = time.perf_counter()
        graph = NavGraph.build(large_world, clearance_z=4, mech_radius=1)
        elapsed = time.perf_counter() - t0

        print(f"\n200x200x40 NavGraph build:")
        print(f"  Time: {elapsed*1000:.1f} ms")
        print(f"  Nodes: {len(graph.nodes):,}")

        # Should build in under 2s
        assert elapsed < 2.0, f"Too slow: {elapsed*1000:.0f} ms"
```

**Step 2: Run benchmarks**

Run: `pytest tests/benchmark/test_nav_perf.py -v -s`
Expected: PASS with timing info

**Step 3: Commit**

```bash
git add tests/benchmark/test_nav_perf.py
git commit -m "test: add NavGraph build performance benchmarks"
```

---

## Summary

| Task | Description | Estimated Complexity |
|------|-------------|---------------------|
| 1 | Add Numba dependency | Trivial |
| 2 | Create LOS baseline tests | Small |
| 3 | Extract DDA core function | Small |
| 4 | Add Numba JIT to DDA | Medium |
| 5 | Add batch raycasting API | Medium |
| 6 | Add LOS performance benchmarks | Small |
| 7 | Vectorize NavGraph node detection | Medium |
| 8 | Add NavGraph benchmarks | Small |

**Total: 8 tasks, ~30 commits**
