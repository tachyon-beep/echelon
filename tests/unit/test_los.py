"""Tests for LOS raycasting correctness and performance."""

from __future__ import annotations

import numpy as np
import pytest

from echelon.sim.los import (
    _raycast_dda_numba,
    _raycast_dda_pure,
    batch_has_los,
    has_los,
    raycast_voxels,
)
from echelon.sim.world import VoxelWorld


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


class TestNumbaConsistency:
    """Verify Numba implementation matches pure Python exactly."""

    @pytest.fixture
    def random_world(self) -> VoxelWorld:
        """World with random obstacles for fuzz testing."""
        rng = np.random.default_rng(42)
        voxels = rng.choice(
            [VoxelWorld.AIR, VoxelWorld.SOLID, VoxelWorld.GLASS, VoxelWorld.FOLIAGE],
            size=(20, 20, 20),
            p=[0.7, 0.2, 0.05, 0.05],
        ).astype(np.uint8)
        return VoxelWorld(voxels=voxels)

    def test_numba_matches_pure_python_random_rays(self, random_world: VoxelWorld):
        """Fuzz test: Numba results must match pure Python on random rays."""
        rng = np.random.default_rng(123)
        blocks_los_lut = random_world.blocks_los_lut()

        for _ in range(100):
            start = rng.uniform(0, 20, size=3)
            end = rng.uniform(0, 20, size=3)

            # Call both pure Python and Numba implementations directly
            result_pure = _raycast_dda_pure(
                random_world.voxels,
                blocks_los_lut,
                float(start[0]),
                float(start[1]),
                float(start[2]),
                float(end[0]),
                float(end[1]),
                float(end[2]),
                False,
            )

            result_numba = _raycast_dda_numba(
                random_world.voxels,
                blocks_los_lut,
                float(start[0]),
                float(start[1]),
                float(start[2]),
                float(end[0]),
                float(end[1]),
                float(end[2]),
                False,
            )

            # Results must be identical
            assert result_pure == result_numba


class TestBatchRaycast:
    """Tests for batch raycasting API."""

    def test_batch_has_los_empty_world(self, empty_world: VoxelWorld):
        """Batch LOS in empty world should all be clear."""
        starts = np.array(
            [
                [1.5, 1.5, 1.5],
                [2.5, 2.5, 2.5],
                [3.5, 3.5, 3.5],
            ],
            dtype=np.float32,
        )
        ends = np.array(
            [
                [8.5, 8.5, 8.5],
                [7.5, 7.5, 7.5],
                [6.5, 6.5, 6.5],
            ],
            dtype=np.float32,
        )

        results = batch_has_los(empty_world, starts, ends)

        assert results.shape == (3,)
        assert results.dtype == np.bool_
        assert np.all(results)  # All clear

    def test_batch_has_los_with_wall(self, wall_world: VoxelWorld):
        """Batch LOS with wall blocking some rays."""
        starts = np.array(
            [
                [2.5, 5.0, 5.0],  # Will hit wall
                [2.5, 5.0, 5.0],  # Will hit wall
                [6.5, 5.0, 5.0],  # After wall, clear to end
            ],
            dtype=np.float32,
        )
        ends = np.array(
            [
                [8.5, 5.0, 5.0],  # Blocked
                [4.0, 5.0, 5.0],  # Clear (before wall)
                [9.5, 5.0, 5.0],  # Clear (both after wall)
            ],
            dtype=np.float32,
        )

        results = batch_has_los(wall_world, starts, ends)

        assert not results[0]  # Blocked by wall
        assert results[1]  # Before wall
        assert results[2]  # After wall

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
