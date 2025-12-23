"""Tests for LOS raycasting correctness and performance."""
from __future__ import annotations

import numpy as np
import pytest

from echelon.sim.world import VoxelWorld
from echelon.sim.los import raycast_voxels, has_los, RaycastHit, _raycast_dda_pure, _raycast_dda_numba


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
            p=[0.7, 0.2, 0.05, 0.05]
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
                float(start[0]), float(start[1]), float(start[2]),
                float(end[0]), float(end[1]), float(end[2]),
                False,
            )

            result_numba = _raycast_dda_numba(
                random_world.voxels,
                blocks_los_lut,
                float(start[0]), float(start[1]), float(start[2]),
                float(end[0]), float(end[1]), float(end[2]),
                False,
            )

            # Results must be identical
            assert result_pure == result_numba
