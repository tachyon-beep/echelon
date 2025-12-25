"""Tests for SpatialAccumulator heatmap generation."""

import numpy as np

from echelon.training.spatial import SpatialAccumulator


class TestSpatialAccumulator:
    """Tests for SpatialAccumulator class."""

    def test_record_death(self) -> None:
        """Death at center of 100x100 world maps to center of 16x16 grid."""
        acc = SpatialAccumulator(grid_size=16)
        acc.record_death(50.0, 50.0, world_size=(100, 100))

        # Should be in center cell (8, 8)
        assert acc.death_locations[8, 8] == 1.0

    def test_record_multiple(self) -> None:
        """Multiple deaths at same location accumulate."""
        acc = SpatialAccumulator(grid_size=16)
        acc.record_death(50.0, 50.0, world_size=(100, 100))
        acc.record_death(50.0, 50.0, world_size=(100, 100))

        assert acc.death_locations[8, 8] == 2.0

    def test_record_damage(self) -> None:
        """Damage records at correct grid cell with damage value."""
        acc = SpatialAccumulator(grid_size=16)
        acc.record_damage(25.0, 75.0, damage=10.0, world_size=(100, 100))

        # (25, 75) -> gx=4, gy=12 in 16x16 grid
        # Array indexed as [gy, gx] so [12, 4]
        assert acc.damage_locations[12, 4] == 10.0

    def test_record_position(self) -> None:
        """Position at origin maps to (0, 0) grid cell."""
        acc = SpatialAccumulator(grid_size=16)
        acc.record_position(0.0, 0.0, world_size=(100, 100))

        assert acc.movement_density[0, 0] == 1.0

    def test_reset(self) -> None:
        """Reset clears all accumulators."""
        acc = SpatialAccumulator(grid_size=16)
        acc.record_death(50.0, 50.0, world_size=(100, 100))
        acc.record_damage(25.0, 75.0, damage=10.0, world_size=(100, 100))
        acc.record_position(10.0, 10.0, world_size=(100, 100))
        acc.reset()

        assert acc.death_locations.sum() == 0.0
        assert acc.damage_locations.sum() == 0.0
        assert acc.movement_density.sum() == 0.0

    def test_to_images_returns_dict_type(self) -> None:
        """to_images returns a dict (may be empty if matplotlib unavailable)."""
        acc = SpatialAccumulator(grid_size=16)
        acc.record_death(50.0, 50.0, world_size=(100, 100))

        # to_images returns dict always - may be empty if matplotlib not installed
        images = acc.to_images()
        assert isinstance(images, dict)

    def test_to_images_empty_for_empty_data(self) -> None:
        """to_images returns empty dict when no data recorded."""
        acc = SpatialAccumulator(grid_size=16)
        # Don't record any events

        images = acc.to_images()
        assert isinstance(images, dict)
        # No data means no images (even if matplotlib is available)
        assert images == {}

    def test_grid_boundary_clamping(self) -> None:
        """Coordinates at world edges clamp to valid grid indices."""
        acc = SpatialAccumulator(grid_size=16)

        # Test max boundary
        acc.record_death(100.0, 100.0, world_size=(100, 100))
        assert acc.death_locations[15, 15] == 1.0

        # Test negative (should clamp to 0)
        acc.record_death(-10.0, -10.0, world_size=(100, 100))
        assert acc.death_locations[0, 0] == 1.0

    def test_default_grid_size(self) -> None:
        """Default grid size is 32."""
        acc = SpatialAccumulator()
        assert acc.grid_size == 32
        assert acc.death_locations.shape == (32, 32)

    def test_accumulator_shapes(self) -> None:
        """All accumulators have correct shape."""
        acc = SpatialAccumulator(grid_size=16)
        assert acc.death_locations.shape == (16, 16)
        assert acc.damage_locations.shape == (16, 16)
        assert acc.movement_density.shape == (16, 16)

    def test_accumulator_dtypes(self) -> None:
        """All accumulators are float32."""
        acc = SpatialAccumulator(grid_size=16)
        assert acc.death_locations.dtype == np.float32
        assert acc.damage_locations.dtype == np.float32
        assert acc.movement_density.dtype == np.float32
