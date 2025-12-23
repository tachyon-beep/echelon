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
        print(f"  Edges: {sum(len(n.edges) for n in graph.nodes.values()):,}")

        # Should build in under 2s
        assert elapsed < 2.0, f"Too slow: {elapsed*1000:.0f} ms"
