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
