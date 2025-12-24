import os
import time

import numpy as np

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig


def test_benchmark_sps():
    """
    Benchmark the simulation Steps Per Second (SPS).
    Fails if performance drops below a critical threshold.
    """
    # Config for a "heavy" load: 2 packs (20 mechs), large map
    cfg = EnvConfig(
        world=WorldConfig(size_x=50, size_y=50, size_z=20, obstacle_fill=0.2),
        num_packs=2,
        seed=123,
        observation_mode="full",
    )
    env = EchelonEnv(cfg)
    env.reset()

    # Pre-allocate random actions to avoid measuring RNG overhead.
    rng = np.random.default_rng(123)
    precomputed_actions = []
    for _ in range(100):
        actions = {}
        for aid in env.agents:
            # Random actions to trigger physics/collisions
            actions[aid] = rng.uniform(-1.0, 1.0, size=env.ACTION_DIM).astype(np.float32)
        precomputed_actions.append(actions)

    warmup_steps = 10
    measure_steps = len(precomputed_actions)

    # Warmup
    for _ in range(warmup_steps):
        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        env.step(actions)

    start_time = time.perf_counter()

    for actions in precomputed_actions:
        env.step(actions)

    end_time = time.perf_counter()
    duration = end_time - start_time
    sps = measure_steps / duration

    min_sps = float(os.environ.get("ECHELON_SPS_MIN", "5.0"))
    print(f"Benchmark SPS: {sps:.2f} (Steps: {measure_steps}, Time: {duration:.2f}s, Min: {min_sps:.2f})")

    # Threshold: configurable for heterogeneous CI and dev hardware.
    assert sps > min_sps, f"SPS too low: {sps:.2f} < {min_sps:.2f}"


def test_memory_stability():
    """
    Run for N steps and ensure memory doesn't explode.
    Uses 'tracemalloc' to check for gross leaks.
    """
    import tracemalloc

    cfg = EnvConfig(
        world=WorldConfig(size_x=30, size_y=30, size_z=10),
        num_packs=1,
        seed=42,
    )
    env = EchelonEnv(cfg)
    env.reset()

    tracemalloc.start()

    # Baseline snapshot
    snapshot1 = tracemalloc.take_snapshot()

    steps = 200
    for _ in range(steps):
        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        env.step(actions)

    snapshot2 = tracemalloc.take_snapshot()

    # Filter for significant growth
    top_stats = snapshot2.compare_to(snapshot1, "lineno")

    # Heuristic: Total allocated size shouldn't grow by > 10MB for 200 steps
    # Note: Python GC is lazy, so this is noisy.
    # We just print top offenders for now and assert strict failure only on massive growth.

    total_growth = sum(stat.size_diff for stat in top_stats)
    print(f"Memory growth over {steps} steps: {total_growth / 1024 / 1024:.2f} MB")

    # 50MB buffer
    assert total_growth < 50 * 1024 * 1024, "Memory usage grew by > 50MB!"

    tracemalloc.stop()
