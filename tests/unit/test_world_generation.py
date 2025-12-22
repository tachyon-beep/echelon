import numpy as np

from echelon.config import WorldConfig
from echelon.sim.world import VoxelWorld


def _seed_for_archetype(target: int, *, max_seed: int = 256) -> int:
    for seed in range(max_seed):
        rng = np.random.default_rng(seed)
        if int(rng.integers(0, 3)) == target:
            return seed
    raise AssertionError(f"Failed to find a seed for archetype {target} in [0,{max_seed})")


def test_world_generate_small_map_highway_does_not_crash():
    # Regression test: `VoxelWorld.generate` used to crash on small maps when archetype==2 (HIGHWAY)
    # due to an invalid `rng.integers(0, size-10)` range.
    cfg = WorldConfig(size_x=10, size_y=10, size_z=10, obstacle_fill=0.0, ensure_connectivity=False)
    seed = _seed_for_archetype(2)
    world = VoxelWorld.generate(cfg, np.random.default_rng(seed))
    assert world.size_x == cfg.size_x
    assert world.size_y == cfg.size_y
    assert world.size_z == cfg.size_z
    assert int(world.meta.get("archetype", -1)) == 2

