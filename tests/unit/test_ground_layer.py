import numpy as np

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig
from echelon.sim.world import VoxelWorld


def test_env_fills_dirt_ground_layer():
    cfg = EnvConfig(
        world=WorldConfig(size_x=20, size_y=20, size_z=10, obstacle_fill=0.1, ensure_connectivity=False),
        num_packs=1,
        seed=123,
        max_episode_seconds=1.0,
    )
    env = EchelonEnv(cfg)
    env.reset(seed=123)
    assert env.world is not None
    assert env.world.size_z > 0
    assert not bool(np.any(env.world.voxels[0] == VoxelWorld.AIR))
