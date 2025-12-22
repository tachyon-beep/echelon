import numpy as np
import pytest
from echelon.config import WorldConfig
from echelon.sim.world import VoxelWorld
from echelon.gen.layout import generate_layout
from echelon.gen.biomes import CATALOG

def test_layout_generation_basics():
    cfg = WorldConfig(size_x=50, size_y=50, size_z=20)
    rng = np.random.default_rng(42)
    world = VoxelWorld.generate(cfg, rng)
    
    # Check Dimensions
    assert world.size_x == 50
    assert world.size_y == 50
    assert world.size_z == 20
    
    # Check Metadata
    assert world.meta.get("generator") == "layout_v2"
    assert "biome_layout" in world.meta
    layout = world.meta["biome_layout"]
    assert len(layout) == 4
    assert set(layout.keys()) == {"TL", "TR", "BL", "BR"}
    
    # Check Biomes are valid
    for biome in layout.values():
        assert biome in CATALOG

def test_layout_jitter_determinism():
    cfg = WorldConfig(size_x=100, size_y=100)
    
    # Same seed -> Same layout
    rng1 = np.random.default_rng(123)
    world1 = generate_layout(cfg, rng1)
    
    rng2 = np.random.default_rng(123)
    world2 = generate_layout(cfg, rng2)
    
    assert world1.meta["biome_layout"] == world2.meta["biome_layout"]
    np.testing.assert_array_equal(world1.voxels, world2.voxels)
    
    # Different seed -> Likely different
    rng3 = np.random.default_rng(124)
    world3 = generate_layout(cfg, rng3)
    # Note: Small chance of collision, but highly unlikely with random quadrant jitter + biome perm
    if world1.meta["biome_layout"] == world3.meta["biome_layout"]:
        # If biomes happen to match, voxels should differ due to brush RNG
        assert not np.array_equal(world1.voxels, world3.voxels)

def test_small_map_support():
    # Should not crash on tiny maps (e.g. 20x20)
    cfg = WorldConfig(size_x=20, size_y=20, size_z=10)
    rng = np.random.default_rng(999)
    world = generate_layout(cfg, rng)
    assert world.size_x == 20

def test_biome_brush_application():
    # Verify that biomes actually place blocks (not empty)
    # We pick a seed that we know produces some output
    cfg = WorldConfig(size_x=50, size_y=50, size_z=20)
    rng = np.random.default_rng(7)
    world = generate_layout(cfg, rng)
    
    # Ensure some solids exist
    assert np.count_nonzero(world.voxels > 0) > 0
    
    # Ensure ground layer is set (not AIR)
    assert np.all(world.voxels[0] != VoxelWorld.AIR)

def test_connectivity_integration_in_env():
    # This tests the integration via Env, which applies the Skeleton + Validator
    # We mock the validator to ensure it's called? Or just run it.
    # Just run it. Smoke test logic.
    from echelon.env.env import EchelonEnv, EnvConfig
    
    env_cfg = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, ensure_connectivity=True),
        seed=123
    )
    env = EchelonEnv(env_cfg)
    env.reset()
    
    world = env.world
    assert world.meta.get("generator") == "layout_v2"
    # Fixups should be a list (empty or populated)
    assert isinstance(world.meta.get("fixups"), list)
    # Corridors should be carved
    assert "corridors" in world.meta