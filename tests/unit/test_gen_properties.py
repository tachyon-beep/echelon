import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from echelon.config import WorldConfig
from echelon.gen.layout import generate_layout
from echelon.sim.world import VoxelWorld


@settings(deadline=500, max_examples=20)
@given(
    seed=st.integers(min_value=0, max_value=1000000),
    size_x=st.integers(min_value=30, max_value=100),
    size_y=st.integers(min_value=30, max_value=100),
)
def test_prop_layout_invariants(seed, size_x, size_y):
    # Constraint: ensure even sizes or handle odds (world handles it, but let's stick to sane inputs)
    cfg = WorldConfig(size_x=size_x, size_y=size_y, size_z=15)
    rng = np.random.default_rng(seed)

    world = generate_layout(cfg, rng)

    # Invariant 1: Metadata consistency
    assert world.meta["generator"] == "layout_v2"
    layout = world.meta["biome_layout"]
    assert len(layout) == 4

    # Invariant 2: Ground layer exists
    # Layer 0 must be non-AIR (floor)
    assert np.all(world.voxels[0, :, :] != VoxelWorld.AIR)

    # Invariant 3: Valid Bounds
    assert world.voxels.shape == (15, size_y, size_x)


@settings(deadline=1000, max_examples=10)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_prop_connectivity_validator_completes(seed):
    # Test that the validator doesn't crash or hang on random noise
    # We construct a noisy world and run the validator
    from echelon.gen.validator import ConnectivityValidator

    size = 30
    rng = np.random.default_rng(seed)

    # Create random noise world
    voxels = np.zeros((10, size, size), dtype=np.uint8)
    voxels[0] = 6  # Dirt

    # Random obstacles
    noise = rng.random((9, size, size)) < 0.2
    voxels[1:] = np.where(noise, 1, 0).astype(np.uint8)

    validator = ConnectivityValidator((10, size, size), clearance_z=3, carve_width=3)

    meta = {"capture_zone": {"center": [size / 2.0, size / 2.0], "radius": 5.0}}
    spawn_corners = {"blue": "BL", "red": "TR"}

    # Should not raise
    fixed_voxels = validator.validate_and_fix(voxels, spawn_corners=spawn_corners, spawn_clear=5, meta=meta)

    assert fixed_voxels.shape == voxels.shape
    assert "fixups" in meta
    # Fixups log should be list of strings
    assert all(isinstance(f, str) for f in meta["fixups"])
