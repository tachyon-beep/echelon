import numpy as np

from echelon.gen.objective import clear_capture_zone
from echelon.sim.world import VoxelWorld


def test_world_solid_mask_matches_collides_flag():
    materials = [
        (VoxelWorld.AIR, False),
        (VoxelWorld.SOLID, True),
        (VoxelWorld.LAVA, False),
        (VoxelWorld.WATER, False),
        (VoxelWorld.HOT_DEBRIS, True),
        (VoxelWorld.KILLED_HULL, True),
        (VoxelWorld.DIRT, False),
        (VoxelWorld.GLASS, True),
        (VoxelWorld.REINFORCED, True),
        (VoxelWorld.FOLIAGE, False),
    ]

    voxels = np.asarray([v for v, _ in materials], dtype=np.uint8).reshape(1, 1, -1)
    world = VoxelWorld(voxels=voxels, voxel_size_m=1.0)

    expected = [b for _, b in materials]
    assert np.asarray(world.solid)[0, 0, :].tolist() == expected


def test_clear_capture_zone_clears_colliding_non_solid_materials():
    voxels = np.zeros((3, 10, 10), dtype=np.uint8)
    voxels[0, 5, 5] = VoxelWorld.GLASS
    voxels[1, 5, 5] = VoxelWorld.REINFORCED
    world = VoxelWorld(voxels=voxels, voxel_size_m=1.0)

    meta = {"capture_zone": {"center": [5.5, 5.5], "radius": 0.1}}
    cleared = clear_capture_zone(world, meta=meta)

    assert cleared == 2
    assert int(world.voxels[0, 5, 5]) == VoxelWorld.AIR
    assert int(world.voxels[1, 5, 5]) == VoxelWorld.AIR
