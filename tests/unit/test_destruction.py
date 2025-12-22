import numpy as np
from echelon.sim.world import VoxelWorld
from echelon.sim.los import has_los

def test_voxel_destruction():
    voxels = np.zeros((5, 10, 10), dtype=np.uint8)
    world = VoxelWorld(voxels=voxels)
    
    # Place a solid block at (5,5,1)
    world.set_box_solid(5, 5, 1, 6, 6, 2, VoxelWorld.SOLID)
    assert world.get_voxel(5, 5, 1) == VoxelWorld.SOLID
    assert world.voxel_hp[1, 5, 5] == 100
    
    # Apply partial damage
    destroyed = world.damage_voxel(5, 5, 1, 40)
    assert not destroyed
    assert world.voxel_hp[1, 5, 5] == 60
    assert world.get_voxel(5, 5, 1) == VoxelWorld.SOLID
    
    # Destroy it
    destroyed = world.damage_voxel(5, 5, 1, 70)
    assert destroyed
    assert world.get_voxel(5, 5, 1) == VoxelWorld.AIR
    assert world.voxel_hp[1, 5, 5] == 0

def test_translucent_material_los():
    voxels = np.zeros((5, 10, 10), dtype=np.uint8)
    world = VoxelWorld(voxels=voxels)
    
    # Place GLASS wall in middle
    world.set_box_solid(5, 0, 0, 6, 10, 5, VoxelWorld.GLASS)
    
    start = np.array([2.0, 5.0, 2.0])
    end = np.array([8.0, 5.0, 2.0])
    
    # Glass should NOT block LOS
    assert has_los(world, start, end)
    
    # SOLID should block LOS
    world.set_box_solid(5, 0, 0, 6, 10, 5, VoxelWorld.SOLID)
    assert not has_los(world, start, end)

def test_soft_material_collision():
    voxels = np.zeros((5, 10, 10), dtype=np.uint8)
    world = VoxelWorld(voxels=voxels)
    
    # FOLIAGE at (5,5,1)
    world.set_box_solid(5, 5, 1, 6, 6, 2, VoxelWorld.FOLIAGE)
    
    aabb_min = np.array([5.1, 5.1, 1.1])
    aabb_max = np.array([5.9, 5.9, 1.9])
    
    # FOLIAGE should NOT collide
    assert not world.aabb_collides(aabb_min, aabb_max)
    
    # SOLID should collide
    world.set_box_solid(5, 5, 1, 6, 6, 2, VoxelWorld.SOLID)
    assert world.aabb_collides(aabb_min, aabb_max)
