import numpy as np
import pytest
from echelon.sim.world import VoxelWorld
from echelon.nav.graph import NavGraph
from echelon.nav.planner import Planner

def test_nav_build_flat_ground():
    # 10x10 flat ground
    voxels = np.zeros((5, 10, 10), dtype=np.uint8)
    world = VoxelWorld(voxels=voxels, voxel_size_m=5.0)
    world.ensure_ground_layer() 
    # DIRT is non-colliding, so the "floor" is the bedrock at z=-1 (world z=0.0).
    # The DIRT layer at z=0 is just "fog" we stand in.
    
    graph = NavGraph.build(world)
    
    # Expect 100 nodes (one per column)
    assert len(graph.nodes) == 100
    
    # Check a center node
    # ID should be (-1, 5, 5) representing standing on bedrock.
    nid = (-1, 5, 5)
    assert nid in graph.nodes
    node = graph.nodes[nid]
    # Pos: (5.5*5, 5.5*5, (-1+1)*5) = (27.5, 27.5, 0.0)
    assert node.pos == (27.5, 27.5, 0.0)
    
    # Check edges (grid interior has 8 neighbors)
    assert len(node.edges) == 8

def test_nav_step_up():
    # 3x3 world. 
    voxels = np.zeros((5, 3, 3), dtype=np.uint8)
    world = VoxelWorld(voxels=voxels, voxel_size_m=1.0)
    world.ensure_ground_layer() # z=0 is DIRT (non-colliding)
    
    # To test a step up, we need a SOLID floor at z=0.
    # We overwrite the center voxel at z=0 with SOLID.
    # Now center has floor at z=0 (NavNode (0,1,1)).
    # Sides have floor at z=-1 (NavNode (-1,1,0)).
    # Delta z = 1.
    world.set_box_solid(1, 1, 0, 2, 2, 1, True) 
    
    graph = NavGraph.build(world)
    
    center = (0, 1, 1) # On top of voxel 0
    side = (-1, 1, 0) # On top of bedrock
    
    assert center in graph.nodes
    assert side in graph.nodes
    
    # Check connectivity side -> center (step up)
    side_node = graph.nodes[side]
    neighbors = [n[0] for n in side_node.edges]
    assert center in neighbors
    
    # Check connectivity center -> side (step down)
    center_node = graph.nodes[center]
    neighbors = [n[0] for n in center_node.edges]
    assert side in neighbors

def test_planner_simple_path():
    # 10x1 strip
    voxels = np.zeros((5, 1, 10), dtype=np.uint8)
    world = VoxelWorld(voxels=voxels, voxel_size_m=1.0)
    world.ensure_ground_layer()
    
    graph = NavGraph.build(world)
    planner = Planner(graph)
    
    start = (-1, 0, 0)
    goal = (-1, 0, 9)
    
    path, stats = planner.find_path(start, goal)
    
    assert stats.found
    assert len(path) == 10 # 0..9 inclusive
    assert path[0] == start
    assert path[-1] == goal

def test_planner_obstacle():
    # 10x3 strip with wall in middle
    voxels = np.zeros((5, 3, 10), dtype=np.uint8)
    world = VoxelWorld(voxels=voxels, voxel_size_m=1.0)
    world.ensure_ground_layer()
    
    # Block middle column (x=5)
    # We need to block the "air" above the floor.
    # Floor is z=-1. Headroom is z=0, z=1...
    # z=0 is DIRT (non-colliding). 
    # If we put SOLID at z=0, it becomes a step up (floor).
    # We need to block the path.
    # If we put SOLID at z=0, and SOLID at z=1... then it's a 2-voxel high wall.
    # Then climbing it requires stepping -1 -> 0 -> 1 -> 2.
    # Wait, if we put SOLID at z=0, it creates a Node at (0, y, x).
    # Step -1 -> 0 is valid.
    # We want to BLOCK.
    # So we need a wall that is TOO HIGH to step up.
    # Step limit is 1.
    # If we have floor at -1, and next floor is at 1 (voxel 1 solid), that's delta=2.
    # But we also need to ensure we can't walk *through* the wall.
    # If voxel 0 is solid, it's a floor.
    # If voxel 0 is AIR (or dirt) and voxel 1 is solid...
    # Then at x=5, we have air at 0, solid at 1.
    # Is (0, y, 5) a node? No, solid is at 1.
    # Is (-1, y, 5) a node? Yes, if 0 is air/dirt.
    # But if 1 is solid, is headroom clear?
    # NavGraph checks 0..clearance.
    # If clearance=2.
    # At (-1, y, 5): check 0 (dirt=ok), check 1 (solid=FAIL).
    # So (-1, y, 5) is NOT a node because of low ceiling!
    # Correct.
    
    # So placing a block at z=1 (leaving z=0 as dirt) should block the path at ground level.
    world.set_box_solid(5, 0, 1, 6, 3, 5, True) 
    
    graph = NavGraph.build(world)
    planner = Planner(graph)
    
    start = (-1, 1, 0)
    goal = (-1, 1, 9)
    
    path, stats = planner.find_path(start, goal)
    
    assert not stats.found
