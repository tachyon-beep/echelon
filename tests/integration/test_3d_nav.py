import numpy as np

from echelon.gen.biomes import fill_arcology_spire
from echelon.nav.graph import NavGraph
from echelon.nav.planner import Planner
from echelon.sim.world import VoxelWorld


def test_arcology_3d_connectivity():
    # 50x50 world, plenty of room for Arcology
    voxels = np.zeros((20, 50, 50), dtype=np.uint8)
    world = VoxelWorld(voxels=voxels, voxel_size_m=1.0)
    world.ensure_ground_layer()

    rng = np.random.default_rng(42)
    # Manually fill the whole world with Arcology biome
    fill_arcology_spire(world, 0, 0, 50, 50, rng)

    # Build NavGraph
    graph = NavGraph.build(world, clearance_z=3, mech_radius=0)
    planner = Planner(graph)

    # Identify nodes at different Z levels
    z_levels = set()
    for nid in graph.nodes:
        z_levels.add(nid[0])

    print(f"Nodes found at Z levels: {sorted(z_levels)}")

    # We expect nodes at -1 (ground), and likely 5, 10, 15 (Arcology stacks)
    assert -1 in z_levels
    # Check if we got any elevated nodes
    elevated = [z for z in z_levels if z > 0]
    assert len(elevated) > 0, "No elevated platforms were generated/detected"

    # Pick a ground node and an elevated node
    ground_node = next(nid for nid in graph.nodes if nid[0] == -1)
    high_node = next(nid for nid in graph.nodes if nid[0] == max(elevated))

    print(f"Attempting path from {ground_node} to {high_node}")

    path, stats = planner.find_path(ground_node, high_node)

    if stats.found:
        print(f"Success! Found path of length {stats.length}")
        # Verify Z variation in path
        path_zs = [nid[0] for nid in path]
        assert len(set(path_zs)) > 1, "Path exists but is purely flat?"
    else:
        # If it fails, it might just be because the random worm didn't link the stacks.
        # But Arcology should generate some ramps.
        # Let's try multiple pairs if the first fails.
        found_any_3d = False
        for high_nid in [nid for nid in graph.nodes if nid[0] > 0]:
            path, stats = planner.find_path(ground_node, high_nid)
            if stats.found:
                found_any_3d = True
                break

        assert found_any_3d, "Could not find any path from ground to an elevated platform in Arcology"


if __name__ == "__main__":
    test_arcology_3d_connectivity()
