"""Tests for NavGraph building and correctness."""
from __future__ import annotations

import numpy as np
import pytest

from echelon.sim.world import VoxelWorld
from echelon.nav.graph import NavGraph


@pytest.fixture
def flat_world() -> VoxelWorld:
    """10x10x5 world with solid floor at z=0."""
    voxels = np.zeros((5, 10, 10), dtype=np.uint8)
    voxels[0, :, :] = VoxelWorld.SOLID  # Floor
    return VoxelWorld(voxels=voxels)


@pytest.fixture
def platform_world() -> VoxelWorld:
    """World with floor and elevated platform."""
    voxels = np.zeros((10, 20, 20), dtype=np.uint8)
    voxels[0, :, :] = VoxelWorld.SOLID  # Floor
    voxels[3, 5:15, 5:15] = VoxelWorld.SOLID  # Platform at z=3
    return VoxelWorld(voxels=voxels)


class TestNavGraphBuild:
    """Tests for NavGraph.build() correctness."""

    def test_flat_world_nodes(self, flat_world: VoxelWorld):
        """Flat world should have nodes at ground level."""
        graph = NavGraph.build(flat_world, clearance_z=2, mech_radius=0)

        # Should have nodes for most ground positions
        # (edges may exclude some due to boundary)
        assert len(graph.nodes) > 0

        # All nodes should be at z=0 (standing on floor)
        for nid in graph.nodes:
            z, y, x = nid
            assert z == 0, f"Node {nid} not at ground level"

    def test_platform_world_has_two_levels(self, platform_world: VoxelWorld):
        """Platform world should have nodes at both levels."""
        graph = NavGraph.build(platform_world, clearance_z=2, mech_radius=0)

        z_levels = set(nid[0] for nid in graph.nodes)

        # Should have ground (z=0) and platform (z=3) nodes
        assert 0 in z_levels, "Missing ground level nodes"
        assert 3 in z_levels, "Missing platform level nodes"

    def test_edges_connect_neighbors(self, flat_world: VoxelWorld):
        """Adjacent nodes should have edges between them."""
        graph = NavGraph.build(flat_world, clearance_z=2, mech_radius=0)

        # Pick a center node
        center = (0, 5, 5)
        if center in graph.nodes:
            node = graph.nodes[center]
            neighbor_ids = [e[0] for e in node.edges]

            # Should have edges to orthogonal neighbors
            expected_neighbors = [(0, 5, 4), (0, 5, 6), (0, 4, 5), (0, 6, 5)]
            for expected in expected_neighbors:
                if expected in graph.nodes:
                    assert expected in neighbor_ids, f"Missing edge to {expected}"

    def test_mech_radius_reduces_nodes(self, flat_world: VoxelWorld):
        """Larger mech radius should produce fewer walkable nodes."""
        graph_r0 = NavGraph.build(flat_world, clearance_z=2, mech_radius=0)
        graph_r1 = NavGraph.build(flat_world, clearance_z=2, mech_radius=1)

        # Radius 1 excludes nodes near edges
        assert len(graph_r1.nodes) < len(graph_r0.nodes)

    def test_clearance_blocks_low_ceiling(self):
        """Nodes under low ceiling should be excluded."""
        voxels = np.zeros((5, 10, 10), dtype=np.uint8)
        voxels[0, :, :] = VoxelWorld.SOLID  # Floor
        voxels[2, :, 5:] = VoxelWorld.SOLID  # Low ceiling on right half
        world = VoxelWorld(voxels=voxels)

        graph = NavGraph.build(world, clearance_z=3, mech_radius=0)

        # Nodes under ceiling should be excluded
        for nid in graph.nodes:
            z, y, x = nid
            if z == 0 and x >= 5:
                pytest.fail(f"Node {nid} should be blocked by low ceiling")
