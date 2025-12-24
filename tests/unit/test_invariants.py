"""Core simulation invariants that must never break.

These tests verify fundamental constraints of the simulation:
- Heat is always non-negative
- Stability is bounded [0, max_stability]
- Dead mechs are immobile
- Positions stay within world bounds
"""

import numpy as np

from echelon.actions import ACTION_DIM
from echelon.config import WorldConfig
from echelon.sim.sim import Sim
from echelon.sim.world import VoxelWorld


class TestHeatInvariants:
    """Heat system invariants that must never break."""

    def test_heat_non_negative_after_dissipation(self, make_mech):
        """Heat remains >= 0 after dissipation step."""
        world = VoxelWorld.generate(WorldConfig(size_x=20, size_y=20, size_z=10), np.random.default_rng(0))
        world.voxels.fill(VoxelWorld.AIR)
        sim = Sim(world, dt_sim=0.1, rng=np.random.default_rng(0))

        mech = make_mech("m", "blue", [5.0, 5.0, 1.0], "heavy")
        mech.heat = 5.0  # Low heat that will dissipate to near-zero
        sim.reset({"m": mech})

        # Step many times to ensure heat dissipates
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(100):
            sim.step({"m": action}, num_substeps=1)
            assert mech.heat >= 0.0, f"Heat went negative: {mech.heat}"

    def test_heat_non_negative_with_venting(self, make_mech):
        """Heat remains >= 0 even with aggressive venting."""
        world = VoxelWorld.generate(WorldConfig(size_x=20, size_y=20, size_z=10), np.random.default_rng(0))
        world.voxels.fill(VoxelWorld.AIR)
        sim = Sim(world, dt_sim=0.1, rng=np.random.default_rng(0))

        mech = make_mech("m", "blue", [5.0, 5.0, 1.0], "heavy")
        mech.heat = 10.0
        sim.reset({"m": mech})

        # Vent action
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[5] = 1.0  # VENT index
        for _ in range(50):
            sim.step({"m": action}, num_substeps=1)
            assert mech.heat >= 0.0, f"Heat went negative during venting: {mech.heat}"

    def test_shutdown_at_heat_capacity(self, make_mech):
        """Mech shuts down when heat exceeds capacity."""
        world = VoxelWorld.generate(WorldConfig(size_x=20, size_y=20, size_z=10), np.random.default_rng(0))
        world.voxels.fill(VoxelWorld.AIR)
        sim = Sim(world, dt_sim=0.1, rng=np.random.default_rng(0))

        mech = make_mech("m", "blue", [5.0, 5.0, 1.0], "heavy")
        mech.heat = mech.spec.heat_cap + 10.0  # Over capacity
        sim.reset({"m": mech})

        # After step, should be shutdown
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        sim.step({"m": action}, num_substeps=1)
        assert mech.shutdown, "Mech should be shutdown when heat > capacity"


class TestStabilityInvariants:
    """Stability system invariants."""

    def test_stability_bounded_above(self, make_mech):
        """Stability never exceeds max_stability."""
        world = VoxelWorld.generate(WorldConfig(size_x=20, size_y=20, size_z=10), np.random.default_rng(0))
        world.voxels.fill(VoxelWorld.AIR)
        sim = Sim(world, dt_sim=0.1, rng=np.random.default_rng(0))

        mech = make_mech("m", "blue", [5.0, 5.0, 1.0], "heavy")
        mech.stability = mech.max_stability  # At max
        sim.reset({"m": mech})

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(20):
            sim.step({"m": action}, num_substeps=1)
            assert (
                mech.stability <= mech.max_stability + 1e-6
            ), f"Stability {mech.stability} exceeded max {mech.max_stability}"

    def test_stability_bounded_below(self, make_mech):
        """Stability never goes negative."""
        world = VoxelWorld.generate(WorldConfig(size_x=20, size_y=20, size_z=10), np.random.default_rng(0))
        world.voxels.fill(VoxelWorld.AIR)
        sim = Sim(world, dt_sim=0.1, rng=np.random.default_rng(0))

        mech = make_mech("m", "blue", [5.0, 5.0, 1.0], "heavy")
        mech.stability = 0.0  # At minimum (fallen)
        sim.reset({"m": mech})

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(20):
            sim.step({"m": action}, num_substeps=1)
            assert mech.stability >= 0.0, f"Stability went negative: {mech.stability}"

    def test_fallen_mech_recovers_stability(self, make_mech):
        """Fallen mech (stability=0) eventually recovers partial stability."""
        world = VoxelWorld.generate(WorldConfig(size_x=20, size_y=20, size_z=10), np.random.default_rng(0))
        world.voxels.fill(VoxelWorld.AIR)
        sim = Sim(world, dt_sim=1.0, rng=np.random.default_rng(0))

        mech = make_mech("m", "blue", [5.0, 5.0, 1.0], "heavy")
        mech.stability = 0.0
        sim.reset({"m": mech})

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        # Wait for knockdown recovery (default ~3 seconds)
        for _ in range(5):
            sim.step({"m": action}, num_substeps=1)

        # After recovery, stability should be restored
        assert mech.stability > 0.0, "Mech should recover some stability after knockdown"
