"""Core simulation invariants that must never break.

These tests verify fundamental constraints of the simulation:
- Heat is always non-negative
- Stability is bounded [0, max_stability]
- Dead mechs are immobile
- Positions stay within world bounds
"""

import numpy as np

from echelon import EchelonEnv, EnvConfig
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


class TestDeadMechInvariants:
    """Dead mech behavior invariants."""

    def test_dead_mechs_stay_dead(self, make_mech):
        """Once dead, a mech stays dead for the episode."""
        world = VoxelWorld.generate(WorldConfig(size_x=20, size_y=20, size_z=10), np.random.default_rng(0))
        world.voxels.fill(VoxelWorld.AIR)
        sim = Sim(world, dt_sim=0.1, rng=np.random.default_rng(0))

        mech = make_mech("m", "blue", [5.0, 5.0, 1.0], "heavy")
        mech.hp = 0.0
        mech.alive = False
        sim.reset({"m": mech})

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(10):
            sim.step({"m": action}, num_substeps=1)
            assert not mech.alive, "Dead mech should stay dead"

    def test_dead_mechs_have_zero_velocity(self, make_mech):
        """Dead mechs should have zero velocity."""
        world = VoxelWorld.generate(WorldConfig(size_x=20, size_y=20, size_z=10), np.random.default_rng(0))
        world.voxels.fill(VoxelWorld.AIR)
        sim = Sim(world, dt_sim=0.1, rng=np.random.default_rng(0))

        mech = make_mech("m", "blue", [5.0, 5.0, 1.0], "heavy")
        mech.hp = 0.0
        mech.alive = False
        mech.vel[:] = 0.0
        sim.reset({"m": mech})

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[0] = 1.0  # Try to move forward
        sim.step({"m": action}, num_substeps=1)

        # Dead mechs shouldn't move
        assert np.allclose(mech.vel, 0.0), f"Dead mech has non-zero velocity: {mech.vel}"


class TestPositionInvariants:
    """Position and world bounds invariants."""

    def test_positions_within_world_bounds(self):
        """All mech positions stay within world boundaries."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15, obstacle_fill=0.1),
            num_packs=1,
            seed=42,
            max_episode_seconds=10.0,
        )
        env = EchelonEnv(cfg)
        obs, _ = env.reset(seed=42)
        assert env.sim is not None
        assert env.world is not None

        # Run episode with random actions
        for _ in range(100):
            actions = {aid: np.random.uniform(-1, 1, env.ACTION_DIM).astype(np.float32) for aid in obs}
            obs, _, terms, truncs, _ = env.step(actions)

            # Check all mech positions
            for aid in env.agents:
                m = env.sim.mechs[aid]
                if m.alive:
                    assert 0 <= m.pos[0] <= env.world.size_x, f"{aid} x={m.pos[0]} out of bounds"
                    assert 0 <= m.pos[1] <= env.world.size_y, f"{aid} y={m.pos[1]} out of bounds"
                    assert 0 <= m.pos[2] <= env.world.size_z, f"{aid} z={m.pos[2]} out of bounds"

            if all(terms.values()) or all(truncs.values()):
                break

    def test_mech_doesnt_fall_through_floor(self, make_mech):
        """Mech doesn't fall through the floor (z >= 0)."""
        world = VoxelWorld.generate(WorldConfig(size_x=20, size_y=20, size_z=10), np.random.default_rng(0))
        world.voxels.fill(VoxelWorld.AIR)
        world.voxels[0, :, :] = VoxelWorld.SOLID  # Floor
        sim = Sim(world, dt_sim=0.1, rng=np.random.default_rng(0))

        mech = make_mech("m", "blue", [10.0, 10.0, 5.0], "heavy")
        sim.reset({"m": mech})

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(50):
            sim.step({"m": action}, num_substeps=1)
            assert mech.pos[2] >= 0.0, f"Mech fell through floor: z={mech.pos[2]}"


class TestDamageConservation:
    """Damage dealt must equal damage taken."""

    def test_laser_damage_conservation(self, make_mech):
        """Laser damage dealt equals damage taken by target."""
        world = VoxelWorld.generate(WorldConfig(size_x=30, size_y=30, size_z=10), np.random.default_rng(0))
        world.voxels.fill(VoxelWorld.AIR)
        sim = Sim(world, dt_sim=0.05, rng=np.random.default_rng(0))

        # Two mechs facing each other
        shooter = make_mech("shooter", "blue", [5.0, 15.0, 1.0], "heavy")
        shooter.yaw = 0.0  # Facing +x
        target = make_mech("target", "red", [15.0, 15.0, 1.0], "medium")
        sim.reset({"shooter": shooter, "target": target})

        initial_hp = float(target.hp)

        # Fire laser
        action_fire = np.zeros(ACTION_DIM, dtype=np.float32)
        action_fire[4] = 1.0  # PRIMARY (laser)
        action_noop = np.zeros(ACTION_DIM, dtype=np.float32)

        events = sim.step({"shooter": action_fire, "target": action_noop}, num_substeps=1)

        # Check damage conservation
        hp_lost = initial_hp - float(target.hp)

        # Find damage dealt in events
        damage_dealt = 0.0
        for ev in events:
            if ev.get("type") == "damage" and ev.get("target") == "target":
                damage_dealt += float(ev.get("amount", 0.0))

        if hp_lost > 0:
            assert (
                abs(hp_lost - damage_dealt) < 1e-3
            ), f"Damage mismatch: HP lost={hp_lost}, damage events={damage_dealt}"
