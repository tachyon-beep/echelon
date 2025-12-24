import numpy as np
import pytest

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig
from echelon.sim.sim import Sim
from echelon.sim.world import VoxelWorld


def test_sim_determinism():
    """
    Verify that two environments initialized with the same seed produce
    identical observations and internal states when fed the same actions.
    """
    seed = 12345
    steps = 100

    cfg = EnvConfig(
        world=WorldConfig(size_x=20, size_y=20, size_z=10, obstacle_fill=0.1),
        num_packs=1,
        observation_mode="full",
        seed=seed,
        max_episode_seconds=10.0,
    )

    env1 = EchelonEnv(cfg)
    env2 = EchelonEnv(cfg)

    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)

    # Check initial observations
    for agent_id in env1.agents:
        np.testing.assert_array_equal(
            obs1[agent_id], obs2[agent_id], err_msg=f"Initial obs mismatch for {agent_id}"
        )

    # Check internal state (positions)
    for mid, m1 in env1.sim.mechs.items():
        m2 = env2.sim.mechs[mid]
        np.testing.assert_array_equal(m1.pos, m2.pos, err_msg=f"Initial pos mismatch for {mid}")

    rng = np.random.default_rng(seed)

    for step in range(steps):
        actions = {}
        for agent_id in env1.agents:
            # Generate random action (same for both envs since we use same rng sequence)
            act = rng.uniform(-1.0, 1.0, size=env1.ACTION_DIM).astype(np.float32)
            actions[agent_id] = act

        o1, r1, t1, _tr1, _i1 = env1.step(actions)
        o2, r2, t2, _tr2, _i2 = env2.step(actions)

        # Check observations
        for agent_id in env1.agents:
            np.testing.assert_array_equal(
                o1[agent_id], o2[agent_id], err_msg=f"Step {step}: Obs mismatch for {agent_id}"
            )
            assert r1[agent_id] == r2[agent_id], f"Step {step}: Reward mismatch for {agent_id}"
            assert t1[agent_id] == t2[agent_id], f"Step {step}: Termination mismatch for {agent_id}"

        # Check deep internal state
        for mid, m1 in env1.sim.mechs.items():
            m2 = env2.sim.mechs[mid]
            np.testing.assert_array_equal(m1.pos, m2.pos, err_msg=f"Step {step}: Pos mismatch for {mid}")
            np.testing.assert_array_equal(m1.vel, m2.vel, err_msg=f"Step {step}: Vel mismatch for {mid}")
            assert m1.hp == m2.hp, f"Step {step}: HP mismatch for {mid}"
            assert m1.heat == m2.heat, f"Step {step}: Heat mismatch for {mid}"

    print(f"Determinism passed for {steps} steps.")


def test_debris_deterministic_across_rng(make_mech):
    cfg = WorldConfig(size_x=10, size_y=10, size_z=6, obstacle_fill=0.0, ensure_connectivity=False)

    world1 = VoxelWorld.generate(cfg, np.random.default_rng(0))
    world2 = VoxelWorld.generate(cfg, np.random.default_rng(1))
    world1.voxels.fill(VoxelWorld.AIR)
    world2.voxels.fill(VoxelWorld.AIR)

    sim1 = Sim(world1, 0.05, np.random.default_rng(0))
    sim2 = Sim(world2, 0.05, np.random.default_rng(123))

    mech1 = make_mech("blue_0", "blue", [5.0, 5.0, 2.0], "heavy")
    mech2 = make_mech("blue_0", "blue", [5.0, 5.0, 2.0], "heavy")
    sim1.reset({"blue_0": mech1})
    sim2.reset({"blue_0": mech2})

    sim1._spawn_debris(mech1)
    sim2._spawn_debris(mech2)

    np.testing.assert_array_equal(world1.voxels, world2.voxels)


class TestDeterminismExtended:
    """Extended determinism verification."""

    @pytest.mark.parametrize("seed", [0, 42, 12345, 999999, 2**30])
    def test_determinism_multiple_seeds(self, seed: int) -> None:
        """Determinism holds for various seeds."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=seed,
            max_episode_seconds=5.0,
        )

        # Run 1
        env1 = EchelonEnv(cfg)
        obs1, _ = env1.reset(seed=seed)
        trajectory1 = [obs1["blue_0"].copy()]

        for _ in range(20):
            actions = {aid: np.zeros(env1.ACTION_DIM, dtype=np.float32) for aid in env1.agents}
            obs1, _, terms, truncs, _ = env1.step(actions)
            trajectory1.append(obs1["blue_0"].copy())
            if all(terms.values()) or all(truncs.values()):
                break

        # Run 2
        env2 = EchelonEnv(cfg)
        obs2, _ = env2.reset(seed=seed)
        trajectory2 = [obs2["blue_0"].copy()]

        for _ in range(20):
            actions = {aid: np.zeros(env2.ACTION_DIM, dtype=np.float32) for aid in env2.agents}
            obs2, _, terms, truncs, _ = env2.step(actions)
            trajectory2.append(obs2["blue_0"].copy())
            if all(terms.values()) or all(truncs.values()):
                break

        # Compare trajectories
        assert len(trajectory1) == len(trajectory2), "Trajectory lengths should match"
        for i, (o1, o2) in enumerate(zip(trajectory1, trajectory2, strict=True)):
            assert np.allclose(o1, o2), f"Observations differ at step {i}"

    def test_reset_restores_initial_state(self) -> None:
        """reset(seed=X) after steps equals fresh env(seed=X)."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=42,
            max_episode_seconds=10.0,
        )

        env = EchelonEnv(cfg)

        # First reset
        obs1, _ = env.reset(seed=42)
        initial_obs = obs1["blue_0"].copy()

        # Run some steps
        for _ in range(10):
            actions = {aid: np.random.uniform(-1, 1, env.ACTION_DIM).astype(np.float32) for aid in env.agents}
            env.step(actions)

        # Reset with same seed
        obs2, _ = env.reset(seed=42)

        # Should match initial state
        assert np.allclose(obs2["blue_0"], initial_obs), "Reset should restore initial state"

    def test_determinism_with_combat(self) -> None:
        """Combat outcomes are deterministic."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=10, obstacle_fill=0.0, ensure_connectivity=False),
            num_packs=1,
            seed=123,
            max_episode_seconds=5.0,
        )

        def run_combat() -> list[float]:
            env = EchelonEnv(cfg)
            env.reset(seed=123)
            assert env.sim is not None

            # Position for combat
            env.sim.mechs["blue_0"].pos[:] = [10, 15, 1]
            env.sim.mechs["blue_0"].yaw = 0.0
            env.sim.mechs["red_0"].pos[:] = [20, 15, 1]

            hp_history = []
            for _ in range(20):
                actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
                actions["blue_0"][4] = 1.0  # Fire laser
                env.step(actions)
                hp_history.append(float(env.sim.mechs["red_0"].hp))

            return hp_history

        hp1 = run_combat()
        hp2 = run_combat()

        assert hp1 == hp2, f"Combat outcomes should be deterministic: {hp1} vs {hp2}"
