import numpy as np
from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig

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
        np.testing.assert_array_equal(obs1[agent_id], obs2[agent_id], err_msg=f"Initial obs mismatch for {agent_id}")

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

        o1, r1, t1, tr1, i1 = env1.step(actions)
        o2, r2, t2, tr2, i2 = env2.step(actions)

        # Check observations
        for agent_id in env1.agents:
            np.testing.assert_array_equal(o1[agent_id], o2[agent_id], err_msg=f"Step {step}: Obs mismatch for {agent_id}")
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
