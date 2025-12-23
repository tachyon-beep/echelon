import numpy as np

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig


def test_golden_replay_reproducibility():
    """
    Generate a replay, then re-run the same actions and verify the
    recorded events/states match exactly. This ensures our replay system
    is a faithful 'Golden' record.
    """
    seed = 999
    cfg = EnvConfig(
        world=WorldConfig(size_x=20, size_y=20, size_z=10),
        num_packs=1,
        seed=seed,
        record_replay=True,  # Enable recording
        max_episode_seconds=5.0,
    )
    env = EchelonEnv(cfg)
    env.reset(seed=seed)

    # 1. Run and Record
    action_log = []
    rng = np.random.default_rng(seed)

    for _ in range(10):
        actions = {}
        for aid in env.agents:
            actions[aid] = rng.uniform(-1.0, 1.0, size=env.ACTION_DIM).astype(np.float32)
        env.step(actions)
        action_log.append(actions)

    replay = env.get_replay()
    assert replay is not None
    assert len(replay["frames"]) == 10

    # 2. Re-run from seed and compare state to replay frames
    env2 = EchelonEnv(cfg)
    env2.reset(seed=seed)

    for i, actions in enumerate(action_log):
        env2.step(actions)

        # Compare current state of env2 with replay frame i
        frame = replay["frames"][i]

        # Check mechs
        for mid, m_state in frame["mechs"].items():
            current_mech = env2.sim.mechs[mid]

            # Position
            rec_pos = np.array(m_state["pos"], dtype=np.float32)
            np.testing.assert_allclose(
                current_mech.pos, rec_pos, rtol=1e-5, err_msg=f"Frame {i}: Pos mismatch for {mid}"
            )

            # HP
            assert abs(current_mech.hp - m_state["hp"]) < 1e-5, f"Frame {i}: HP mismatch"

    print("Golden replay reproducibility passed.")
