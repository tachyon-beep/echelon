import numpy as np
import pytest

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig


def test_api_fuzzing_nan_inf():
    """
    Verify that the environment handles NaN and Inf values in actions
    without crashing or corrupting the simulation state.
    """
    cfg = EnvConfig(
        world=WorldConfig(size_x=20, size_y=20, size_z=10),
        num_packs=1,
        seed=42,
    )
    env = EchelonEnv(cfg)
    env.reset()

    # Create messy actions
    actions = {}
    for agent_id in env.agents:
        act = np.zeros(env.ACTION_DIM, dtype=np.float32)
        # NaN in movement
        act[0] = np.nan
        # Inf in rotation
        act[3] = np.inf
        # NegInf in fire
        act[4] = -np.inf
        actions[agent_id] = act

    # Step should not crash
    try:
        _obs, _rewards, _terminations, _truncations, _infos = env.step(actions)
    except Exception as e:
        pytest.fail(f"Env crashed on NaN/Inf inputs: {e}")

    # Verify state is still valid (not NaN)
    for mid, mech in env.sim.mechs.items():
        assert np.all(np.isfinite(mech.pos)), f"Mech {mid} pos became non-finite"
        assert np.all(np.isfinite(mech.vel)), f"Mech {mid} vel became non-finite"
        assert np.isfinite(mech.yaw), f"Mech {mid} yaw became non-finite"


def test_api_fuzzing_oob_actions():
    """
    Verify that out-of-bounds actions (e.g. > 1.0) are clipped or handled safely.
    """
    cfg = EnvConfig(
        world=WorldConfig(size_x=20, size_y=20, size_z=10),
        num_packs=1,
        seed=42,
    )
    env = EchelonEnv(cfg)
    env.reset()

    actions = {}
    for agent_id in env.agents:
        act = np.zeros(env.ACTION_DIM, dtype=np.float32)
        # Huge movement value
        act[0] = 1000.0
        actions[agent_id] = act

    _obs, _, _, _, _ = env.step(actions)

    # Mech should not have teleported 1000 units
    # Max speed is ~5-7 m/s. dt=0.1s. Max step ~0.7m.
    for mid, mech in env.sim.mechs.items():
        # Heuristic check: didn't explode
        assert np.linalg.norm(mech.vel) < 50.0, f"Mech {mid} velocity exploded: {mech.vel}"
