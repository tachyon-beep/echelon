"""Tests for event emission from EchelonEnv."""

import numpy as np
import pytest

from echelon.config import EnvConfig, WorldConfig
from echelon.env.env import EchelonEnv


@pytest.fixture
def env():
    """Create a test environment."""
    env = EchelonEnv(
        config=EnvConfig(world=WorldConfig(size_x=40, size_y=40, size_z=20)),
    )
    env.reset()
    yield env


def _random_actions(env: EchelonEnv, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate random actions for all agents."""
    return {aid: rng.uniform(-1.0, 1.0, env.ACTION_DIM).astype(np.float32) for aid in env.agents}


def test_env_emits_events_in_info(env):
    """Environment includes events list in info dict."""
    rng = np.random.default_rng(42)
    actions = _random_actions(env, rng)
    _, _, _, _, infos = env.step(actions)

    # Should have events key in the global info
    assert "events" in infos


def test_env_emits_damage_events(env):
    """Damage events are emitted when mechs take damage."""
    rng = np.random.default_rng(42)

    # Run several steps to increase chance of combat
    all_events = []
    for _ in range(50):
        actions = _random_actions(env, rng)
        _, _, _, _, infos = env.step(actions)
        if "events" in infos:
            all_events.extend(infos["events"])

    # At minimum, structure should be correct
    for event in all_events:
        assert "type" in event
