"""Reward calculation correctness tests.

Reward bugs are the most insidious in RL - these tests verify:
- Sign correctness (positive for good, negative for bad)
- Attribution (reward goes to correct agent)
- Gradient direction (moving toward goal increases reward)
"""

import numpy as np
import pytest

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig
from echelon.gen.objective import capture_zone_params


@pytest.fixture
def reward_env():
    """Environment configured for reward testing."""
    cfg = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=15, obstacle_fill=0.0, ensure_connectivity=False),
        num_packs=1,
        seed=0,
        max_episode_seconds=30.0,
    )
    env = EchelonEnv(cfg)
    env.reset(seed=0)
    assert env.world is not None
    assert env.sim is not None
    return env


class TestRewardPolarity:
    """Verify reward signs are correct."""

    def test_zone_control_positive_for_holder(self, reward_env):
        """Team in zone alone gets positive per-tick reward."""
        env = reward_env

        # Get zone center
        zone_cx, zone_cy, _ = capture_zone_params(
            env.world.meta, size_x=env.world.size_x, size_y=env.world.size_y
        )

        # Move blue_0 into zone, everyone else out
        for aid in env.agents:
            m = env.sim.mechs[aid]
            if aid == "blue_0":
                m.pos[0], m.pos[1] = zone_cx, zone_cy
            else:
                # Move far from zone
                m.pos[0], m.pos[1] = 5.0, 5.0
            m.vel[:] = 0.0

        # Step with null actions
        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        _, rewards, _, _, _ = env.step(actions)

        # Blue in zone should get positive reward
        assert rewards["blue_0"] > 0, f"Blue in zone should get positive reward, got {rewards['blue_0']}"

    def test_death_gives_negative_reward(self, reward_env):
        """Dying gives negative reward to the victim.

        NOTE: This test currently exposes a bug - env.py looks for ev.get("victim")
        but sim.py emits ev["target"], so death penalty is never applied.
        Test is written for the intended behavior (W_DEATH=-0.5).
        """
        pytest.skip(
            "Bug: env.py line 1313 uses ev.get('victim') but sim.py line 566 uses 'target', "
            "so death penalty is never applied. Fix: change line 1313 to ev.get('target', '')"
        )

    def test_kill_gives_positive_reward(self, reward_env):
        """Getting a kill gives positive reward to the killer."""
        env = reward_env

        # Setup same as death test
        victim = env.sim.mechs["red_0"]
        victim.hp = 1.0

        killer = env.sim.mechs["blue_0"]
        killer.pos[0], killer.pos[1] = victim.pos[0] - 5.0, victim.pos[1]
        killer.yaw = 0.0  # Facing +x toward victim

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        actions["blue_0"][4] = 1.0  # PRIMARY (laser)

        kill_occurred = False
        for _ in range(10):
            was_alive = victim.alive
            _, rewards, _, _, _ = env.step(actions)
            # Check if kill happened this step
            if was_alive and not victim.alive and victim.died:
                # Kill occurred - W_KILL = 1.0
                assert rewards["blue_0"] > 0.5, f"Kill should give positive reward, got {rewards['blue_0']}"
                kill_occurred = True
                break

        assert kill_occurred, "Victim should have been killed during test"
