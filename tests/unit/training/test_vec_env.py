"""Unit tests for VectorEnv."""

import pytest

from echelon import EchelonEnv, EnvConfig, WorldConfig
from echelon.training.vec_env import VectorEnv


@pytest.fixture
def small_env_cfg() -> EnvConfig:
    """Create a minimal environment config for fast tests."""
    return EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=20),
        num_packs=1,
        dt_sim=0.05,
        decision_repeat=5,
        max_episode_seconds=10.0,  # Short episodes
        observation_mode="full",
        seed=0,
        record_replay=False,
    )


def _get_action_dim(cfg: EnvConfig) -> int:
    """Get action dimension from a temporary environment."""
    env = EchelonEnv(cfg)
    return env.ACTION_DIM


def test_vec_env_single_env(small_env_cfg: EnvConfig):
    """Test VectorEnv with single environment (fast unit test)."""
    vec_env = VectorEnv(num_envs=1, env_cfg=small_env_cfg)

    try:
        # Reset
        obs_list, info_list = vec_env.reset(seeds=[42])
        assert len(obs_list) == 1
        assert len(info_list) == 1
        assert isinstance(obs_list[0], dict)
        assert len(obs_list[0]) > 0  # Should have observations for agents

        # Get agent IDs from first observation
        agent_ids = list(obs_list[0].keys())
        assert len(agent_ids) > 0

        # Create dummy actions (zeros) - use action dimension from config
        action_dim = _get_action_dim(small_env_cfg)
        actions = {aid: [0.0] * action_dim for aid in agent_ids}

        # Step
        obs_list, rew_list, term_list, trunc_list, info_list = vec_env.step([actions])
        assert len(obs_list) == 1
        assert len(rew_list) == 1
        assert len(term_list) == 1
        assert len(trunc_list) == 1
        assert len(info_list) == 1

        # Check rewards dict
        assert isinstance(rew_list[0], dict)
        assert len(rew_list[0]) == len(agent_ids)

        # Check terminations/truncations
        assert isinstance(term_list[0], dict)
        assert isinstance(trunc_list[0], dict)

    finally:
        vec_env.close()


def test_vec_env_context_manager(small_env_cfg: EnvConfig):
    """Test VectorEnv context manager ensures cleanup."""
    with VectorEnv(num_envs=1, env_cfg=small_env_cfg) as vec_env:
        obs_list, _ = vec_env.reset(seeds=[42])
        assert len(obs_list) == 1

    # After context exit, processes should be cleaned up
    # Check that all processes have terminated
    for p in vec_env.ps:
        assert not p.is_alive()


def test_vec_env_reset_protocol(small_env_cfg: EnvConfig):
    """Test reset with different seeds and indices."""
    with VectorEnv(num_envs=2, env_cfg=small_env_cfg) as vec_env:
        # Reset all envs
        obs_list, info_list = vec_env.reset(seeds=[42, 43])
        assert len(obs_list) == 2
        assert len(info_list) == 2

        # Reset specific env
        obs_list2, info_list2 = vec_env.reset(seeds=[100], indices=[0])
        assert len(obs_list2) == 1
        assert len(info_list2) == 1


def test_vec_env_team_alive_protocol(small_env_cfg: EnvConfig):
    """Test get_team_alive protocol."""
    with VectorEnv(num_envs=1, env_cfg=small_env_cfg) as vec_env:
        vec_env.reset(seeds=[42])

        # Get team alive status
        team_alive_list = vec_env.get_team_alive()
        assert len(team_alive_list) == 1
        assert "blue" in team_alive_list[0]
        assert "red" in team_alive_list[0]
        assert isinstance(team_alive_list[0]["blue"], bool)
        assert isinstance(team_alive_list[0]["red"], bool)


def test_vec_env_last_outcomes_protocol(small_env_cfg: EnvConfig):
    """Test get_last_outcomes protocol."""
    with VectorEnv(num_envs=1, env_cfg=small_env_cfg) as vec_env:
        vec_env.reset(seeds=[42])

        # Get last outcomes (should be None after reset)
        outcomes = vec_env.get_last_outcomes()
        assert len(outcomes) == 1
        # Outcome is None or a dict depending on episode state


def test_vec_env_heuristic_actions_protocol(small_env_cfg: EnvConfig):
    """Test get_heuristic_actions protocol."""
    with VectorEnv(num_envs=1, env_cfg=small_env_cfg) as vec_env:
        obs_list, _ = vec_env.reset(seeds=[42])

        # Identify red agents (assume second half of agents are red)
        all_ids = list(obs_list[0].keys())
        red_ids = all_ids[len(all_ids) // 2 :]  # Second half

        # Get heuristic actions for red team
        heuristic_actions_list = vec_env.get_heuristic_actions(red_ids)
        assert len(heuristic_actions_list) == 1
        assert isinstance(heuristic_actions_list[0], dict)
        assert len(heuristic_actions_list[0]) == len(red_ids)


def test_vec_env_multi_step_rollout(small_env_cfg: EnvConfig):
    """Integration test: collect short rollout."""
    action_dim = _get_action_dim(small_env_cfg)
    with VectorEnv(num_envs=1, env_cfg=small_env_cfg) as vec_env:
        obs_list, _ = vec_env.reset(seeds=[42])

        agent_ids = list(obs_list[0].keys())
        rollout_steps = 10

        for _ in range(rollout_steps):
            # Create dummy actions using actual action dimension
            actions = {aid: [0.0] * action_dim for aid in agent_ids}

            # Step
            obs_list, _rew_list, term_list, trunc_list, _ = vec_env.step([actions])

            # Check episode termination
            if any(term_list[0].values()) or any(trunc_list[0].values()):
                # Episode ended, reset
                obs_list, _ = vec_env.reset(seeds=[43])
                break

        # Should complete without errors
        assert len(obs_list) == 1


def test_vec_env_curriculum_parameters(small_env_cfg: EnvConfig):
    """Test that curriculum parameters can be dynamically updated."""
    vec_env = VectorEnv(num_envs=1, env_cfg=small_env_cfg)
    try:
        # Set curriculum parameters
        vec_env.set_curriculum(
            weapon_prob=0.3,
            map_size_range=(40, 60),
        )

        # Verify it doesn't crash
        pass
    finally:
        vec_env.close()


def test_vec_env_get_curriculum(small_env_cfg: EnvConfig):
    """Test getting current curriculum state."""
    vec_env = VectorEnv(num_envs=1, env_cfg=small_env_cfg)
    try:
        curriculum = vec_env.get_curriculum()

        assert "weapon_prob" in curriculum
        assert "map_size_range" in curriculum
    finally:
        vec_env.close()


def test_vec_env_set_curriculum_updates_values(small_env_cfg: EnvConfig):
    """Test that set_curriculum actually updates values."""
    vec_env = VectorEnv(num_envs=1, env_cfg=small_env_cfg)
    try:
        # Set new values
        vec_env.set_curriculum(
            weapon_prob=0.75,
            map_size_range=(30, 50),
        )

        # Get updated values
        updated = vec_env.get_curriculum()

        assert updated["weapon_prob"] == 0.75
        assert updated["map_size_range"] == (30, 50)
    finally:
        vec_env.close()


def test_vec_env_curriculum_partial_update(small_env_cfg: EnvConfig):
    """Test that set_curriculum can update only specific parameters."""
    vec_env = VectorEnv(num_envs=1, env_cfg=small_env_cfg)
    try:
        # Set initial values
        vec_env.set_curriculum(
            weapon_prob=0.5,
            map_size_range=(40, 60),
        )

        # Update only weapon_prob
        vec_env.set_curriculum(weapon_prob=0.8)

        curriculum = vec_env.get_curriculum()
        assert curriculum["weapon_prob"] == 0.8
        assert curriculum["map_size_range"] == (40, 60)  # Should be unchanged

        # Update only map_size_range
        vec_env.set_curriculum(map_size_range=(50, 70))

        curriculum = vec_env.get_curriculum()
        assert curriculum["weapon_prob"] == 0.8  # Should be unchanged
        assert curriculum["map_size_range"] == (50, 70)
    finally:
        vec_env.close()
