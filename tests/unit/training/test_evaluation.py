"""Tests for evaluation module."""

import pytest
import torch

from echelon import EchelonEnv, EnvConfig, WorldConfig
from echelon.rl.model import ActorCriticLSTM
from echelon.training.evaluation import EvalStats, evaluate_vs_heuristic


def test_eval_stats_dataclass() -> None:
    """Test EvalStats dataclass structure."""
    stats = EvalStats(
        win_rate=0.75,
        mean_hp_margin=150.0,
        mean_episode_length=120.5,
        episodes=10,
    )
    assert stats.win_rate == 0.75
    assert stats.mean_hp_margin == 150.0
    assert stats.mean_episode_length == 120.5
    assert stats.episodes == 10


def test_evaluate_vs_heuristic_short_episode() -> None:
    """Integration test: evaluate policy vs heuristic with short episodes."""
    # Small env for fast test
    env_cfg = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=20),
        num_packs=1,
        dt_sim=0.05,
        decision_repeat=5,
        max_episode_seconds=10.0,  # Short episode
        observation_mode="full",
        seed=42,
        record_replay=False,
    )

    # Get actual obs_dim from environment
    temp_env = EchelonEnv(env_cfg)
    action_dim = temp_env.ACTION_DIM
    obs, _ = temp_env.reset(seed=42)
    obs_dim = next(iter(obs.values())).shape[0]

    # Create a model
    model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)
    device = torch.device("cpu")

    # Run evaluation with 2 episodes
    episodes = 2
    seeds = [100, 101]

    stats, replay = evaluate_vs_heuristic(
        model=model,
        env_cfg=env_cfg,
        episodes=episodes,
        seeds=seeds,
        device=device,
        save_replay=False,
    )

    # Check stats
    assert isinstance(stats, EvalStats)
    assert stats.episodes == episodes
    assert 0.0 <= stats.win_rate <= 1.0
    assert isinstance(stats.mean_hp_margin, float)
    assert stats.mean_episode_length > 0
    assert replay is None  # save_replay=False


def test_evaluate_vs_heuristic_with_replay() -> None:
    """Test replay generation from evaluation."""
    env_cfg = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=20),
        num_packs=1,
        dt_sim=0.05,
        decision_repeat=5,
        max_episode_seconds=10.0,
        observation_mode="full",
        seed=42,
        record_replay=False,  # Will be overridden by evaluate_vs_heuristic
    )

    # Get actual obs_dim from environment
    temp_env = EchelonEnv(env_cfg)
    action_dim = temp_env.ACTION_DIM
    obs, _ = temp_env.reset(seed=42)
    obs_dim = next(iter(obs.values())).shape[0]

    model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)
    device = torch.device("cpu")

    episodes = 1
    seeds = [200]

    stats, replay = evaluate_vs_heuristic(
        model=model,
        env_cfg=env_cfg,
        episodes=episodes,
        seeds=seeds,
        device=device,
        save_replay=True,
    )

    # Check replay exists
    assert replay is not None
    assert isinstance(replay, dict)
    # Replay has "frames" at top level (each frame has events)
    assert "frames" in replay
    assert len(replay["frames"]) > 0
    assert stats.episodes == episodes


def test_evaluate_seeds_mismatch() -> None:
    """Test error when seeds length doesn't match episodes."""
    env_cfg = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=20),
        num_packs=1,
        seed=42,
    )

    # Get actual obs_dim from environment
    temp_env = EchelonEnv(env_cfg)
    action_dim = temp_env.ACTION_DIM
    obs, _ = temp_env.reset(seed=42)
    obs_dim = next(iter(obs.values())).shape[0]

    model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)
    device = torch.device("cpu")

    # Mismatch: 3 episodes but only 2 seeds
    with pytest.raises(ValueError, match=r"seeds length .* must match episodes"):
        evaluate_vs_heuristic(
            model=model,
            env_cfg=env_cfg,
            episodes=3,
            seeds=[100, 101],
            device=device,
            save_replay=False,
        )


def test_evaluate_multiple_episodes() -> None:
    """Test evaluation aggregates statistics across multiple episodes."""
    env_cfg = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=20),
        num_packs=1,
        dt_sim=0.05,
        decision_repeat=5,
        max_episode_seconds=10.0,
        observation_mode="full",
        seed=42,
    )

    # Get actual obs_dim from environment
    temp_env = EchelonEnv(env_cfg)
    action_dim = temp_env.ACTION_DIM
    obs, _ = temp_env.reset(seed=42)
    obs_dim = next(iter(obs.values())).shape[0]

    model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)
    device = torch.device("cpu")

    # Run with 3 different seeds
    episodes = 3
    seeds = [12345, 54321, 98765]

    stats, _ = evaluate_vs_heuristic(
        model=model,
        env_cfg=env_cfg,
        episodes=episodes,
        seeds=seeds,
        device=device,
        save_replay=False,
    )

    # Check stats aggregation
    assert stats.episodes == episodes
    assert 0.0 <= stats.win_rate <= 1.0
    assert isinstance(stats.mean_hp_margin, float)
    assert stats.mean_episode_length > 0
