"""Arena Match tests.

Tests for match execution:
- Observation stacking
- Config extraction from checkpoints
- Policy loading
- Match execution and outcomes
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from echelon.arena.match import (
    LoadedPolicy,
    MatchOutcome,
    _env_cfg_from_checkpoint,
    _stack_obs,
    load_policy,
    play_match,
)
from echelon.config import EnvConfig, WorldConfig
from echelon.env.env import EchelonEnv
from echelon.rl.model import ActorCriticLSTM


class TestStackObs:
    """Test observation stacking helper."""

    def test_stack_obs_single_agent(self):
        """_stack_obs works with single agent."""
        obs = {"a1": np.array([1.0, 2.0, 3.0])}
        stacked = _stack_obs(obs, ["a1"])

        assert stacked.shape == (1, 3)
        np.testing.assert_array_equal(stacked[0], [1.0, 2.0, 3.0])

    def test_stack_obs_multiple_agents(self):
        """_stack_obs stacks in order of ids."""
        obs = {
            "a1": np.array([1.0, 2.0]),
            "a2": np.array([3.0, 4.0]),
            "a3": np.array([5.0, 6.0]),
        }
        stacked = _stack_obs(obs, ["a3", "a1", "a2"])

        assert stacked.shape == (3, 2)
        np.testing.assert_array_equal(stacked[0], [5.0, 6.0])  # a3
        np.testing.assert_array_equal(stacked[1], [1.0, 2.0])  # a1
        np.testing.assert_array_equal(stacked[2], [3.0, 4.0])  # a2


class TestEnvCfgFromCheckpoint:
    """Test config extraction from checkpoint dict."""

    def test_extracts_basic_config(self):
        """_env_cfg_from_checkpoint extracts EnvConfig correctly."""
        world_cfg = WorldConfig(size_x=50, size_y=50, size_z=20)

        ckpt = {
            "env_cfg": {
                "world": asdict(world_cfg),
                "num_packs": 2,
                "max_episode_seconds": 120.0,
                "observation_mode": "partial",
                "dt_sim": 0.05,
                "decision_repeat": 5,
                "comm_dim": 8,
                "enable_target_selection": True,
                "enable_ewar": True,
                "enable_obs_control": True,
                "enable_comm": True,
                "nav_mode": "off",
                "record_replay": False,
                "seed": None,
            }
        }

        extracted = _env_cfg_from_checkpoint(ckpt)

        assert extracted.world.size_x == 50
        assert extracted.world.size_y == 50
        assert extracted.num_packs == 2
        assert extracted.max_episode_seconds == 120.0
        assert extracted.observation_mode == "partial"

    def test_handles_nested_world_config(self):
        """Correctly handles nested WorldConfig dict."""
        ckpt = {
            "env_cfg": {
                "world": {
                    "size_x": 100,
                    "size_y": 100,
                    "size_z": 30,
                    "voxel_size_m": 2.0,
                    "obstacle_fill": 0.1,
                    "ensure_connectivity": False,
                    "connectivity_clearance_z": 5,
                    "connectivity_obstacle_inflate_radius": 2,
                    "connectivity_wall_cost": 100.0,
                    "connectivity_penalty_radius": 4,
                    "connectivity_penalty_cost": 25.0,
                    "connectivity_carve_width": 6,
                },
                "num_packs": 1,
                "dt_sim": 0.05,
                "decision_repeat": 5,
                "max_episode_seconds": 60.0,
                "observation_mode": "full",
                "comm_dim": 8,
                "enable_target_selection": True,
                "enable_ewar": True,
                "enable_obs_control": True,
                "enable_comm": True,
                "nav_mode": "off",
                "record_replay": False,
                "seed": None,
            }
        }

        extracted = _env_cfg_from_checkpoint(ckpt)

        assert extracted.world.size_x == 100
        assert extracted.world.voxel_size_m == 2.0
        assert extracted.world.ensure_connectivity is False


class TestLoadedPolicy:
    """Test LoadedPolicy dataclass."""

    def test_loaded_policy_immutable(self):
        """LoadedPolicy is frozen dataclass."""
        model = ActorCriticLSTM(obs_dim=10, action_dim=9)
        env_cfg = EnvConfig()
        policy = LoadedPolicy(
            ckpt_path=Path("/test.pt"),
            env_cfg=env_cfg,
            model=model,
        )

        with pytest.raises(AttributeError):
            policy.ckpt_path = Path("/other.pt")  # type: ignore


class TestMatchOutcome:
    """Test MatchOutcome dataclass."""

    def test_match_outcome_immutable(self):
        """MatchOutcome is frozen dataclass."""
        outcome = MatchOutcome(
            winner="blue",
            hp={"blue": 500.0, "red": 0.0},
            seed=42,
        )

        with pytest.raises(AttributeError):
            outcome.winner = "red"  # type: ignore

    def test_match_outcome_fields(self):
        """MatchOutcome has correct fields."""
        outcome = MatchOutcome(
            winner="draw",
            hp={"blue": 100.0, "red": 100.0},
            seed=123,
        )

        assert outcome.winner == "draw"
        assert outcome.hp["blue"] == 100.0
        assert outcome.seed == 123


class TestLoadPolicy:
    """Test policy loading from checkpoints."""

    def test_load_policy_with_cached_dims(self):
        """load_policy uses cached obs_dim and action_dim."""
        # Create a model and save it
        obs_dim = 150
        action_dim = 9
        model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)

        env_cfg = EnvConfig(
            world=WorldConfig(size_x=20, size_y=20, size_z=10),
            num_packs=1,
        )

        ckpt = {
            "model_state": model.state_dict(),
            "env_cfg": {
                "world": asdict(env_cfg.world),
                "num_packs": 1,
                "dt_sim": 0.05,
                "decision_repeat": 5,
                "max_episode_seconds": 60.0,
                "observation_mode": "full",
                "comm_dim": 8,
                "enable_target_selection": True,
                "enable_ewar": True,
                "enable_obs_control": True,
                "enable_comm": True,
                "nav_mode": "off",
                "record_replay": False,
                "seed": None,
            },
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(ckpt, f.name)
            path = Path(f.name)

        try:
            device = torch.device("cpu")
            loaded = load_policy(path, device=device)

            assert loaded.ckpt_path == path.resolve()
            assert loaded.model.obs_dim == obs_dim
            assert loaded.model.action_dim == action_dim
            assert loaded.env_cfg.num_packs == 1
        finally:
            path.unlink()


class TestPlayMatch:
    """Test match execution."""

    @pytest.fixture
    def small_env_config(self):
        """Small environment config for fast tests."""
        return EnvConfig(
            world=WorldConfig(
                size_x=40,
                size_y=40,
                size_z=10,
                obstacle_fill=0.03,
                ensure_connectivity=True,
            ),
            num_packs=1,
            max_episode_seconds=5.0,  # Short matches
            observation_mode="full",
            seed=42,
        )

    @pytest.fixture
    def random_policy(self, small_env_config):
        """Create a random (untrained) policy for testing."""
        env = EchelonEnv(small_env_config)
        obs, _ = env.reset(seed=42)
        obs_dim = next(iter(obs.values())).shape[0]
        action_dim = env.ACTION_DIM

        model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)
        model.eval()
        return model

    def test_play_match_returns_outcome(self, small_env_config, random_policy):
        """play_match returns a valid MatchOutcome."""
        device = torch.device("cpu")

        outcome = play_match(
            env_cfg=small_env_config,
            blue_policy=random_policy,
            red_policy=random_policy,
            seed=123,
            device=device,
            max_steps=50,  # Limit steps for test speed
        )

        assert isinstance(outcome, MatchOutcome)
        assert outcome.winner in ("blue", "red", "draw")
        assert "blue" in outcome.hp
        assert "red" in outcome.hp
        assert outcome.seed == 123

    def test_play_match_deterministic(self, small_env_config):
        """play_match is deterministic for same seed and model state."""
        device = torch.device("cpu")

        # Create environment to get dimensions
        env = EchelonEnv(small_env_config)
        obs, _ = env.reset(seed=42)
        obs_dim = next(iter(obs.values())).shape[0]
        action_dim = env.ACTION_DIM

        # Create model with fixed seed for deterministic weights
        torch.manual_seed(12345)
        model1 = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)
        model1.eval()

        # Run first match
        outcome1 = play_match(
            env_cfg=small_env_config,
            blue_policy=model1,
            red_policy=model1,
            seed=999,
            device=device,
            max_steps=30,
        )

        # Create identical model with same seed
        torch.manual_seed(12345)
        model2 = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)
        model2.eval()

        # Run second match
        outcome2 = play_match(
            env_cfg=small_env_config,
            blue_policy=model2,
            red_policy=model2,
            seed=999,
            device=device,
            max_steps=30,
        )

        assert outcome1.winner == outcome2.winner
        assert outcome1.hp == outcome2.hp

    def test_play_match_different_seeds_vary(self, small_env_config, random_policy):
        """Different seeds produce different outcomes (usually)."""
        device = torch.device("cpu")

        outcomes = []
        for seed in range(10):
            outcome = play_match(
                env_cfg=small_env_config,
                blue_policy=random_policy,
                red_policy=random_policy,
                seed=seed,
                device=device,
                max_steps=100,  # Enough steps for random policies to deal some damage
            )
            outcomes.append(outcome)

        # Should have some variety in winners or HP
        hp_values = [o.hp["blue"] for o in outcomes]
        winners = [o.winner for o in outcomes]
        # Either HP varies or winners vary
        assert (
            len(set(hp_values)) > 1 or len(set(winners)) > 1
        ), "All matches had identical HP and winners - suspicious"

    def test_play_match_respects_max_steps(self, small_env_config, random_policy):
        """play_match terminates at max_steps."""
        device = torch.device("cpu")

        # Very short match
        outcome = play_match(
            env_cfg=small_env_config,
            blue_policy=random_policy,
            red_policy=random_policy,
            seed=42,
            device=device,
            max_steps=5,
        )

        # Should still return valid outcome
        assert outcome.winner in ("blue", "red", "draw")

    def test_play_match_hp_values_reasonable(self, small_env_config, random_policy):
        """HP values in outcome are reasonable."""
        device = torch.device("cpu")

        outcome = play_match(
            env_cfg=small_env_config,
            blue_policy=random_policy,
            red_policy=random_policy,
            seed=42,
            device=device,
            max_steps=100,
        )

        # HP should be non-negative
        assert outcome.hp["blue"] >= 0
        assert outcome.hp["red"] >= 0

    def test_play_match_with_different_policies(self, small_env_config):
        """play_match works with different policies for each team."""
        env = EchelonEnv(small_env_config)
        obs, _ = env.reset(seed=42)
        obs_dim = next(iter(obs.values())).shape[0]
        action_dim = env.ACTION_DIM

        # Create two different models
        torch.manual_seed(1)
        blue_model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)
        blue_model.eval()

        torch.manual_seed(2)
        red_model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)
        red_model.eval()

        device = torch.device("cpu")

        outcome = play_match(
            env_cfg=small_env_config,
            blue_policy=blue_model,
            red_policy=red_model,
            seed=42,
            device=device,
            max_steps=50,
        )

        assert isinstance(outcome, MatchOutcome)


class TestMatchIntegration:
    """Integration tests for the full match workflow."""

    def test_save_load_run_match(self):
        """Full workflow: create policy, save, load, run match."""
        # Create small config
        env_cfg = EnvConfig(
            world=WorldConfig(
                size_x=30,
                size_y=30,
                size_z=10,
                obstacle_fill=0.02,
            ),
            num_packs=1,
            max_episode_seconds=3.0,
            seed=42,
        )

        # Get dims from env
        env = EchelonEnv(env_cfg)
        obs, _ = env.reset(seed=42)
        obs_dim = next(iter(obs.values())).shape[0]
        action_dim = env.ACTION_DIM

        # Create and save a model
        model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)

        ckpt = {
            "model_state": model.state_dict(),
            "env_cfg": {
                "world": asdict(env_cfg.world),
                "num_packs": env_cfg.num_packs,
                "dt_sim": env_cfg.dt_sim,
                "decision_repeat": env_cfg.decision_repeat,
                "max_episode_seconds": env_cfg.max_episode_seconds,
                "observation_mode": env_cfg.observation_mode,
                "comm_dim": env_cfg.comm_dim,
                "enable_target_selection": env_cfg.enable_target_selection,
                "enable_ewar": env_cfg.enable_ewar,
                "enable_obs_control": env_cfg.enable_obs_control,
                "enable_comm": env_cfg.enable_comm,
                "nav_mode": env_cfg.nav_mode,
                "record_replay": env_cfg.record_replay,
                "seed": env_cfg.seed,
            },
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(ckpt, f.name)
            path = Path(f.name)

        try:
            # Load the policy
            device = torch.device("cpu")
            loaded = load_policy(path, device=device)

            # Run a match using the loaded policy
            outcome = play_match(
                env_cfg=loaded.env_cfg,
                blue_policy=loaded.model,
                red_policy=loaded.model,
                seed=123,
                device=device,
                max_steps=30,
            )

            assert outcome.winner in ("blue", "red", "draw")
        finally:
            path.unlink()
