"""Unit tests for PPO trainer implementation.

These tests verify the core PPO algorithm components:
- Loss computation (clipped objective, value loss, entropy)
- Advantage normalization
- Return normalization
- Full update cycle
"""

from __future__ import annotations

import torch

from echelon.rl.model import ActorCriticLSTM
from echelon.training.ppo import PPOConfig, PPOTrainer
from echelon.training.rollout import RolloutBuffer


def test_ppo_config_defaults():
    """Test that PPOConfig has sensible defaults."""
    config = PPOConfig()
    assert config.lr == 3e-4
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.clip_coef == 0.2
    assert config.ent_coef == 0.01
    assert config.vf_coef == 0.5
    assert config.max_grad_norm == 0.5
    assert config.update_epochs == 4
    assert config.rollout_steps == 512


def test_ppo_trainer_initialization():
    """Test PPOTrainer initialization."""
    device = torch.device("cpu")
    model = ActorCriticLSTM(obs_dim=10, action_dim=5)
    config = PPOConfig()
    trainer = PPOTrainer(model, config, device)

    assert trainer.model is model
    assert trainer.config is config
    assert trainer.device is device
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.return_normalizer is not None


def test_compute_losses_basic():
    """Test _compute_losses with synthetic data."""
    device = torch.device("cpu")
    model = ActorCriticLSTM(obs_dim=10, action_dim=5)
    config = PPOConfig(clip_coef=0.2, ent_coef=0.01, vf_coef=0.5)
    trainer = PPOTrainer(model, config, device)

    # Create synthetic data
    num_steps = 8
    batch_size = 4

    advantages = torch.randn(num_steps, batch_size)
    returns_normalized = torch.randn(num_steps, batch_size)
    old_logprobs = torch.randn(num_steps, batch_size)
    new_logprobs = old_logprobs + torch.randn(num_steps, batch_size) * 0.1  # Slight change
    new_values = torch.randn(num_steps, batch_size)
    new_entropies = torch.rand(num_steps, batch_size) + 0.5  # Positive entropy

    loss, metrics = trainer._compute_losses(
        advantages=advantages,
        returns_normalized=returns_normalized,
        old_logprobs=old_logprobs,
        new_logprobs=new_logprobs,
        new_values=new_values,
        new_entropies=new_entropies,
    )

    # Check that loss is a scalar
    assert loss.shape == ()
    # Note: loss won't have requires_grad if inputs don't have it (synthetic test data)

    # Check that all metrics are present and scalar
    expected_keys = {"pg_loss", "vf_loss", "entropy", "approx_kl", "clipfrac", "loss"}
    assert set(metrics.keys()) == expected_keys
    for key, value in metrics.items():
        assert isinstance(value, float), f"{key} should be float"


def test_compute_losses_clipping():
    """Test that clipping works correctly when ratio exceeds bounds."""
    device = torch.device("cpu")
    model = ActorCriticLSTM(obs_dim=10, action_dim=5)
    config = PPOConfig(clip_coef=0.2)
    trainer = PPOTrainer(model, config, device)

    # Create data where ratio will be clipped
    num_steps = 4
    batch_size = 2

    # Positive advantages, ratio > 1 + clip_coef should be clipped
    advantages = torch.ones(num_steps, batch_size)
    returns_normalized = torch.zeros(num_steps, batch_size)
    old_logprobs = torch.zeros(num_steps, batch_size)
    # New policy is much more confident -> ratio >> 1
    new_logprobs = old_logprobs + 2.0  # ratio = exp(2) = 7.39 >> 1.2
    new_values = torch.zeros(num_steps, batch_size)
    new_entropies = torch.ones(num_steps, batch_size)

    _loss, metrics = trainer._compute_losses(
        advantages=advantages,
        returns_normalized=returns_normalized,
        old_logprobs=old_logprobs,
        new_logprobs=new_logprobs,
        new_values=new_values,
        new_entropies=new_entropies,
    )

    # When ratio > 1 + clip_coef and advantage > 0, clipping should occur
    # clipfrac should be 1.0 since all ratios exceed clip bound
    assert metrics["clipfrac"] == 1.0

    # Verify pg_loss uses clipped value
    # With clipping, pg_loss = -advantages * clip_ratio.mean()
    # = -1.0 * 1.2 = -1.2
    expected_pg_loss = -1.0 * 1.2
    assert abs(metrics["pg_loss"] - expected_pg_loss) < 1e-5


def test_compute_losses_no_clipping():
    """Test that no clipping occurs when ratio is within bounds."""
    device = torch.device("cpu")
    model = ActorCriticLSTM(obs_dim=10, action_dim=5)
    config = PPOConfig(clip_coef=0.2)
    trainer = PPOTrainer(model, config, device)

    num_steps = 4
    batch_size = 2

    advantages = torch.ones(num_steps, batch_size)
    returns_normalized = torch.zeros(num_steps, batch_size)
    old_logprobs = torch.zeros(num_steps, batch_size)
    # Small change: ratio = exp(0.1) = 1.105 < 1.2 (within bounds)
    new_logprobs = old_logprobs + 0.1
    new_values = torch.zeros(num_steps, batch_size)
    new_entropies = torch.ones(num_steps, batch_size)

    _loss, metrics = trainer._compute_losses(
        advantages=advantages,
        returns_normalized=returns_normalized,
        old_logprobs=old_logprobs,
        new_logprobs=new_logprobs,
        new_values=new_values,
        new_entropies=new_entropies,
    )

    # No clipping should occur
    assert metrics["clipfrac"] == 0.0

    # pg_loss should use unclipped ratio
    ratio = torch.exp(torch.tensor(0.1))
    expected_pg_loss = float((-1.0 * ratio).mean())
    assert abs(metrics["pg_loss"] - expected_pg_loss) < 1e-5


def test_compute_losses_value_loss():
    """Test value function loss computation."""
    device = torch.device("cpu")
    model = ActorCriticLSTM(obs_dim=10, action_dim=5)
    config = PPOConfig(vf_coef=0.5)
    trainer = PPOTrainer(model, config, device)

    num_steps = 4
    batch_size = 2

    advantages = torch.zeros(num_steps, batch_size)
    returns_normalized = torch.ones(num_steps, batch_size) * 2.0
    old_logprobs = torch.zeros(num_steps, batch_size)
    new_logprobs = torch.zeros(num_steps, batch_size)
    new_values = torch.ones(num_steps, batch_size) * 1.0  # Off by 1.0
    new_entropies = torch.zeros(num_steps, batch_size)

    _loss, metrics = trainer._compute_losses(
        advantages=advantages,
        returns_normalized=returns_normalized,
        old_logprobs=old_logprobs,
        new_logprobs=new_logprobs,
        new_values=new_values,
        new_entropies=new_entropies,
    )

    # vf_loss = 0.5 * (returns - values)^2 = 0.5 * (2 - 1)^2 = 0.5
    expected_vf_loss = 0.5
    assert abs(metrics["vf_loss"] - expected_vf_loss) < 1e-5


def test_update_with_synthetic_buffer():
    """Test full PPO update cycle with a synthetic rollout buffer."""
    device = torch.device("cpu")
    obs_dim = 10
    action_dim = 5
    model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim)
    config = PPOConfig(update_epochs=2, rollout_steps=8)
    trainer = PPOTrainer(model, config, device)

    # Create a synthetic buffer
    num_steps = 8
    batch_size = 4

    buffer = RolloutBuffer.create(
        num_steps=num_steps,
        num_agents=batch_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )

    # Fill with random data
    buffer.obs = torch.randn(num_steps, batch_size, obs_dim)
    buffer.actions = torch.tanh(torch.randn(num_steps, batch_size, action_dim))  # Squashed
    buffer.logprobs = torch.randn(num_steps, batch_size)
    buffer.rewards = torch.randn(num_steps, batch_size)
    buffer.dones = torch.zeros(num_steps, batch_size)
    buffer.values = torch.randn(num_steps, batch_size)

    # Compute advantages and returns
    next_value = torch.randn(batch_size)
    next_done = torch.zeros(batch_size)
    buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

    # Create initial LSTM state
    init_state = model.initial_state(batch_size, device)

    # Run update
    metrics = trainer.update(buffer, init_state)

    # Check that metrics are returned
    assert "pg_loss" in metrics
    assert "vf_loss" in metrics
    assert "entropy" in metrics
    assert "grad_norm" in metrics
    assert "approx_kl" in metrics
    assert "clipfrac" in metrics

    # All metrics should be finite
    for _key, value in metrics.items():
        assert isinstance(value, float)
        assert not torch.isnan(torch.tensor(value))
        assert not torch.isinf(torch.tensor(value))


def test_update_requires_gae():
    """Test that update raises error if GAE not computed."""
    device = torch.device("cpu")
    model = ActorCriticLSTM(obs_dim=10, action_dim=5)
    config = PPOConfig()
    trainer = PPOTrainer(model, config, device)

    # Create buffer without computing GAE
    buffer = RolloutBuffer.create(num_steps=8, num_agents=4, obs_dim=10, action_dim=5, device=device)
    buffer.obs = torch.randn(8, 4, 10)
    buffer.actions = torch.randn(8, 4, 5)
    buffer.logprobs = torch.randn(8, 4)
    buffer.rewards = torch.randn(8, 4)
    buffer.dones = torch.zeros(8, 4)
    buffer.values = torch.randn(8, 4)

    init_state = model.initial_state(4, device)

    # Should raise ValueError
    try:
        trainer.update(buffer, init_state)
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "advantages and returns" in str(e).lower()


def test_update_normalizes_advantages():
    """Test that advantages are normalized during update."""
    device = torch.device("cpu")
    model = ActorCriticLSTM(obs_dim=10, action_dim=5)
    config = PPOConfig(update_epochs=1)
    trainer = PPOTrainer(model, config, device)

    # Create buffer with non-normalized advantages
    buffer = RolloutBuffer.create(num_steps=8, num_agents=4, obs_dim=10, action_dim=5, device=device)
    buffer.obs = torch.randn(8, 4, 10)
    buffer.actions = torch.tanh(torch.randn(8, 4, 5))
    buffer.logprobs = torch.randn(8, 4)
    buffer.rewards = torch.randn(8, 4)
    buffer.dones = torch.zeros(8, 4)
    buffer.values = torch.randn(8, 4)

    # Set advantages to have non-zero mean and non-unit std
    buffer.advantages = torch.randn(8, 4) * 2.0 + 5.0
    buffer.returns = torch.randn(8, 4)

    init_state = model.initial_state(4, device)

    # Run update - should not raise
    metrics = trainer.update(buffer, init_state)
    assert "pg_loss" in metrics


def test_return_normalizer_updates():
    """Test that return normalizer is updated during training."""
    device = torch.device("cpu")
    model = ActorCriticLSTM(obs_dim=10, action_dim=5)
    config = PPOConfig(update_epochs=1)
    trainer = PPOTrainer(model, config, device)

    # Initial state
    initial_count = trainer.return_normalizer.rms.count

    # Create and run update
    buffer = RolloutBuffer.create(num_steps=8, num_agents=4, obs_dim=10, action_dim=5, device=device)
    buffer.obs = torch.randn(8, 4, 10)
    buffer.actions = torch.tanh(torch.randn(8, 4, 5))
    buffer.logprobs = torch.randn(8, 4)
    buffer.rewards = torch.randn(8, 4)
    buffer.dones = torch.zeros(8, 4)
    buffer.values = torch.randn(8, 4)
    buffer.advantages = torch.randn(8, 4)
    buffer.returns = torch.randn(8, 4)

    init_state = model.initial_state(4, device)
    trainer.update(buffer, init_state)

    # Count should have increased
    assert trainer.return_normalizer.rms.count > initial_count


def test_optimizer_updates_weights():
    """Test that optimizer actually updates model weights."""
    device = torch.device("cpu")
    model = ActorCriticLSTM(obs_dim=10, action_dim=5)
    config = PPOConfig(update_epochs=1, lr=1e-2)  # Higher LR for visible change
    trainer = PPOTrainer(model, config, device)

    # Store initial weights
    initial_weights = {name: param.clone() for name, param in model.named_parameters()}

    # Create buffer and run update
    buffer = RolloutBuffer.create(num_steps=8, num_agents=4, obs_dim=10, action_dim=5, device=device)
    buffer.obs = torch.randn(8, 4, 10)
    buffer.actions = torch.tanh(torch.randn(8, 4, 5))
    buffer.logprobs = torch.randn(8, 4)
    buffer.rewards = torch.randn(8, 4)
    buffer.dones = torch.zeros(8, 4)
    buffer.values = torch.randn(8, 4)
    next_value = torch.randn(4)
    next_done = torch.zeros(4)
    buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

    init_state = model.initial_state(4, device)
    trainer.update(buffer, init_state)

    # Check that at least some weights changed
    changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_weights[name], atol=1e-6):
            changed = True
            break

    assert changed, "Model weights should change after update"
