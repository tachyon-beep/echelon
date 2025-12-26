"""Unit tests for RolloutBuffer and GAE computation.

Tests trajectory storage, buffer pre-allocation, and Generalized Advantage Estimation
with various edge cases and multi-step scenarios.
"""

import torch

from echelon.training.rollout import RolloutBuffer


class TestRolloutBufferCreation:
    """Test RolloutBuffer.create() pre-allocation."""

    def test_create_allocates_correct_shapes(self):
        """Test create() allocates tensors with correct shapes."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=10, num_agents=5, obs_dim=32, action_dim=8, device=device)

        assert buffer.obs.shape == (10, 5, 32)
        assert buffer.actions.shape == (10, 5, 8)
        assert buffer.logprobs.shape == (10, 5)
        assert buffer.rewards.shape == (10, 5)
        assert buffer.dones.shape == (10, 5)
        assert buffer.values.shape == (10, 5)
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.shape == (10, 5)
        assert buffer.returns.shape == (10, 5)

    def test_preallocates_advantages_and_returns(self):
        """Verify advantages and returns are pre-allocated in create()."""
        buffer = RolloutBuffer.create(
            num_steps=10,
            num_agents=5,
            obs_dim=32,
            action_dim=9,
            device=torch.device("cpu"),
        )

        # Should be pre-allocated, not None
        assert buffer.advantages is not None
        assert buffer.returns is not None

        # Correct shape
        assert buffer.advantages.shape == (10, 5)
        assert buffer.returns.shape == (10, 5)

        # Should be zeros initially
        assert (buffer.advantages == 0).all()
        assert (buffer.returns == 0).all()

        # gae_computed should be False until compute_gae is called
        assert buffer.gae_computed is False

    def test_gae_computed_flag_set_after_compute_gae(self):
        """Verify gae_computed is True only after compute_gae is called."""
        buffer = RolloutBuffer.create(
            num_steps=5,
            num_agents=2,
            obs_dim=8,
            action_dim=4,
            device=torch.device("cpu"),
        )

        # Before compute_gae
        assert buffer.gae_computed is False

        # Fill with some data
        buffer.rewards[:] = 1.0
        buffer.values[:] = 0.5

        # Call compute_gae
        next_value = torch.zeros(2)
        next_done = torch.zeros(2)
        buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

        # After compute_gae
        assert buffer.gae_computed is True

    def test_create_allocates_zeros(self):
        """Test create() initializes tensors to zero."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=2, num_agents=3, obs_dim=4, action_dim=2, device=device)

        assert torch.all(buffer.obs == 0.0)
        assert torch.all(buffer.actions == 0.0)
        assert torch.all(buffer.logprobs == 0.0)
        assert torch.all(buffer.rewards == 0.0)
        assert torch.all(buffer.dones == 0.0)
        assert torch.all(buffer.values == 0.0)
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert torch.all(buffer.advantages == 0.0)
        assert torch.all(buffer.returns == 0.0)

    def test_create_uses_device(self):
        """Test create() allocates tensors on specified device."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=5, num_agents=2, obs_dim=16, action_dim=4, device=device)

        assert buffer.obs.device == device
        assert buffer.actions.device == device
        assert buffer.logprobs.device == device
        assert buffer.rewards.device == device
        assert buffer.dones.device == device
        assert buffer.values.device == device
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.device == device
        assert buffer.returns.device == device


class TestGAESingleStep:
    """Test GAE computation with single-step rollouts."""

    def test_single_step_no_discount(self):
        """Test GAE with single step and gamma=1, lambda=1 (no discounting)."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=1, num_agents=1, obs_dim=1, action_dim=1, device=device)

        # Simple case: r=1, V(s)=0, V(s')=0, not done
        buffer.rewards[0, 0] = 1.0
        buffer.values[0, 0] = 0.0
        next_value = torch.tensor([0.0])
        next_done = torch.tensor([0.0])

        buffer.compute_gae(next_value, next_done, gamma=1.0, gae_lambda=1.0)

        # delta = r + gamma*V(s') - V(s) = 1 + 1*0 - 0 = 1
        # A = delta = 1
        # R = A + V(s) = 1 + 0 = 1
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert torch.allclose(buffer.advantages[0, 0], torch.tensor(1.0))
        assert torch.allclose(buffer.returns[0, 0], torch.tensor(1.0))

    def test_single_step_with_value_baseline(self):
        """Test GAE with non-zero value baseline."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=1, num_agents=1, obs_dim=1, action_dim=1, device=device)

        # r=5, V(s)=3, V(s')=2, not done, gamma=0.99
        buffer.rewards[0, 0] = 5.0
        buffer.values[0, 0] = 3.0
        next_value = torch.tensor([2.0])
        next_done = torch.tensor([0.0])

        buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=1.0)

        # delta = 5 + 0.99*2 - 3 = 5 + 1.98 - 3 = 3.98
        # A = delta = 3.98
        # R = A + V(s) = 3.98 + 3 = 6.98
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert torch.allclose(buffer.advantages[0, 0], torch.tensor(3.98), atol=1e-5)
        assert torch.allclose(buffer.returns[0, 0], torch.tensor(6.98), atol=1e-5)

    def test_single_step_terminal_state(self):
        """Test GAE with terminal state (done=True masks next value)."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=1, num_agents=1, obs_dim=1, action_dim=1, device=device)

        # r=10, V(s)=5, V(s')=100 (but terminal, so ignored), done=True
        buffer.rewards[0, 0] = 10.0
        buffer.values[0, 0] = 5.0
        next_value = torch.tensor([100.0])  # Should be masked out
        next_done = torch.tensor([1.0])

        buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

        # delta = r + gamma*V(s')*(1-done) - V(s) = 10 + 0.99*100*0 - 5 = 5
        # A = delta = 5
        # R = A + V(s) = 5 + 5 = 10
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert torch.allclose(buffer.advantages[0, 0], torch.tensor(5.0))
        assert torch.allclose(buffer.returns[0, 0], torch.tensor(10.0))

    def test_single_step_multiple_agents(self):
        """Test GAE with multiple agents in parallel."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=1, num_agents=3, obs_dim=1, action_dim=1, device=device)

        # Different rewards, values for each agent
        buffer.rewards[0] = torch.tensor([1.0, 2.0, 3.0])
        buffer.values[0] = torch.tensor([0.5, 1.0, 1.5])
        next_value = torch.tensor([0.0, 0.0, 0.0])
        next_done = torch.tensor([0.0, 0.0, 0.0])

        buffer.compute_gae(next_value, next_done, gamma=1.0, gae_lambda=1.0)

        # Agent 0: delta = 1 + 0 - 0.5 = 0.5
        # Agent 1: delta = 2 + 0 - 1.0 = 1.0
        # Agent 2: delta = 3 + 0 - 1.5 = 1.5
        assert buffer.advantages is not None
        assert buffer.returns is not None
        expected_adv = torch.tensor([0.5, 1.0, 1.5])
        expected_ret = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(buffer.advantages[0], expected_adv)
        assert torch.allclose(buffer.returns[0], expected_ret)


class TestGAEMultiStep:
    """Test GAE computation with multi-step rollouts."""

    def test_two_step_backward_recursion(self):
        """Test two-step GAE shows proper backward recursion."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=2, num_agents=1, obs_dim=1, action_dim=1, device=device)

        # Simple sequence: all rewards=1, all values=0, gamma=0.9, lambda=1
        buffer.rewards[:, 0] = torch.tensor([1.0, 1.0])
        buffer.values[:, 0] = torch.tensor([0.0, 0.0])
        buffer.dones[:, 0] = torch.tensor([0.0, 0.0])
        next_value = torch.tensor([0.0])
        next_done = torch.tensor([0.0])

        buffer.compute_gae(next_value, next_done, gamma=0.9, gae_lambda=1.0)

        # Step 1 (t=1, final step):
        #   delta_1 = 1 + 0.9*0 - 0 = 1
        #   A_1 = delta_1 = 1
        # Step 0 (t=0):
        #   delta_0 = 1 + 0.9*0 - 0 = 1
        #   A_0 = delta_0 + 0.9*1*A_1 = 1 + 0.9*1 = 1.9
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert torch.allclose(buffer.advantages[1, 0], torch.tensor(1.0))
        assert torch.allclose(buffer.advantages[0, 0], torch.tensor(1.9), atol=1e-5)

    def test_multi_step_with_lambda_discount(self):
        """Test GAE with lambda < 1 (bias-variance tradeoff)."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=3, num_agents=1, obs_dim=1, action_dim=1, device=device)

        # Constant rewards=1, values=0
        buffer.rewards[:, 0] = torch.ones(3)
        buffer.values[:, 0] = torch.zeros(3)
        buffer.dones[:, 0] = torch.zeros(3)
        next_value = torch.tensor([0.0])
        next_done = torch.tensor([0.0])

        # gamma=0.99, lambda=0.95 (typical PPO values)
        buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

        # All advantages should be > 0 (positive rewards)
        # Earlier advantages should be larger (accumulated recursion)
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert torch.all(buffer.advantages > 0)
        assert buffer.advantages[0, 0] > buffer.advantages[1, 0]
        assert buffer.advantages[1, 0] > buffer.advantages[2, 0]

    def test_terminal_in_middle_of_rollout(self):
        """Test GAE correctly handles terminal state mid-rollout."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=3, num_agents=1, obs_dim=1, action_dim=1, device=device)

        buffer.rewards[:, 0] = torch.tensor([1.0, 1.0, 1.0])
        buffer.values[:, 0] = torch.tensor([0.0, 0.0, 0.0])
        # Terminal at step 1
        buffer.dones[:, 0] = torch.tensor([0.0, 1.0, 0.0])  # done at START of step 1
        next_value = torch.tensor([0.0])
        next_done = torch.tensor([0.0])

        buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

        # Step 2: delta = 1 + 0.99*0 - 0 = 1, A = 1
        # Step 1: delta = 1 + 0.99*0*(1-done[2]) - 0 = 1, A = 1 + 0.99*0.95*0*(1-0) = 1
        #   (done[1]=1 is at START of step 1, doesn't affect step 1's own computation)
        # Step 0: delta = 1 + 0.99*0*(1-done[1]) - 0 = 1, A = 1 + 0 (done masks recursion)
        assert buffer.advantages is not None
        assert buffer.returns is not None

        # Step 0 should have no GAE accumulation from step 1 due to done
        # So A[0] ≈ delta[0] = 1.0
        assert torch.allclose(buffer.advantages[0, 0], torch.tensor(1.0), atol=1e-5)

    def test_all_zeros_rollout(self):
        """Test GAE with zero rewards and values (edge case)."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=5, num_agents=2, obs_dim=1, action_dim=1, device=device)

        # All zeros
        next_value = torch.zeros(2)
        next_done = torch.zeros(2)

        buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

        # All advantages and returns should be zero
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert torch.allclose(buffer.advantages, torch.zeros_like(buffer.advantages))
        assert torch.allclose(buffer.returns, torch.zeros_like(buffer.returns))

    def test_large_values_numerical_stability(self):
        """Test GAE doesn't overflow with large reward/value magnitudes."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=10, num_agents=1, obs_dim=1, action_dim=1, device=device)

        # Large rewards and values
        buffer.rewards[:, 0] = torch.ones(10) * 1000.0
        buffer.values[:, 0] = torch.ones(10) * 500.0
        buffer.dones[:, 0] = torch.zeros(10)
        next_value = torch.tensor([500.0])
        next_done = torch.tensor([0.0])

        buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

        # Should remain finite
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert torch.isfinite(buffer.advantages).all()
        assert torch.isfinite(buffer.returns).all()

    def test_negative_rewards(self):
        """Test GAE with negative rewards (common in RL)."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=3, num_agents=1, obs_dim=1, action_dim=1, device=device)

        buffer.rewards[:, 0] = torch.tensor([-1.0, -2.0, -3.0])
        buffer.values[:, 0] = torch.zeros(3)
        buffer.dones[:, 0] = torch.zeros(3)
        next_value = torch.tensor([0.0])
        next_done = torch.tensor([0.0])

        buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

        # Advantages should all be negative
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert torch.all(buffer.advantages < 0)
        assert torch.all(buffer.returns < 0)

    def test_mixed_terminal_states_multi_agent(self):
        """Test GAE with different agents having different terminal states."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=2, num_agents=3, obs_dim=1, action_dim=1, device=device)

        buffer.rewards[:, :] = torch.ones(2, 3)
        buffer.values[:, :] = torch.zeros(2, 3)
        # Agent 0: terminal at step 1, Agent 1: no terminal, Agent 2: terminal at step 0
        buffer.dones[0, :] = torch.tensor([0.0, 0.0, 1.0])
        buffer.dones[1, :] = torch.tensor([1.0, 0.0, 0.0])
        next_value = torch.zeros(3)
        next_done = torch.zeros(3)

        buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

        assert buffer.advantages is not None
        assert buffer.returns is not None

        # Each agent should have advantages computed independently
        # Agent 0: terminal at step 1, so step 0 has no recursion from step 1
        # Agent 1: no terminal, full recursion
        # Agent 2: terminal at step 0 (start of step 0), affects nothing

        # Verify all are finite
        assert torch.isfinite(buffer.advantages).all()
        assert torch.isfinite(buffer.returns).all()


class TestGAEMatchesReference:
    """Test GAE matches the reference implementation from train_ppo.py."""

    def test_matches_train_ppo_example(self):
        """Test GAE output matches known correct computation.

        This uses a simple case where we can manually compute the expected result.
        """
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=3, num_agents=1, obs_dim=1, action_dim=1, device=device)

        # Known inputs
        buffer.rewards[:, 0] = torch.tensor([1.0, 2.0, 3.0])
        buffer.values[:, 0] = torch.tensor([0.5, 1.0, 1.5])
        buffer.dones[:, 0] = torch.tensor([0.0, 0.0, 0.0])
        next_value = torch.tensor([2.0])
        next_done = torch.tensor([0.0])

        gamma = 0.99
        gae_lambda = 0.95

        buffer.compute_gae(next_value, next_done, gamma, gae_lambda)

        # Manual computation (backward from t=2):
        # t=2: delta_2 = 3 + 0.99*2 - 1.5 = 3.48, A_2 = 3.48
        # t=1: delta_1 = 2 + 0.99*1.5 - 1.0 = 2.485
        #      A_1 = delta_1 + 0.99*0.95*A_2 = 2.485 + 0.9405*3.48 = 5.75794
        # t=0: delta_0 = 1 + 0.99*1.0 - 0.5 = 1.49
        #      A_0 = delta_0 + 0.99*0.95*A_1 = 1.49 + 0.9405*5.75794 = 6.905343

        expected_adv_2 = 3.48
        expected_adv_1 = 2.485 + gamma * gae_lambda * expected_adv_2
        expected_adv_0 = 1.49 + gamma * gae_lambda * expected_adv_1

        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert torch.allclose(buffer.advantages[2, 0], torch.tensor(expected_adv_2), atol=1e-3)
        assert torch.allclose(buffer.advantages[1, 0], torch.tensor(expected_adv_1), atol=1e-3)
        assert torch.allclose(buffer.advantages[0, 0], torch.tensor(expected_adv_0), atol=1e-3)

        # Returns should be advantages + values
        assert torch.allclose(buffer.returns[0, 0], torch.tensor(expected_adv_0 + 0.5), atol=1e-3)
        assert torch.allclose(buffer.returns[1, 0], torch.tensor(expected_adv_1 + 1.0), atol=1e-3)
        assert torch.allclose(buffer.returns[2, 0], torch.tensor(expected_adv_2 + 1.5), atol=1e-3)


class TestRolloutBufferIntegration:
    """Integration tests for RolloutBuffer in PPO-like usage."""

    def test_typical_ppo_rollout(self):
        """Test typical PPO usage pattern."""
        device = torch.device("cpu")
        num_steps = 128
        num_agents = 20
        obs_dim = 256
        action_dim = 9

        # Create buffer
        buffer = RolloutBuffer.create(num_steps, num_agents, obs_dim, action_dim, device)

        # Simulate filling buffer (would normally come from env interaction)
        buffer.obs = torch.randn(num_steps, num_agents, obs_dim)
        buffer.actions = torch.randn(num_steps, num_agents, action_dim)
        buffer.logprobs = torch.randn(num_steps, num_agents)
        buffer.rewards = torch.randn(num_steps, num_agents)
        buffer.dones = torch.zeros(num_steps, num_agents)  # No terminals for simplicity
        buffer.values = torch.randn(num_steps, num_agents)

        next_value = torch.randn(num_agents)
        next_done = torch.zeros(num_agents)

        # Compute GAE
        buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

        # Verify outputs are reasonable
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.shape == (num_steps, num_agents)
        assert buffer.returns.shape == (num_steps, num_agents)
        assert torch.isfinite(buffer.advantages).all()
        assert torch.isfinite(buffer.returns).all()

    def test_buffer_reuse(self):
        """Test that buffer can be reused across multiple rollouts."""
        device = torch.device("cpu")
        buffer = RolloutBuffer.create(num_steps=10, num_agents=5, obs_dim=16, action_dim=4, device=device)

        for _ in range(3):
            # Fill with new data
            buffer.rewards = torch.randn(10, 5)
            buffer.values = torch.randn(10, 5)
            buffer.dones = torch.zeros(10, 5)

            # Compute GAE
            next_value = torch.randn(5)
            next_done = torch.zeros(5)
            buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

            # Should always produce valid results
            assert buffer.advantages is not None
            assert buffer.returns is not None
            assert torch.isfinite(buffer.advantages).all()
            assert torch.isfinite(buffer.returns).all()

    def test_cuda_device_if_available(self):
        """Test buffer works on CUDA if available."""
        if not torch.cuda.is_available():
            return  # Skip if CUDA not available

        device = torch.device("cuda:0")
        buffer = RolloutBuffer.create(num_steps=5, num_agents=3, obs_dim=8, action_dim=2, device=device)

        # Verify all tensors on CUDA
        assert buffer.obs.device.type == "cuda"
        assert buffer.rewards.device.type == "cuda"

        # Fill with data
        buffer.rewards = torch.ones(5, 3, device=device)
        buffer.values = torch.zeros(5, 3, device=device)
        buffer.dones = torch.zeros(5, 3, device=device)

        next_value = torch.zeros(3, device=device)
        next_done = torch.zeros(3, device=device)

        # Compute GAE on CUDA
        buffer.compute_gae(next_value, next_done, gamma=0.99, gae_lambda=0.95)

        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.device.type == "cuda"
        assert buffer.returns.device.type == "cuda"
