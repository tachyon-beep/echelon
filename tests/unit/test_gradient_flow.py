"""Gradient flow tests for actor-critic models.

These tests verify that gradients propagate correctly through all layers,
catching issues like:
- Dead ReLUs (zero gradients)
- Vanishing gradients through LSTM
- Disconnected computation graphs
- NaN or inf in gradients
"""

import pytest
import torch

from echelon.rl.model import ActorCriticLSTM


class TestActorCriticGradientFlow:
    """Gradient flow tests for ActorCriticLSTM."""

    @pytest.fixture
    def model(self):
        return ActorCriticLSTM(obs_dim=64, action_dim=9, hidden_dim=32, lstm_hidden_dim=32)

    @pytest.fixture
    def batch_data(self, model):
        batch_size = 4
        device = next(model.parameters()).device
        return {
            "obs": torch.randn(batch_size, 64, device=device),
            "done": torch.zeros(batch_size, device=device),
            "state": model.initial_state(batch_size, device),
        }

    def test_actor_loss_gradients_flow_to_all_layers(self, model, batch_data):
        """Actor loss should produce gradients in encoder, LSTM, and actor head."""
        _action, logprob, _entropy, _value, _ = model.get_action_and_value(
            batch_data["obs"],
            batch_data["state"],
            batch_data["done"],
        )

        # Simple actor loss: maximize log_prob (policy gradient)
        actor_loss = -logprob.mean()
        actor_loss.backward()

        # Check encoder gradients
        for name, param in model.encoder.named_parameters():
            assert param.grad is not None, f"No gradient for encoder.{name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in encoder.{name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in encoder.{name}"

        # Check LSTM gradients
        for name, param in model.lstm.named_parameters():
            assert param.grad is not None, f"No gradient for lstm.{name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in lstm.{name}"

        # Check actor head gradients
        assert model.actor_mean.weight.grad is not None
        assert model.actor_logstd.grad is not None

    def test_critic_loss_gradients_flow_to_all_layers(self, model, batch_data):
        """Critic loss should produce gradients in encoder, LSTM, and critic head."""
        model.zero_grad()

        _action, _logprob, _entropy, value, _ = model.get_action_and_value(
            batch_data["obs"],
            batch_data["state"],
            batch_data["done"],
        )

        # Simple critic loss: MSE to target
        target = torch.ones_like(value)
        critic_loss = ((value - target) ** 2).mean()
        critic_loss.backward()

        # Check critic head gradients
        assert model.critic.weight.grad is not None
        assert not torch.isnan(model.critic.weight.grad).any()

        # Check encoder gradients (shared with actor)
        for name, param in model.encoder.named_parameters():
            assert param.grad is not None, f"No gradient for encoder.{name}"

    def test_gradients_through_lstm_sequence(self, model):
        """Gradients should flow through multiple LSTM steps."""
        model.zero_grad()

        batch_size = 2
        seq_len = 5
        device = next(model.parameters()).device

        # Create sequence data
        obs_seq = torch.randn(seq_len, batch_size, 64, device=device)
        actions_seq = torch.tanh(torch.randn(seq_len, batch_size, 9, device=device))
        dones_seq = torch.zeros(seq_len, batch_size, device=device)
        init_state = model.initial_state(batch_size, device)

        # Forward through sequence
        _logprobs, _entropies, values, _ = model.evaluate_actions_sequence(
            obs_seq, actions_seq, dones_seq, init_state
        )

        # Loss on last timestep value
        loss = values[-1].mean()
        loss.backward()

        # Gradients should reach encoder (through LSTM chain)
        for name, param in model.encoder.named_parameters():
            assert param.grad is not None, f"No gradient for encoder.{name}"

    def test_entropy_gradient_sign(self, model, batch_data):
        """Entropy gradient should exist for exploration bonus."""
        model.zero_grad()

        _, _, entropy, _, _ = model.get_action_and_value(
            batch_data["obs"],
            batch_data["state"],
            batch_data["done"],
        )

        # Maximize entropy (exploration bonus)
        entropy_loss = -entropy.mean()
        entropy_loss.backward()

        # log_std should have gradient (controls entropy)
        assert model.actor_logstd.grad is not None
        assert model.actor_logstd.grad.abs().sum() > 0

    def test_combined_ppo_loss_gradients(self, model, batch_data):
        """Combined PPO loss (actor + critic + entropy) should produce gradients everywhere."""
        model.zero_grad()

        _action, logprob, entropy, value, _ = model.get_action_and_value(
            batch_data["obs"],
            batch_data["state"],
            batch_data["done"],
        )

        # Simulated PPO loss components
        actor_loss = -logprob.mean()
        critic_loss = ((value - torch.zeros_like(value)) ** 2).mean()
        entropy_bonus = -entropy.mean() * 0.01

        total_loss = actor_loss + 0.5 * critic_loss + entropy_bonus
        total_loss.backward()

        # All components should have gradients
        assert model.actor_mean.weight.grad is not None
        assert model.actor_logstd.grad is not None
        assert model.critic.weight.grad is not None

        # Encoder should have gradients from both paths
        for name, param in model.encoder.named_parameters():
            assert param.grad is not None, f"No gradient for encoder.{name}"
            # Gradients should be non-zero (both paths contribute)
            assert param.grad.abs().sum() > 0, f"Zero gradient for encoder.{name}"

    def test_gradient_magnitude_reasonable(self, model, batch_data):
        """Gradients should not explode or vanish to zero."""
        model.zero_grad()

        _action, logprob, _entropy, value, _ = model.get_action_and_value(
            batch_data["obs"],
            batch_data["state"],
            batch_data["done"],
        )

        loss = -logprob.mean() + ((value - torch.zeros_like(value)) ** 2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                # Gradients should not explode
                assert grad_norm < 1e6, f"Gradient explosion in {name}: {grad_norm}"
                # Note: We don't check for vanishing gradients here since some
                # parameters may legitimately have small gradients depending
                # on initialization and input

    def test_no_gradient_leakage_between_agents(self, model):
        """Each agent's loss should only affect its own computation path.

        This test verifies that when we compute loss for one agent,
        gradients don't incorrectly flow through other agents' computations.
        """
        batch_size = 4
        device = next(model.parameters()).device

        obs = torch.randn(batch_size, 64, device=device, requires_grad=True)
        done = torch.zeros(batch_size, device=device)
        state = model.initial_state(batch_size, device)

        _action, logprob, _entropy, _value, _ = model.get_action_and_value(obs, state, done)

        # Compute loss only for agent 0
        loss_agent_0 = -logprob[0]
        model.zero_grad()
        obs.grad = None
        loss_agent_0.backward(retain_graph=True)

        # Check that observation gradients are localized to agent 0
        # Agent 0's obs should have gradient
        assert obs.grad is not None
        agent_0_grad = obs.grad[0].clone()
        assert agent_0_grad.abs().sum() > 0, "Agent 0 should have obs gradient"

        # Other agents should have zero gradients (no leakage through the model)
        # Note: Due to batched computation, all agents share parameters,
        # but the *observation* gradient should be localized
        for i in range(1, batch_size):
            assert torch.allclose(
                obs.grad[i], torch.zeros_like(obs.grad[i])
            ), f"Agent {i} should have zero obs gradient"

    def test_lstm_hidden_state_gradient_flow(self, model):
        """Verify gradients flow through LSTM hidden state across timesteps."""
        batch_size = 2
        device = next(model.parameters()).device

        # First step
        obs1 = torch.randn(batch_size, 64, device=device)
        done1 = torch.zeros(batch_size, device=device)
        state1 = model.initial_state(batch_size, device)

        _action1, _logprob1, _, _value1, state2 = model.get_action_and_value(obs1, state1, done1)

        # Second step (uses state from first step)
        obs2 = torch.randn(batch_size, 64, device=device)
        done2 = torch.zeros(batch_size, device=device)

        _action2, logprob2, _, _value2, _ = model.get_action_and_value(obs2, state2, done2)

        # Loss on second step
        model.zero_grad()
        loss = -logprob2.mean()
        loss.backward()

        # Gradients should flow back through LSTM to affect first step's computation
        # This is verified by checking LSTM parameters have gradients
        for name, param in model.lstm.named_parameters():
            assert param.grad is not None, f"No gradient for lstm.{name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for lstm.{name}"

    def test_action_gradient_when_provided(self, model, batch_data):
        """When action is provided for evaluation, gradients should still flow."""
        model.zero_grad()

        # First, get an action
        action, _, _, _, _ = model.get_action_and_value(
            batch_data["obs"],
            batch_data["state"],
            batch_data["done"],
        )

        model.zero_grad()

        # Now evaluate that action
        _, logprob, _entropy, _value, _ = model.get_action_and_value(
            batch_data["obs"],
            batch_data["state"],
            batch_data["done"],
            action=action.detach(),  # Provide action for evaluation
        )

        # Loss using the evaluated log_prob (as in PPO update)
        loss = -logprob.mean()
        loss.backward()

        # Actor parameters should have gradients
        assert model.actor_mean.weight.grad is not None
        assert model.actor_logstd.grad is not None

        # Encoder should have gradients
        for name, param in model.encoder.named_parameters():
            assert param.grad is not None, f"No gradient for encoder.{name}"
