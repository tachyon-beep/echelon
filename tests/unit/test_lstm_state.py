"""LSTM state handling tests.

The most common failure mode in recurrent policies is incorrect state reset
at episode boundaries. These tests verify:
- State resets when done=1
- State preserved when done=0
- Per-agent isolation
"""

import pytest
import torch

from echelon.rl.model import ActorCriticLSTM, LSTMState


@pytest.fixture
def model():
    """Create a test model."""
    return ActorCriticLSTM(obs_dim=64, action_dim=9, hidden_dim=32, lstm_hidden_dim=32)


class TestLSTMStateHandling:
    """Verify LSTM state reset behavior."""

    def test_lstm_state_resets_on_done(self, model):
        """LSTM state zeros when done flag is 1."""
        batch_size = 2
        device = torch.device("cpu")

        # Initial state with non-zero values
        initial_state = LSTMState(
            h=torch.ones(1, batch_size, model.lstm_hidden_dim, device=device),
            c=torch.ones(1, batch_size, model.lstm_hidden_dim, device=device),
        )

        obs = torch.randn(batch_size, model.obs_dim, device=device)
        done = torch.ones(batch_size, device=device)  # All done

        _, _, _, _, next_state = model.get_action_and_value(obs, initial_state, done, action=None)

        # State should be reset (zeros after done masking, then LSTM processes)
        # The key is that the INPUT to LSTM was zeros (done mask applied)
        # Check that the internal _step_lstm properly zeroed the state
        # We can verify by checking that different initial states give same output when done=1

        different_initial = LSTMState(
            h=torch.randn(1, batch_size, model.lstm_hidden_dim, device=device),
            c=torch.randn(1, batch_size, model.lstm_hidden_dim, device=device),
        )

        _, _, _, _, next_state2 = model.get_action_and_value(obs, different_initial, done, action=None)

        # Both should produce same output since done=1 resets state
        assert torch.allclose(
            next_state.h, next_state2.h, atol=1e-5
        ), "LSTM output should be same regardless of initial state when done=1"

    def test_lstm_state_preserved_mid_episode(self, model):
        """LSTM state carries through when done=0."""
        batch_size = 2
        device = torch.device("cpu")

        # Different initial states
        state1 = LSTMState(
            h=torch.ones(1, batch_size, model.lstm_hidden_dim, device=device),
            c=torch.ones(1, batch_size, model.lstm_hidden_dim, device=device),
        )
        state2 = LSTMState(
            h=torch.zeros(1, batch_size, model.lstm_hidden_dim, device=device),
            c=torch.zeros(1, batch_size, model.lstm_hidden_dim, device=device),
        )

        obs = torch.randn(batch_size, model.obs_dim, device=device)
        done = torch.zeros(batch_size, device=device)  # Not done

        _, _, _, _, next1 = model.get_action_and_value(obs, state1, done, action=None)
        _, _, _, _, next2 = model.get_action_and_value(obs, state2, done, action=None)

        # Different initial states should give different outputs when done=0
        assert not torch.allclose(
            next1.h, next2.h, atol=1e-3
        ), "Different initial states should give different outputs when done=0"

    def test_lstm_state_per_agent_isolation(self, model):
        """Agent A's done doesn't reset Agent B's state."""
        batch_size = 3
        device = torch.device("cpu")

        initial_state = LSTMState(
            h=torch.ones(1, batch_size, model.lstm_hidden_dim, device=device),
            c=torch.ones(1, batch_size, model.lstm_hidden_dim, device=device),
        )

        obs = torch.randn(batch_size, model.obs_dim, device=device)
        done = torch.tensor([1.0, 0.0, 0.0], device=device)  # Only agent 0 done

        _, _, _, _, next_state = model.get_action_and_value(obs, initial_state, done, action=None)

        # Agent 0 should have reset state (different from initial)
        # Agents 1 and 2 should have preserved state influence

        # Run again with only agent 0 done, different initial for agent 0
        initial_state2 = LSTMState(
            h=torch.cat(
                [
                    torch.zeros(1, 1, model.lstm_hidden_dim, device=device),  # Agent 0 different
                    torch.ones(1, 2, model.lstm_hidden_dim, device=device),  # Agents 1,2 same
                ],
                dim=1,
            ),
            c=torch.cat(
                [
                    torch.zeros(1, 1, model.lstm_hidden_dim, device=device),
                    torch.ones(1, 2, model.lstm_hidden_dim, device=device),
                ],
                dim=1,
            ),
        )

        _, _, _, _, next_state2 = model.get_action_and_value(obs, initial_state2, done, action=None)

        # Agent 0 outputs should be same (both reset due to done)
        assert torch.allclose(
            next_state.h[:, 0], next_state2.h[:, 0], atol=1e-5
        ), "Agent 0 should have same output regardless of initial (done=1)"

        # Agents 1,2 should have same output (same initial, done=0)
        assert torch.allclose(
            next_state.h[:, 1:], next_state2.h[:, 1:], atol=1e-5
        ), "Agents 1,2 should have same output (same initial, done=0)"

    def test_initial_state_is_zeros(self, model):
        """initial_state() returns zero tensors."""
        batch_size = 5
        state = model.initial_state(batch_size)

        assert torch.all(state.h == 0.0), "Initial h should be zeros"
        assert torch.all(state.c == 0.0), "Initial c should be zeros"
        assert state.h.shape == (1, batch_size, model.lstm_hidden_dim)
        assert state.c.shape == (1, batch_size, model.lstm_hidden_dim)
