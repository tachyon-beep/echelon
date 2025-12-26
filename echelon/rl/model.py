from __future__ import annotations

import torch
from torch import nn

from .lstm_state import LSTMState

# Epsilon for numerical stability in tanh squashing (consistent throughout)
TANH_EPS = 1e-6


def _atanh(x: torch.Tensor) -> torch.Tensor:
    # Clamp to avoid atanh(±1) = ±inf; use 1 - TANH_EPS for consistency
    x = torch.clamp(x, -1.0 + TANH_EPS, 1.0 - TANH_EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class ActorCriticLSTM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(self.hidden_dim, self.lstm_hidden_dim)
        self.actor_mean = nn.Linear(self.lstm_hidden_dim, self.action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(self.action_dim))
        self.critic = nn.Linear(self.lstm_hidden_dim, 1)

        # Small init helps keep early actions mild.
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> LSTMState:
        device = device or next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        return LSTMState(h=h, c=c)

    def _step_lstm(
        self, x: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor
    ) -> tuple[torch.Tensor, LSTMState]:
        # x: [batch, feat]
        # done: [batch] with 1.0 meaning episode boundary before this step.
        x_seq = x.unsqueeze(0)  # [1, batch, feat]
        done_seq = done.reshape(1, -1, 1)
        h = lstm_state.h * (1.0 - done_seq)
        c = lstm_state.c * (1.0 - done_seq)
        y_seq, (h2, c2) = self.lstm(x_seq, (h, c))
        y = y_seq.squeeze(0)
        return y, LSTMState(h=h2, c=c2)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        lstm_state: LSTMState,
        done: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LSTMState]:
        """
        obs: [batch, obs_dim]
        done: [batch] float32/bool, 1 means reset hidden before processing obs
        action: optional [batch, action_dim] in [-1, 1] (tanh-squashed)
        """
        obs = obs.float()
        done = done.float()

        x = self.encoder(obs)
        y, next_state = self._step_lstm(x, lstm_state=lstm_state, done=done)

        mean = self.actor_mean(y)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)

        dist = torch.distributions.Normal(mean, std)

        if action is None:
            u = dist.rsample()
            action = torch.tanh(u)
        else:
            action = action.float()
            u = _atanh(action)

        # Log-prob with Jacobian correction for tanh squashing
        log_jacobian = torch.log(1.0 - action.pow(2) + TANH_EPS).sum(-1)
        logprob = dist.log_prob(u).sum(-1) - log_jacobian

        # Entropy with squashing correction: H(tanh(u)) = H(u) + E[log|det(J)|]
        # Since log|det(J)| = log(1 - tanh(u)^2).sum(), and this is negative,
        # the true entropy is less than Gaussian entropy.
        entropy = dist.entropy().sum(-1) + log_jacobian
        value = self.critic(y).squeeze(-1)
        return action, logprob, entropy, value, next_state

    @torch.no_grad()
    def get_value(
        self, obs: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor
    ) -> tuple[torch.Tensor, LSTMState]:
        obs = obs.float()
        done = done.float()
        x = self.encoder(obs)
        y, next_state = self._step_lstm(x, lstm_state=lstm_state, done=done)
        value = self.critic(y).squeeze(-1)
        return value, next_state

    def evaluate_actions_sequence(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        dones_seq: torch.Tensor,
        init_state: LSTMState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LSTMState]:
        """Evaluate a sequence of actions efficiently with batched encoder/heads.

        This method is optimized for PPO update: it processes an entire chunk at once
        by batching the encoder and actor/critic heads, while still stepping through
        the LSTM sequentially (required due to per-timestep done masking).

        Performance benefit: ~2-5x speedup vs calling get_action_and_value per timestep.

        Args:
            obs_seq: Observations [seq_len, batch, obs_dim]
            actions_seq: Actions taken [seq_len, batch, action_dim]
            dones_seq: Done flags [seq_len, batch]
            init_state: LSTM state at start of sequence

        Returns:
            logprobs: Log probabilities [seq_len, batch]
            entropies: Entropy values [seq_len, batch]
            values: Value estimates [seq_len, batch]
            final_state: LSTM state after processing sequence
        """
        seq_len, batch_size, _ = obs_seq.shape
        obs_seq = obs_seq.float()
        actions_seq = actions_seq.float()
        dones_seq = dones_seq.float()

        # Batch encode all timesteps at once: [seq_len, batch, obs_dim] -> [seq_len, batch, hidden]
        # Flatten to [seq_len * batch, obs_dim], encode, reshape back
        obs_flat = obs_seq.reshape(seq_len * batch_size, -1)
        encoded_flat = self.encoder(obs_flat)
        encoded_seq = encoded_flat.reshape(seq_len, batch_size, -1)

        # Process LSTM sequentially (required due to per-timestep done masking)
        # Collect LSTM outputs for all timesteps
        lstm_outputs = []
        lstm_state = init_state
        for t in range(seq_len):
            # Reset hidden state where done=1
            done_t = dones_seq[t].reshape(1, -1, 1)
            h = lstm_state.h * (1.0 - done_t)
            c = lstm_state.c * (1.0 - done_t)

            # Process single timestep through LSTM
            x_t = encoded_seq[t].unsqueeze(0)  # [1, batch, hidden]
            y_t, (h2, c2) = self.lstm(x_t, (h, c))
            lstm_outputs.append(y_t.squeeze(0))  # [batch, lstm_hidden]
            lstm_state = LSTMState(h=h2, c=c2)

        # Stack LSTM outputs: [seq_len, batch, lstm_hidden]
        lstm_out_seq = torch.stack(lstm_outputs, dim=0)

        # Batch compute actor/critic heads on all timesteps at once
        # Flatten: [seq_len * batch, lstm_hidden]
        lstm_out_flat = lstm_out_seq.reshape(seq_len * batch_size, -1)

        # Critic: [seq_len * batch, 1] -> [seq_len, batch]
        values_flat = self.critic(lstm_out_flat).squeeze(-1)
        values = values_flat.reshape(seq_len, batch_size)

        # Actor: compute mean and log_prob for given actions
        mean_flat = self.actor_mean(lstm_out_flat)  # [seq_len * batch, action_dim]
        logstd = self.actor_logstd.expand_as(mean_flat)
        std = torch.exp(logstd)

        dist = torch.distributions.Normal(mean_flat, std)

        # Compute log_prob and entropy for provided actions
        actions_flat = actions_seq.reshape(seq_len * batch_size, -1)
        u_flat = _atanh(actions_flat)

        # Log-prob with Jacobian correction for tanh squashing
        log_jacobian_flat = torch.log(1.0 - actions_flat.pow(2) + TANH_EPS).sum(-1)
        logprob_flat = dist.log_prob(u_flat).sum(-1) - log_jacobian_flat

        # Entropy with squashing correction
        entropy_flat = dist.entropy().sum(-1) + log_jacobian_flat

        # Reshape back to [seq_len, batch]
        logprobs = logprob_flat.reshape(seq_len, batch_size)
        entropies = entropy_flat.reshape(seq_len, batch_size)

        return logprobs, entropies, values, lstm_state
