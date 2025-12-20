from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999, 0.999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


@dataclass(frozen=True)
class LSTMState:
    h: torch.Tensor  # [1, batch, hidden]
    c: torch.Tensor  # [1, batch, hidden]


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

    def _step_lstm(self, x: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor) -> tuple[torch.Tensor, LSTMState]:
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

        logprob = dist.log_prob(u).sum(-1) - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(y).squeeze(-1)
        return action, logprob, entropy, value, next_state

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor, lstm_state: LSTMState, done: torch.Tensor) -> tuple[torch.Tensor, LSTMState]:
        obs = obs.float()
        done = done.float()
        x = self.encoder(obs)
        y, next_state = self._step_lstm(x, lstm_state=lstm_state, done=done)
        value = self.critic(y).squeeze(-1)
        return value, next_state
