"""Combat Suite Actor-Critic Model.

This model processes multiple observation streams with appropriate encoders:
- Contacts: DeepSetEncoder (variable length, permutation invariant)
- Squad positions: DeepSetEncoder (friendly positions)
- Ego state: MLP (fixed-size self state)
- Suite descriptor: Direct concatenation (role conditioning)
- Panel stats: Direct concatenation (count/mass summaries)

The encoded streams are fused and fed through an LSTM for temporal reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .lstm_state import LSTMState
from .suite import SUITE_DESCRIPTOR_DIM, DeepSetEncoder

# Epsilon for numerical stability in tanh squashing
TANH_EPS = 1e-6

# Computed flat observation dimension for default parameters
# This must match the layout in flat_obs_to_suite_obs()
FLAT_OBS_DIM = (
    20 * 25  # contacts: max_contacts * contact_dim
    + 20  # contact_mask: max_contacts
    + 12 * 10  # squad: max_squad * squad_dim
    + 12  # squad_mask: max_squad
    + 32  # ego_state: ego_dim
    + SUITE_DESCRIPTOR_DIM  # suite_descriptor: 14
    + 8  # panel_stats: panel_dim
)  # = 500 + 20 + 120 + 12 + 32 + 14 + 8 = 706


def _atanh(x: Tensor) -> Tensor:
    """Inverse hyperbolic tangent with clamping for stability."""
    x = torch.clamp(x, -1.0 + TANH_EPS, 1.0 - TANH_EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


@dataclass(frozen=True)
class SuiteObservation:
    """Structured observation for combat suite model.

    All tensors are batched: [batch, ...]
    """

    # Variable-length contact streams (with masks)
    contacts: Tensor  # [batch, max_contacts, contact_dim]
    contact_mask: Tensor  # [batch, max_contacts] - 1.0 for padding

    # Variable-length squad positions
    squad: Tensor  # [batch, max_squad, squad_dim]
    squad_mask: Tensor  # [batch, max_squad] - 1.0 for padding

    # Fixed-size streams
    ego_state: Tensor  # [batch, ego_dim] - self HP, heat, pos, vel, etc
    suite_descriptor: Tensor  # [batch, SUITE_DESCRIPTOR_DIM] - role conditioning
    panel_stats: Tensor  # [batch, panel_dim] - counts, masses, alerts


class SuiteStreamEncoder(nn.Module):
    """Encodes all observation streams into a unified representation."""

    def __init__(
        self,
        contact_dim: int = 25,
        squad_dim: int = 10,
        ego_dim: int = 32,
        panel_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 128,
        max_contacts: int = 20,
        max_squad: int = 12,
    ):
        super().__init__()
        self.contact_dim = contact_dim
        self.squad_dim = squad_dim
        self.ego_dim = ego_dim
        self.panel_dim = panel_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_contacts = max_contacts
        self.max_squad = max_squad

        # Stream encoders
        self.contact_encoder = DeepSetEncoder(
            entity_dim=contact_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            aggregation="mean",
        )

        self.squad_encoder = DeepSetEncoder(
            entity_dim=squad_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim // 2,
            aggregation="mean",
        )

        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Fusion layer
        # Inputs: contacts(hidden) + squad(hidden/2) + ego(hidden) + suite_desc + panel
        fusion_input_dim = hidden_dim + hidden_dim // 2 + hidden_dim + SUITE_DESCRIPTOR_DIM + panel_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),  # LayerNorm + ReLU instead of Tanh for better gradient flow
            nn.ReLU(),
        )

    def forward(self, obs: SuiteObservation) -> Tensor:
        """Encode all streams into unified representation.

        Args:
            obs: Structured observation with all streams

        Returns:
            [batch, output_dim] unified representation
        """
        # Encode variable-length streams
        contact_enc = self.contact_encoder(obs.contacts, obs.contact_mask)
        squad_enc = self.squad_encoder(obs.squad, obs.squad_mask)

        # Encode fixed-size streams
        ego_enc = self.ego_encoder(obs.ego_state)

        # Concatenate all streams
        fused = torch.cat(
            [
                contact_enc,
                squad_enc,
                ego_enc,
                obs.suite_descriptor,
                obs.panel_stats,
            ],
            dim=-1,
        )

        result: Tensor = self.fusion(fused)
        return result


class CombatSuiteActorCritic(nn.Module):
    """Actor-Critic model for combat suite observations.

    Uses stream encoders for different observation types,
    LSTM for temporal reasoning, and separate actor/critic heads.
    """

    def __init__(
        self,
        action_dim: int,
        contact_dim: int = 25,
        squad_dim: int = 10,
        ego_dim: int = 32,
        panel_dim: int = 8,
        hidden_dim: int = 64,
        lstm_hidden_dim: int = 128,
        max_contacts: int = 20,
        max_squad: int = 12,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # Stream encoder
        self.encoder = SuiteStreamEncoder(
            contact_dim=contact_dim,
            squad_dim=squad_dim,
            ego_dim=ego_dim,
            panel_dim=panel_dim,
            hidden_dim=hidden_dim,
            output_dim=lstm_hidden_dim,
            max_contacts=max_contacts,
            max_squad=max_squad,
        )

        # Temporal reasoning
        self.lstm = nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, batch_first=False)

        # Actor head (continuous actions with tanh squashing)
        self.actor_mean = nn.Linear(lstm_hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(lstm_hidden_dim, 1)

        # Initialize for stable early training
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> LSTMState:
        """Create initial LSTM state."""
        device = device or next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        return LSTMState(h=h, c=c)

    def _step_lstm(
        self,
        x: Tensor,
        lstm_state: LSTMState,
        done: Tensor,
    ) -> tuple[Tensor, LSTMState]:
        """Single LSTM step with reset on done.

        Args:
            x: [batch, feat] encoded observation
            lstm_state: Previous hidden state
            done: [batch] - 1.0 means episode boundary before this step

        Returns:
            output: [batch, lstm_hidden_dim]
            next_state: Updated LSTM state
        """
        x_seq = x.unsqueeze(0)  # [1, batch, feat]
        done_seq = done.reshape(1, -1, 1)

        # Reset state on episode boundary
        h = lstm_state.h * (1.0 - done_seq)
        c = lstm_state.c * (1.0 - done_seq)

        y_seq, (h2, c2) = self.lstm(x_seq, (h, c))
        y = y_seq.squeeze(0)

        return y, LSTMState(h=h2, c=c2)

    def get_action_and_value(
        self,
        obs: SuiteObservation,
        lstm_state: LSTMState,
        done: Tensor,
        action: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, LSTMState]:
        """Forward pass for training/inference.

        Args:
            obs: Structured observation
            lstm_state: Previous LSTM state
            done: [batch] episode boundary flags
            action: Optional [batch, action_dim] in [-1, 1] for log_prob computation

        Returns:
            action: [batch, action_dim] sampled or provided action
            log_prob: [batch] log probability of action
            entropy: [batch] policy entropy
            value: [batch] value estimate
            next_state: Updated LSTM state
        """
        done = done.float()

        # Encode observation streams
        x = self.encoder(obs)

        # Temporal reasoning
        y, next_state = self._step_lstm(x, lstm_state, done)

        # Actor: Gaussian with tanh squashing
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

        # Entropy with squashing correction
        entropy = dist.entropy().sum(-1) + log_jacobian

        # Critic
        value = self.critic(y).squeeze(-1)

        return action, logprob, entropy, value, next_state

    @torch.no_grad()
    def get_value(
        self,
        obs: SuiteObservation,
        lstm_state: LSTMState,
        done: Tensor,
    ) -> tuple[Tensor, LSTMState]:
        """Get value estimate only (for GAE computation)."""
        done = done.float()
        x = self.encoder(obs)
        y, next_state = self._step_lstm(x, lstm_state, done)
        value = self.critic(y).squeeze(-1)
        return value, next_state


def flat_obs_to_suite_obs(
    flat_obs: Tensor,
    max_contacts: int = 20,
    contact_dim: int = 25,
    max_squad: int = 12,
    squad_dim: int = 10,
    ego_dim: int = 32,
    panel_dim: int = 8,
) -> SuiteObservation:
    """Convert a flat observation vector to structured SuiteObservation.

    This is a compatibility layer for transitioning from flat observations.
    The flat observation is assumed to be laid out as:
    [contacts, contact_mask, squad, squad_mask, ego_state, suite_descriptor, panel_stats]

    In practice, you'd build SuiteObservation directly in the env.

    Args:
        flat_obs: [batch, flat_dim] flat observation tensor
        max_contacts: Maximum number of contacts
        contact_dim: Dimension of each contact feature
        max_squad: Maximum squad size
        squad_dim: Dimension of each squad member feature
        ego_dim: Dimension of ego state
        panel_dim: Dimension of panel stats

    Returns:
        SuiteObservation with properly shaped tensors

    Raises:
        ValueError: If flat_obs dimension doesn't match expected layout
    """
    batch_size = flat_obs.shape[0]

    # Calculate expected dimension
    expected_dim = (
        max_contacts * contact_dim
        + max_contacts
        + max_squad * squad_dim
        + max_squad
        + ego_dim
        + SUITE_DESCRIPTOR_DIM
        + panel_dim
    )

    if flat_obs.shape[1] != expected_dim:
        raise ValueError(
            f"flat_obs has {flat_obs.shape[1]} dims, expected {expected_dim}. "
            f"Check observation dimension alignment."
        )

    idx = 0

    # Contacts
    contact_size = max_contacts * contact_dim
    contacts = flat_obs[:, idx : idx + contact_size].reshape(batch_size, max_contacts, contact_dim)
    idx += contact_size

    # Contact mask
    contact_mask = flat_obs[:, idx : idx + max_contacts]
    idx += max_contacts

    # Squad
    squad_size = max_squad * squad_dim
    squad = flat_obs[:, idx : idx + squad_size].reshape(batch_size, max_squad, squad_dim)
    idx += squad_size

    # Squad mask
    squad_mask = flat_obs[:, idx : idx + max_squad]
    idx += max_squad

    # Ego state
    ego_state = flat_obs[:, idx : idx + ego_dim]
    idx += ego_dim

    # Suite descriptor
    suite_descriptor = flat_obs[:, idx : idx + SUITE_DESCRIPTOR_DIM]
    idx += SUITE_DESCRIPTOR_DIM

    # Panel stats
    panel_stats = flat_obs[:, idx : idx + panel_dim]
    idx += panel_dim

    # Sanity check: we should have consumed exactly all dimensions
    assert idx == expected_dim, f"Consumed {idx} dims but expected {expected_dim}"

    return SuiteObservation(
        contacts=contacts,
        contact_mask=contact_mask,
        squad=squad,
        squad_mask=squad_mask,
        ego_state=ego_state,
        suite_descriptor=suite_descriptor,
        panel_stats=panel_stats,
    )
