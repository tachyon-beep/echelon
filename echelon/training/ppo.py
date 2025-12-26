"""PPO (Proximal Policy Optimization) trainer implementation.

This module provides the PPOTrainer class that encapsulates the PPO algorithm
with clear educational boundaries. The trainer handles rollout collection,
GAE computation, and PPO updates with clipped objective.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from echelon.rl.lstm_state import LSTMState
from echelon.training.normalization import ReturnNormalizer

if TYPE_CHECKING:
    from echelon.rl.model import ActorCriticLSTM
    from echelon.training.rollout import RolloutBuffer


@dataclass
class PPOConfig:
    """PPO hyperparameters.

    These are the standard PPO hyperparameters following CleanRL conventions.
    Defaults are reasonable starting points for most tasks.

    Attributes:
        lr: Learning rate for Adam optimizer
        gamma: Discount factor for rewards (typically 0.99)
        gae_lambda: GAE lambda for bias-variance tradeoff (typically 0.95)
        clip_coef: PPO clipping coefficient (typically 0.1-0.3)
        ent_coef: Entropy bonus coefficient (encourages exploration)
        vf_coef: Value function loss coefficient (typically 0.5)
        max_grad_norm: Maximum gradient norm for clipping
        update_epochs: Number of PPO update epochs per rollout
        rollout_steps: Number of steps to collect per rollout
        num_minibatches: Number of minibatches per epoch (for TBPTT chunking)
        tbptt_chunk_length: Chunk length for truncated backprop through time (0=auto)
    """

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    rollout_steps: int = 512
    num_minibatches: int = 8  # Divide rollout into 8 chunks for better sample efficiency
    tbptt_chunk_length: int = 0  # 0 = auto-compute from rollout_steps / num_minibatches


class PPOTrainer:
    """Proximal Policy Optimization trainer.

    This class encapsulates the PPO update algorithm. Rollout collection remains
    the responsibility of the caller (train_ppo.py) since it's tightly coupled
    with environment stepping and opponent handling.

    The trainer owns:
    - PPO update logic (clipped objective, value loss, entropy bonus)
    - Optimizer and learning rate
    - Return normalization for value function stability

    Usage:
        trainer = PPOTrainer(model, config, device)

        # Caller handles rollout collection
        buffer = collect_trajectories(...)  # See train_ppo.py for details

        # Trainer handles PPO updates
        init_state = ...  # LSTM state at start of rollout
        metrics = trainer.update(buffer, init_state)

    Attributes:
        model: ActorCriticLSTM policy network
        config: PPOConfig hyperparameters
        device: Torch device (CPU or CUDA)
        optimizer: Adam optimizer
        return_normalizer: Running statistics for value function normalization
    """

    def __init__(
        self,
        model: ActorCriticLSTM,
        config: PPOConfig,
        device: torch.device,
    ):
        """Initialize PPO trainer.

        Args:
            model: ActorCriticLSTM policy to train
            config: PPO hyperparameters
            device: Device to run training on (CPU or CUDA)
        """
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)
        self.return_normalizer = ReturnNormalizer()

    def update(self, buffer: RolloutBuffer, init_state: LSTMState) -> dict[str, float]:
        """Run PPO update epochs on collected buffer with TBPTT minibatching.

        This method implements the core PPO algorithm with truncated backpropagation
        through time (TBPTT) for improved sample efficiency with recurrent policies:

        1. First pass: sequentially process rollout, saving LSTM states at chunk boundaries
        2. For each epoch:
           - Shuffle chunk order (but maintain temporal order within chunks)
           - For each minibatch of chunks:
             - Re-evaluate actions starting from saved LSTM state
             - Compute clipped surrogate objective
             - Compute value loss with normalized returns
             - Add entropy bonus
             - Gradient update

        This approach gives num_minibatches gradient updates per epoch instead of 1,
        improving sample efficiency by 4-8x while respecting LSTM temporal structure.

        Args:
            buffer: RolloutBuffer with collected trajectories (must have advantages/returns)
            init_state: Initial LSTM state at start of rollout

        Returns:
            Metrics dictionary with:
                - pg_loss: Policy gradient loss (clipped objective)
                - vf_loss: Value function loss
                - entropy: Mean policy entropy
                - grad_norm: Gradient norm after clipping
                - approx_kl: Approximate KL divergence (for monitoring)
                - clipfrac: Fraction of ratios that were clipped
        """
        if buffer.advantages is None or buffer.returns is None:
            raise ValueError("Buffer must have advantages and returns computed (call compute_gae first)")

        # Guard against pre-allocated but never-computed buffers (P2 fix)
        # Allow: gae_computed=True (normal path) OR non-zero advantages (manually filled)
        if not buffer.gae_computed and (buffer.advantages == 0).all():
            raise ValueError(
                "Buffer has zeroed advantages - either call compute_gae() or manually populate advantages/returns"
            )

        # Normalize advantages (standard practice for stability)
        advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

        # Update return normalizer and normalize returns
        self.return_normalizer.update(buffer.returns)
        returns_normalized = self.return_normalizer.normalize(buffer.returns)

        # Track metrics (we'll return averages across all minibatches)
        metrics_accum: dict[str, list[float]] = {
            "pg_loss": [],
            "vf_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clipfrac": [],
            "loss": [],
            "grad_norm": [],
        }

        # Get actual rollout length from buffer
        num_steps = buffer.obs.size(0)
        num_agents = buffer.obs.size(1)

        # Compute chunk length for TBPTT
        num_minibatches = max(1, self.config.num_minibatches)
        if self.config.tbptt_chunk_length > 0:
            chunk_length = self.config.tbptt_chunk_length
        else:
            chunk_length = max(1, num_steps // num_minibatches)

        num_chunks = (num_steps + chunk_length - 1) // chunk_length  # Ceiling division

        # First pass: compute LSTM states at chunk boundaries (detached)
        # This allows us to start each chunk from the correct state
        chunk_states: list[LSTMState] = []
        with torch.no_grad():
            lstm_state = LSTMState(h=init_state.h.detach(), c=init_state.c.detach())
            for chunk_idx in range(num_chunks):
                # Save state at start of this chunk
                chunk_states.append(LSTMState(h=lstm_state.h.clone(), c=lstm_state.c.clone()))

                # Process chunk to get state for next chunk
                start_t = chunk_idx * chunk_length
                end_t = min(start_t + chunk_length, num_steps)
                for t in range(start_t, end_t):
                    _, lstm_state = self.model._step_lstm(
                        self.model.encoder(buffer.obs[t]), lstm_state, buffer.dones[t]
                    )

        # Create chunk indices for shuffling
        chunk_indices = list(range(num_chunks))

        for _epoch in range(self.config.update_epochs):
            # Shuffle chunk order each epoch (maintains temporal order within chunks)
            random.shuffle(chunk_indices)

            for chunk_idx in chunk_indices:
                start_t = chunk_idx * chunk_length
                end_t = min(start_t + chunk_length, num_steps)
                chunk_len = end_t - start_t

                # Pre-allocate chunk buffers
                chunk_logprobs = torch.zeros(chunk_len, num_agents, device=self.device)
                chunk_values = torch.zeros(chunk_len, num_agents, device=self.device)
                chunk_entropies = torch.zeros(chunk_len, num_agents, device=self.device)

                # Start from saved state (detached to prevent cross-chunk gradients)
                lstm_state = LSTMState(
                    h=chunk_states[chunk_idx].h.detach(), c=chunk_states[chunk_idx].c.detach()
                )

                # Process chunk sequentially
                for i, t in enumerate(range(start_t, end_t)):
                    _, logprob, entropy, value, lstm_state = self.model.get_action_and_value(
                        buffer.obs[t], lstm_state, buffer.dones[t], action=buffer.actions[t]
                    )
                    chunk_logprobs[i] = logprob
                    chunk_values[i] = value
                    chunk_entropies[i] = entropy

                # Compute losses for this chunk
                loss, chunk_metrics = self._compute_losses(
                    advantages=advantages[start_t:end_t],
                    returns_normalized=returns_normalized[start_t:end_t],
                    old_logprobs=buffer.logprobs[start_t:end_t],
                    new_logprobs=chunk_logprobs,
                    new_values=chunk_values,
                    new_entropies=chunk_entropies,
                )

                # Gradient update for this minibatch
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Accumulate metrics
                for key in chunk_metrics:
                    metrics_accum[key].append(chunk_metrics[key])
                metrics_accum["grad_norm"].append(
                    float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
                )

        # Return average metrics across all minibatches
        metrics = {key: sum(vals) / len(vals) for key, vals in metrics_accum.items() if vals}
        return metrics

    def _compute_losses(
        self,
        advantages: torch.Tensor,
        returns_normalized: torch.Tensor,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        new_values: torch.Tensor,
        new_entropies: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute PPO clipped objective + value loss + entropy bonus.

        This is the core PPO loss computation following the original paper.
        The clipped objective prevents the policy from changing too much in
        a single update.

        Args:
            advantages: Normalized advantages [steps, agents]
            returns_normalized: Normalized returns [steps, agents]
            old_logprobs: Log probabilities from rollout policy [steps, agents]
            new_logprobs: Log probabilities from current policy [steps, agents]
            new_values: Value estimates from current policy [steps, agents]
            new_entropies: Entropies from current policy [steps, agents]

        Returns:
            loss: Total loss to optimize (policy + value + entropy)
            metrics: Dictionary with individual loss components and diagnostics
        """
        # Compute probability ratio and clipped surrogate objective
        logratio = new_logprobs - old_logprobs
        ratio = logratio.exp()

        # Policy loss: max(unclipped, clipped)
        # The negative sign is because we want to maximize advantage-weighted probability
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss: MSE between predicted values and normalized returns
        vf_loss = 0.5 * (returns_normalized - new_values).pow(2).mean()

        # Entropy bonus (negative because we maximize entropy)
        entropy_loss = new_entropies.mean()

        # Total loss
        loss = pg_loss - self.config.ent_coef * entropy_loss + self.config.vf_coef * vf_loss

        # Diagnostic metrics
        with torch.no_grad():
            # Approximate KL divergence (for monitoring, not used in loss)
            approx_kl = ((ratio - 1) - logratio).mean()

            # Fraction of ratios that were clipped
            clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.clip_coef).float())

        metrics = {
            "pg_loss": float(pg_loss.item()),
            "vf_loss": float(vf_loss.item()),
            "entropy": float(entropy_loss.item()),
            "approx_kl": float(approx_kl.item()),
            "clipfrac": float(clipfrac.item()),
            "loss": float(loss.item()),
        }

        return loss, metrics
