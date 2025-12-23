"""PPO (Proximal Policy Optimization) trainer implementation.

This module provides the PPOTrainer class that encapsulates the PPO algorithm
with clear educational boundaries. The trainer handles rollout collection,
GAE computation, and PPO updates with clipped objective.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from echelon.training.normalization import ReturnNormalizer

if TYPE_CHECKING:
    from echelon.rl.model import ActorCriticLSTM, LSTMState
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
        """Run PPO update epochs on collected buffer.

        This method implements the core PPO algorithm:
        1. For each epoch:
           - Re-evaluate actions with current policy
           - Compute clipped surrogate objective
           - Compute value loss with normalized returns
           - Add entropy bonus
           - Update via gradient descent

        The implementation uses full-batch recurrent updates (no minibatching)
        which is appropriate for LSTM policies with relatively small batch sizes.

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

        # Normalize advantages (standard practice for stability)
        advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

        # Update return normalizer and normalize returns
        self.return_normalizer.update(buffer.returns)
        returns_normalized = self.return_normalizer.normalize(buffer.returns)

        # Track metrics across epochs (we'll return the last epoch's values)
        metrics: dict[str, float] = {}

        # Get actual rollout length from buffer
        num_steps = buffer.obs.size(0)

        for _epoch in range(self.config.update_epochs):
            # Re-evaluate actions with current policy
            new_logprobs = torch.zeros_like(buffer.logprobs)
            new_values = torch.zeros_like(buffer.values)
            new_entropies = torch.zeros_like(buffer.logprobs)

            lstm_state = init_state
            for t in range(num_steps):
                _, logprob, entropy, value, lstm_state = self.model.get_action_and_value(
                    buffer.obs[t], lstm_state, buffer.dones[t], action=buffer.actions[t]
                )
                new_logprobs[t] = logprob
                new_values[t] = value
                new_entropies[t] = entropy

            # Compute losses
            loss, metrics = self._compute_losses(
                advantages=advantages,
                returns_normalized=returns_normalized,
                old_logprobs=buffer.logprobs,
                new_logprobs=new_logprobs,
                new_values=new_values,
                new_entropies=new_entropies,
            )

            # Gradient update
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            metrics["grad_norm"] = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

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
