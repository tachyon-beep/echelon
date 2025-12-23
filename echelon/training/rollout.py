"""Rollout buffer for PPO trajectory storage and GAE computation.

This module provides the RolloutBuffer dataclass that stores trajectory data
collected during PPO rollouts and computes advantages using Generalized Advantage
Estimation (GAE).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBuffer:
    """Stores trajectory data for PPO updates.

    Shape convention: [num_steps, num_agents] for all tensors.

    The buffer pre-allocates storage for one rollout's worth of experience and
    provides in-place GAE computation to minimize memory allocations.

    Attributes:
        obs: Observations at each step [steps, agents, obs_dim]
        actions: Actions taken at each step [steps, agents, action_dim]
        logprobs: Log probabilities of actions [steps, agents]
        rewards: Rewards received at each step [steps, agents]
        dones: Done flags at start of step [steps, agents]
        values: Value function estimates [steps, agents]
        advantages: GAE advantages (computed by compute_gae) [steps, agents]
        returns: Discounted returns (computed by compute_gae) [steps, agents]
    """

    obs: torch.Tensor  # [steps, agents, obs_dim]
    actions: torch.Tensor  # [steps, agents, action_dim]
    logprobs: torch.Tensor  # [steps, agents]
    rewards: torch.Tensor  # [steps, agents]
    dones: torch.Tensor  # [steps, agents]
    values: torch.Tensor  # [steps, agents]
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    @classmethod
    def create(
        cls, num_steps: int, num_agents: int, obs_dim: int, action_dim: int, device: torch.device
    ) -> RolloutBuffer:
        """Pre-allocate buffer tensors on the specified device.

        Args:
            num_steps: Number of timesteps in the rollout
            num_agents: Number of agents per timestep
            obs_dim: Observation dimension per agent
            action_dim: Action dimension per agent
            device: Device to allocate tensors on (CPU or CUDA)

        Returns:
            RolloutBuffer with pre-allocated zero tensors
        """
        return cls(
            obs=torch.zeros(num_steps, num_agents, obs_dim, device=device),
            actions=torch.zeros(num_steps, num_agents, action_dim, device=device),
            logprobs=torch.zeros(num_steps, num_agents, device=device),
            rewards=torch.zeros(num_steps, num_agents, device=device),
            dones=torch.zeros(num_steps, num_agents, device=device),
            values=torch.zeros(num_steps, num_agents, device=device),
            advantages=None,
            returns=None,
        )

    def compute_gae(
        self,
        next_value: torch.Tensor,
        next_done: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute advantages and returns in-place using GAE.

        Generalized Advantage Estimation (Schulman et al., 2016) computes advantages
        via backward recursion with a bias-variance tradeoff controlled by gae_lambda.

        The algorithm:
        1. Compute TD error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
        2. Compute advantage: A_t = delta_t + gamma*lambda * (1 - done_{t+1}) * A_{t+1}
        3. Compute returns: R_t = A_t + V(s_t)

        Done masking follows CleanRL convention: dones[t] represents the done flag
        at the START of step t (i.e., whether s_t is terminal from the previous step).

        Args:
            next_value: Value estimate for state after last step [agents]
            next_done: Done flag after last step [agents]
            gamma: Discount factor (typically 0.99)
            gae_lambda: GAE lambda for bias-variance tradeoff (typically 0.95)

        Side effects:
            Sets self.advantages and self.returns to computed tensors [steps, agents]
        """
        num_steps = self.rewards.size(0)
        advantages = torch.zeros_like(self.rewards)
        lastgaelam = torch.zeros(self.rewards.size(1), device=self.rewards.device)

        # Backward recursion through timesteps
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # Bootstrap from next_value and next_done
                next_nonterminal = 1.0 - next_done.float()
                next_values = next_value
            else:
                # Use next step's value and done from buffer
                next_nonterminal = 1.0 - self.dones[t + 1].float()
                next_values = self.values[t + 1]

            # TD error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]

            # GAE recursion: A_t = delta_t + gamma*lambda * (1 - done_{t+1}) * A_{t+1}
            lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam

        # Returns are advantages + values (advantage definition: A = Q - V, Q = R)
        returns = advantages + self.values

        self.advantages = advantages
        self.returns = returns
