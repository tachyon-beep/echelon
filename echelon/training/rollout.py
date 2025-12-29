"""Rollout buffer for PPO trajectory storage and GAE computation.

This module provides the RolloutBuffer dataclass that stores trajectory data
collected during PPO rollouts and computes advantages using Generalized Advantage
Estimation (GAE).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from echelon.constants import (
    PACK_HEAVY_IDX,
    PACK_LEADER_IDX,
    PACK_LIGHT_IDX,
    PACK_MEDIUM_A_IDX,
    PACK_MEDIUM_B_IDX,
    PACK_SCOUT_IDX,
    PACK_SIZE,
)

# Role indices for per-role advantage normalization
ROLE_SCOUT = 0
ROLE_LIGHT = 1
ROLE_MEDIUM = 2
ROLE_HEAVY = 3
NUM_ROLES = 4


def compute_role_indices(num_agents: int, num_envs: int, device: torch.device) -> torch.Tensor:
    """Compute role indices for per-role advantage normalization.

    Maps agent indices to role categories based on pack structure:
    - ROLE_SCOUT (0): Scout mechs (recon, painting)
    - ROLE_LIGHT (1): Light mechs including pack leader
    - ROLE_MEDIUM (2): Medium mechs (2 per pack + squad leader)
    - ROLE_HEAVY (3): Heavy mechs

    Pack composition (6 mechs):
      idx 0: Scout, idx 1: Light, idx 2-3: Medium, idx 4: Heavy, idx 5: Pack Leader (light)

    Args:
        num_agents: Total number of agents (batch_size = agents_per_env * num_envs)
        num_envs: Number of parallel environments
        device: Device to place tensor on

    Returns:
        role_indices: [num_agents] int64 tensor with role indices
    """
    agents_per_env = num_agents // num_envs
    role_indices = torch.zeros(num_agents, dtype=torch.int64, device=device)

    # Detect squad structure: squad leader exists when num_packs >= 2
    num_packs = agents_per_env // PACK_SIZE
    total_in_packs = num_packs * PACK_SIZE
    has_squad_leader = num_packs >= 2 and agents_per_env > total_in_packs

    for env_idx in range(num_envs):
        base = env_idx * agents_per_env
        for i in range(agents_per_env):
            agent_idx = base + i

            # Squad leader is the last mech after all packs (medium-equivalent chassis)
            if has_squad_leader and i == total_in_packs:
                role_indices[agent_idx] = ROLE_MEDIUM
                continue

            idx_in_pack = i % PACK_SIZE

            if idx_in_pack == PACK_SCOUT_IDX:
                role_indices[agent_idx] = ROLE_SCOUT
            elif idx_in_pack in (PACK_LIGHT_IDX, PACK_LEADER_IDX):
                role_indices[agent_idx] = ROLE_LIGHT
            elif idx_in_pack in (PACK_MEDIUM_A_IDX, PACK_MEDIUM_B_IDX):
                role_indices[agent_idx] = ROLE_MEDIUM
            elif idx_in_pack == PACK_HEAVY_IDX:
                role_indices[agent_idx] = ROLE_HEAVY
            else:
                # Fallback (shouldn't reach here with valid pack structure)
                role_indices[agent_idx] = ROLE_MEDIUM

    return role_indices


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
        role_indices: Optional role index per agent for per-role normalization [agents]
                      (0=Scout, 1=Light, 2=Medium, 3=Heavy). Static across timesteps.
    """

    obs: torch.Tensor  # [steps, agents, obs_dim]
    actions: torch.Tensor  # [steps, agents, action_dim]
    logprobs: torch.Tensor  # [steps, agents]
    rewards: torch.Tensor  # [steps, agents]
    dones: torch.Tensor  # [steps, agents]
    values: torch.Tensor  # [steps, agents]
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    role_indices: torch.Tensor | None = None  # [agents] int64, role index per agent
    gae_computed: bool = False  # Flag to ensure compute_gae was called before update

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
            advantages=torch.zeros(num_steps, num_agents, device=device),
            returns=torch.zeros(num_steps, num_agents, device=device),
        )

    @torch.no_grad()
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
        # Use pre-allocated buffers, ensure they exist
        if self.advantages is None or self.returns is None:
            raise ValueError("Buffer must be created with create() - advantages/returns not pre-allocated")

        num_steps = self.rewards.size(0)
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
            self.advantages[t] = lastgaelam

        # Returns are advantages + values (advantage definition: A = Q - V, Q = R)
        # Use in-place copy to avoid allocation
        self.returns.copy_(self.advantages + self.values)

        # Mark GAE as computed (P2 fix: ensures PPO doesn't silently use zeroed values)
        self.gae_computed = True
