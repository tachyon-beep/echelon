"""Statistical normalization utilities for PPO training.

This module provides running statistics computation using Welford's algorithm
and return normalization for stable value function training.
"""

from __future__ import annotations

import torch  # noqa: TC002


class RunningMeanStd:
    """Welford's online algorithm for running mean and variance.

    Used for observation normalization (optional) and return normalization.
    Tracks running statistics incrementally without storing all historical values.

    Attributes:
        mean: Current running mean
        var: Current running variance
        count: Total number of samples seen (initialized with small epsilon for stability)
    """

    def __init__(self, epsilon: float = 1e-8):
        """Initialize running statistics.

        Args:
            epsilon: Small initial count for numerical stability
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon  # Small initial count for stability

    def update(self, x: torch.Tensor) -> None:
        """Update statistics with a batch of values.

        Uses Welford's algorithm for numerically stable online computation
        of mean and variance from batched data.

        Args:
            x: Batch of values to incorporate into running statistics
        """
        batch_mean = float(x.mean().item())
        # Use population variance (unbiased=False) for Welford's algorithm
        # Bessel's correction (default) would overestimate variance for small batches
        batch_var = float(x.var(unbiased=False).item()) if x.numel() > 1 else 0.0
        batch_count = x.numel()

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize values using running statistics.

        Args:
            x: Values to normalize

        Returns:
            Normalized values: (x - mean) / sqrt(var + epsilon)
        """
        return (x - self.mean) / (self.var**0.5 + 1e-8)  # type: ignore[no-any-return]

    def state_dict(self) -> dict:
        """Return state dictionary for checkpointing.

        Returns:
            Dictionary containing mean, var, and count
        """
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint.

        Args:
            state: Dictionary containing mean, var, and count
        """
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class ReturnNormalizer:
    """Normalizes returns for stable value function training.

    Tracks running statistics of returns and normalizes by standard deviation.
    This helps stabilize value function training by keeping targets in a
    consistent scale regardless of reward magnitude.
    """

    def __init__(self):
        """Initialize return normalizer with RunningMeanStd."""
        self.rms = RunningMeanStd()

    def update(self, returns: torch.Tensor) -> None:
        """Update running statistics with new returns.

        Args:
            returns: Batch of computed returns from rollout
        """
        self.rms.update(returns)

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        """Normalize returns using running statistics.

        Args:
            returns: Returns to normalize

        Returns:
            Normalized returns
        """
        return self.rms.normalize(returns)

    def state_dict(self) -> dict:
        """Return state dictionary for checkpointing.

        Returns:
            Nested state dict containing RMS statistics
        """
        return self.rms.state_dict()

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint.

        Args:
            state: State dictionary to load
        """
        self.rms.load_state_dict(state)
