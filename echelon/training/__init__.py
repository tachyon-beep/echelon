"""Training infrastructure for Echelon.

This package provides testable, maintainable components for PPO training:
- normalization: Statistical normalization utilities (RunningMeanStd, ReturnNormalizer)
- rollout: Trajectory storage (RolloutBuffer)
- ppo: PPO algorithm implementation (PPOTrainer, PPOConfig)
- vec_env: Parallel environment execution (VectorEnv)
- evaluation: Policy evaluation utilities
"""

from echelon.training.normalization import ReturnNormalizer, RunningMeanStd
from echelon.training.rollout import RolloutBuffer
from echelon.training.vec_env import VectorEnv

__all__ = [
    "ReturnNormalizer",
    "RolloutBuffer",
    "RunningMeanStd",
    "VectorEnv",
]
