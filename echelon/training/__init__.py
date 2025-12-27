"""Training infrastructure for Echelon.

This package provides testable, maintainable components for PPO training:
- normalization: Statistical normalization utilities (RunningMeanStd, ReturnNormalizer)
- rollout: Trajectory storage (RolloutBuffer)
- ppo: PPO algorithm implementation (PPOTrainer, PPOConfig)
- vec_env: Parallel environment execution (VectorEnv)
- evaluation: Policy evaluation utilities (evaluate_vs_heuristic, EvalStats)
- spatial: Spatial data accumulation for heatmap visualization (SpatialAccumulator)
"""

from echelon.training.evaluation import EvalStats, evaluate_vs_heuristic
from echelon.training.normalization import ReturnNormalizer, RunningMeanStd
from echelon.training.ppo import PPOConfig, PPOTrainer
from echelon.training.rollout import RolloutBuffer
from echelon.training.spatial import SpatialAccumulator
from echelon.training.vec_env import VectorEnv

__all__ = [
    "EvalStats",
    "PPOConfig",
    "PPOTrainer",
    "ReturnNormalizer",
    "RolloutBuffer",
    "RunningMeanStd",
    "SpatialAccumulator",
    "VectorEnv",
    "evaluate_vs_heuristic",
]
