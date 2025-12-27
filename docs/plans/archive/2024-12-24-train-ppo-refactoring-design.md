# train_ppo.py Refactoring Design

**Date:** 2024-12-24
**Status:** Approved
**Priority:** P1

## Goal

Refactor `scripts/train_ppo.py` (1,354 lines) into a testable, maintainable training package that serves as educational reference for PPO implementation.

## Priorities

1. **Testability** - Enable unit testing of PPO components independently
2. **Maintainability** - Make code easier to understand and modify
3. **Extensibility** - Enable alternative algorithms (lower priority, but structure should allow it)

## Architecture

### Module Structure

```
echelon/training/
├── __init__.py         # Public API exports
├── vec_env.py          # VectorEnv + worker process
├── rollout.py          # RolloutBuffer (stores trajectories)
├── ppo.py              # PPOTrainer class + PPOConfig
├── normalization.py    # RunningMeanStd, ReturnNormalizer
└── evaluation.py       # evaluate_vs_heuristic + EvalStats

scripts/train_ppo.py    # CLI: args → config → trainer.train() (~250 lines)
```

Each module has a single responsibility:
- **vec_env**: Parallel environment execution
- **rollout**: Trajectory storage (obs, actions, rewards, dones, values, logprobs)
- **ppo**: The PPO algorithm itself
- **normalization**: Statistical normalization utilities
- **evaluation**: Policy evaluation against baselines

---

## Component Designs

### RolloutBuffer (rollout.py)

Stores one rollout's worth of trajectories. Core data structure that PPO operates on.

```python
@dataclass
class RolloutBuffer:
    """Stores trajectory data for PPO updates.

    Shape convention: [num_steps, num_agents] for all tensors.
    """
    obs: torch.Tensor           # [steps, agents, obs_dim]
    actions: torch.Tensor       # [steps, agents, action_dim]
    logprobs: torch.Tensor      # [steps, agents]
    rewards: torch.Tensor       # [steps, agents]
    dones: torch.Tensor         # [steps, agents]
    values: torch.Tensor        # [steps, agents]

    # Computed after collection
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    @classmethod
    def create(cls, num_steps: int, num_agents: int, obs_dim: int,
               action_dim: int, device: torch.device) -> RolloutBuffer:
        """Pre-allocate buffer tensors."""
        ...

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor,
                    gamma: float, gae_lambda: float) -> None:
        """Compute advantages and returns in-place."""
        ...
```

**Design rationale:**
- **Dataclass** - Simple, inspectable, no magic
- **Tensors not numpy** - Stays on device, avoids copies
- **compute_gae as method** - GAE operates on the buffer's own data
- **Pre-allocation** - `create()` allocates once, rollout fills in

**Testability:** Unit test `compute_gae` with synthetic rewards/values.

---

### PPOTrainer (ppo.py)

Encapsulates the PPO algorithm with clear, educational method boundaries.

```python
@dataclass
class PPOConfig:
    """PPO hyperparameters."""
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

    Usage:
        trainer = PPOTrainer(model, config, device)
        for update in range(num_updates):
            buffer = trainer.collect_rollout(vec_env, blue_ids)
            metrics = trainer.update(buffer)
    """

    def __init__(self, model: ActorCriticLSTM, config: PPOConfig,
                 device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.return_normalizer = ReturnNormalizer()

    def collect_rollout(self, vec_env: VectorEnv, agent_ids: list[str],
                        lstm_state: LSTMState) -> tuple[RolloutBuffer, LSTMState]:
        """Collect rollout_steps of experience from vectorized env."""
        ...

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO update epochs on collected buffer.

        Returns metrics: {pg_loss, vf_loss, entropy, grad_norm, ...}
        """
        ...

    def _compute_losses(self, buffer: RolloutBuffer,
                        new_logprobs: torch.Tensor,
                        new_values: torch.Tensor,
                        new_entropies: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute PPO clipped objective + value loss + entropy bonus."""
        ...
```

**Design rationale:**
- **Config dataclass** - All hyperparams in one place, easy to log/serialize
- **collect_rollout separate from update** - Clear phase separation
- **Returns metrics dict** - Logging is caller's responsibility
- **_compute_losses private** - Implementation detail, but still testable

**Testability:**
- `update()` can be tested with a synthetic `RolloutBuffer`
- `_compute_losses()` can be tested with known inputs/outputs

---

### VectorEnv (vec_env.py)

Minimal changes from current - already works well. Main improvement is robustness and testability.

```python
def _env_worker(remote: Connection, env_cfg: EnvConfig) -> None:
    """Worker process that runs a single environment.

    Protocol:
        ("step", actions_dict) -> (obs, reward, term, trunc, info)
        ("reset", seed) -> (obs, info)
        ("close", None) -> exits
    """
    ...


class VectorEnv:
    """Vectorized environment using multiprocessing.

    Each env runs in a separate process to avoid GIL contention
    and CUDA fork issues (uses spawn context).

    Usage:
        vec_env = VectorEnv(num_envs=4, env_cfg=cfg)
        obs_list, infos = vec_env.reset(seeds=[0, 1, 2, 3])
        obs_list, rewards, terms, truncs, infos = vec_env.step(actions_list)
        vec_env.close()
    """

    def __init__(self, num_envs: int, env_cfg: EnvConfig): ...
    def step(self, actions_list: list[dict]) -> tuple[...]: ...
    def reset(self, seeds: list[int], indices: list[int] | None = None) -> tuple[...]: ...
    def close(self) -> None: ...
    def __enter__(self) -> VectorEnv: return self
    def __exit__(self, *args) -> None: self.close()
```

**Changes from current:**
- **Context manager support** - Ensures cleanup in tests
- **Explicit close()** - Current version lacks clean shutdown
- **Docstrings** - Protocol documented for learners

**Testability:** Can test with `num_envs=1` for fast unit tests. Context manager prevents leaked processes.

---

### Normalization (normalization.py)

```python
class RunningMeanStd:
    """Welford's online algorithm for running mean/variance.

    Used for observation normalization (optional) and return normalization.
    """
    def __init__(self, shape: tuple[int, ...] = ()):
        self.mean: np.ndarray
        self.var: np.ndarray
        self.count: float

    def update(self, x: np.ndarray | torch.Tensor) -> None: ...
    def normalize(self, x: torch.Tensor) -> torch.Tensor: ...


class ReturnNormalizer:
    """Normalizes returns for stable value function training.

    Tracks running statistics of returns and normalizes by std.
    """
    def __init__(self):
        self.rms = RunningMeanStd()

    def update(self, returns: torch.Tensor) -> None: ...
    def normalize(self, returns: torch.Tensor) -> torch.Tensor: ...
```

**Testability:** Test `RunningMeanStd` numerical accuracy with known sequences.

---

### Evaluation (evaluation.py)

```python
@dataclass
class EvalStats:
    """Evaluation results."""
    win_rate: float
    mean_hp_margin: float
    mean_episode_length: float
    episodes: int


def evaluate_vs_heuristic(
    model: ActorCriticLSTM,
    env_cfg: EnvConfig,
    episodes: int,
    seeds: list[int],
    device: torch.device,
    save_replay: bool = False,
) -> tuple[EvalStats, dict | None]:
    """Evaluate policy against HeuristicPolicy baseline.

    Returns:
        stats: Win rate, mean HP margin, episode lengths
        replay: Optional replay dict from first episode
    """
    ...
```

**Testability:** Integration test with short episodes.

---

### Slimmed train_ppo.py

The script becomes orchestration only (~250 lines, down from 1,354):

```python
#!/usr/bin/env python
"""PPO training script for Echelon.

Usage:
    uv run python scripts/train_ppo.py --size 100 --updates 200
    uv run python scripts/train_ppo.py --wandb --wandb-run-name "experiment-01"
"""
from echelon.training import PPOTrainer, PPOConfig, VectorEnv, evaluate_vs_heuristic

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ...  # ~80 lines, unchanged

def build_config(args: argparse.Namespace) -> tuple[EnvConfig, PPOConfig]:
    """Build environment and PPO configs from args."""
    ...  # ~30 lines

def setup_logging(args, run_dir: Path) -> tuple[Logger, WandbRun | None]:
    """Initialize metrics file and optional W&B."""
    ...  # ~50 lines

def main() -> None:
    args = parse_args()
    env_cfg, ppo_cfg = build_config(args)
    logger, wandb_run = setup_logging(args, run_dir)

    with VectorEnv(args.num_envs, env_cfg) as vec_env:
        model = ActorCriticLSTM(obs_dim, action_dim).to(device)
        trainer = PPOTrainer(model, ppo_cfg, device)

        lstm_state = model.init_state(num_agents, device)

        for update in range(start_update, end_update):
            # Collect experience
            buffer, lstm_state = trainer.collect_rollout(vec_env, blue_ids, lstm_state)

            # PPO update
            metrics = trainer.update(buffer)

            # Logging
            logger.log_update(update, metrics)

            # Periodic evaluation
            if update % args.eval_every == 0:
                eval_stats, replay = evaluate_vs_heuristic(model, env_cfg, ...)
                logger.log_eval(update, eval_stats)

            # Checkpointing
            if update % args.save_every == 0:
                save_checkpoint(run_dir, model, trainer, update)

if __name__ == "__main__":
    main()
```

The main loop reads like pseudocode:
1. Collect rollout
2. Update policy
3. Log metrics
4. Evaluate periodically
5. Save checkpoints

---

## Test Plan

| Module | Test Type | What to Test |
|--------|-----------|--------------|
| `rollout.py` | Unit | `compute_gae` with synthetic rewards/values |
| `ppo.py` | Unit | `_compute_losses` with known inputs |
| `ppo.py` | Unit | `update()` with synthetic buffer |
| `normalization.py` | Unit | `RunningMeanStd` numerical accuracy |
| `vec_env.py` | Integration | Single-env rollout collection |
| `evaluation.py` | Integration | Short evaluation episodes |

---

## Migration Strategy

1. Create `echelon/training/` package with `__init__.py`
2. Extract modules one at a time (normalization first - simplest)
3. Add tests for each extracted module before moving to next
4. Keep `train_ppo.py` working throughout (import from new locations)
5. Final cleanup: remove dead code from `train_ppo.py`

---

## Effort Estimate

- **Phase 1:** Extract normalization + tests (1 hour)
- **Phase 2:** Extract rollout + GAE + tests (2 hours)
- **Phase 3:** Extract VectorEnv + tests (1 hour)
- **Phase 4:** Extract PPOTrainer + tests (3 hours)
- **Phase 5:** Extract evaluation + tests (1 hour)
- **Phase 6:** Slim down train_ppo.py (1 hour)
- **Phase 7:** Final cleanup + documentation (1 hour)

**Total:** ~10 hours

---

## Out of Scope

- Arena/self-play refactoring (keep in train_ppo.py for now)
- Alternative algorithms (SAC, A2C) - structure allows it, but not implementing
- Distributed training - single-GPU focus

---

*Approved via brainstorming session on 2024-12-24*
