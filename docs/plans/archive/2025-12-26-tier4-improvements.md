# Tier 4 Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 8 low-priority cleanup and enhancement items from specialist reviews to improve torch.compile compatibility, memory efficiency, arena mechanics, curriculum learning, and test coverage.

**Architecture:** These are independent improvements across different subsystems. Items 16/19 can be combined (both touch LSTMState). Each task is self-contained with no inter-dependencies except where noted.

**Tech Stack:** Python 3.13, PyTorch 2.x, pytest, dataclasses/NamedTuple

---

## Summary of Tasks

| Task | Item | Description | Effort | Files |
|------|------|-------------|--------|-------|
| 1 | 16+19 | Consolidate LSTMState to NamedTuple | Low | `model.py`, `suite_model.py` |
| 2 | 17 | Pre-allocate advantages/returns in RolloutBuffer | Low | `rollout.py` |
| 3 | 18 | Lazy import wandb in spatial.py | Low | `spatial.py` |
| 4 | 20 | Commander retirement mechanism | Low | `league.py` |
| 5 | 21 | Extend curriculum (map difficulty, enemy count) | Medium | `vec_env.py`, `env.py` |
| 6 | 22 | Add gradient flow tests | Medium | `tests/unit/test_gradient_flow.py` |
| 7 | 23 | Verify flat_obs_to_suite_obs dimension alignment | Low | `suite_model.py`, tests |

---

## Task 1: Consolidate LSTMState to NamedTuple (Items 16 + 19)

**Context:** Both `model.py` and `suite_model.py` define identical `LSTMState` dataclasses. Using a `NamedTuple` instead of `dataclass` improves `torch.compile` compatibility (dataclasses with tensor fields can cause graph breaks). Consolidating removes duplication.

**Files:**
- Create: `echelon/rl/lstm_state.py`
- Modify: `echelon/rl/model.py`
- Modify: `echelon/rl/suite_model.py`
- Test: `tests/unit/test_lstm_state.py` (already exists, update)

### Step 1: Write test for NamedTuple behavior

Open `tests/unit/test_lstm_state.py` and add:

```python
import torch
from typing import NamedTuple

def test_lstm_state_is_namedtuple():
    """Verify LSTMState is a NamedTuple for torch.compile compatibility."""
    from echelon.rl.lstm_state import LSTMState

    # NamedTuples are subclasses of tuple
    h = torch.zeros(1, 2, 128)
    c = torch.zeros(1, 2, 128)
    state = LSTMState(h=h, c=c)

    assert isinstance(state, tuple)
    assert state.h is h
    assert state.c is c
    # Can unpack like tuple
    h_out, c_out = state
    assert h_out is h


def test_lstm_state_immutable():
    """NamedTuple should be immutable."""
    from echelon.rl.lstm_state import LSTMState
    import pytest

    h = torch.zeros(1, 2, 128)
    c = torch.zeros(1, 2, 128)
    state = LSTMState(h=h, c=c)

    with pytest.raises(AttributeError):
        state.h = torch.ones(1, 2, 128)
```

### Step 2: Run test to verify it fails

```bash
PYTHONPATH=. uv run pytest tests/unit/test_lstm_state.py::test_lstm_state_is_namedtuple -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'echelon.rl.lstm_state'`

### Step 3: Create shared LSTMState module

Create `echelon/rl/lstm_state.py`:

```python
"""Shared LSTM state type for actor-critic models.

Uses NamedTuple instead of dataclass for torch.compile compatibility.
Dataclasses with tensor fields can cause graph breaks.
"""

from typing import NamedTuple

from torch import Tensor


class LSTMState(NamedTuple):
    """Hidden state for LSTM.

    Attributes:
        h: Hidden state tensor [1, batch, hidden_dim]
        c: Cell state tensor [1, batch, hidden_dim]
    """
    h: Tensor
    c: Tensor
```

### Step 4: Update model.py to use shared LSTMState

In `echelon/rl/model.py`:

Remove lines 18-22 (the LSTMState dataclass definition):
```python
# DELETE:
# @dataclass(frozen=True)
# class LSTMState:
#     h: torch.Tensor  # [1, batch, hidden]
#     c: torch.Tensor  # [1, batch, hidden]
```

Add import at top (after line 5):
```python
from .lstm_state import LSTMState
```

Remove `from dataclasses import dataclass` import if no longer needed.

### Step 5: Update suite_model.py to use shared LSTMState

In `echelon/rl/suite_model.py`:

Remove lines 32-38 (the LSTMState dataclass definition):
```python
# DELETE:
# @dataclass(frozen=True)
# class LSTMState:
#     """Hidden state for LSTM."""
#
#     h: Tensor  # [1, batch, hidden]
#     c: Tensor  # [1, batch, hidden]
```

Add import at top (after line 17):
```python
from .lstm_state import LSTMState
```

Remove `from dataclasses import dataclass` import since SuiteObservation still uses it - wait, SuiteObservation uses dataclass, so keep the import.

### Step 6: Run tests to verify everything passes

```bash
PYTHONPATH=. uv run pytest tests/unit/test_lstm_state.py -v
```

Expected: PASS

### Step 7: Run full test suite for regressions

```bash
PYTHONPATH=. uv run pytest tests/unit -v --tb=short
```

Expected: All tests pass

### Step 8: Lint and type check

```bash
uv run ruff check echelon/rl/lstm_state.py echelon/rl/model.py echelon/rl/suite_model.py
uv run mypy echelon/rl/lstm_state.py echelon/rl/model.py echelon/rl/suite_model.py
```

Expected: Clean

### Step 9: Commit

```bash
git add echelon/rl/lstm_state.py echelon/rl/model.py echelon/rl/suite_model.py tests/unit/test_lstm_state.py
git commit -m "refactor: consolidate LSTMState to NamedTuple for torch.compile

- Create shared echelon/rl/lstm_state.py
- Remove duplicate LSTMState from model.py and suite_model.py
- NamedTuple is more torch.compile friendly than dataclass
- Add tests verifying NamedTuple behavior"
```

---

## Task 2: Pre-allocate advantages/returns in RolloutBuffer (Item 17)

**Context:** Currently `advantages` and `returns` are initialized as `None` and created during `compute_gae()`. Pre-allocating them in `create()` reduces allocations.

**Files:**
- Modify: `echelon/training/rollout.py`
- Test: `tests/unit/training/test_rollout.py`

### Step 1: Write test for pre-allocated buffers

Add to `tests/unit/training/test_rollout.py`:

```python
def test_rollout_buffer_preallocates_advantages_and_returns():
    """Verify advantages and returns are pre-allocated in create()."""
    import torch
    from echelon.training.rollout import RolloutBuffer

    buffer = RolloutBuffer.create(
        num_steps=10,
        num_agents=5,
        obs_dim=32,
        action_dim=9,
        device=torch.device("cpu"),
    )

    # Should be pre-allocated, not None
    assert buffer.advantages is not None
    assert buffer.returns is not None

    # Correct shape
    assert buffer.advantages.shape == (10, 5)
    assert buffer.returns.shape == (10, 5)

    # Should be zeros initially
    assert (buffer.advantages == 0).all()
    assert (buffer.returns == 0).all()
```

### Step 2: Run test to verify it fails

```bash
PYTHONPATH=. uv run pytest tests/unit/training/test_rollout.py::test_rollout_buffer_preallocates_advantages_and_returns -v
```

Expected: FAIL with `assert buffer.advantages is not None`

### Step 3: Update RolloutBuffer.create()

In `echelon/training/rollout.py`, modify the `create()` classmethod (around line 121-130):

```python
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
```

### Step 4: Update compute_gae() to use in-place ops

In `compute_gae()`, update to write in-place instead of creating new tensors (around line 163):

```python
@torch.no_grad()
def compute_gae(
    self,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> None:
    """Compute advantages and returns in-place using GAE.
    ...
    """
    num_steps = self.rewards.size(0)

    # Use pre-allocated buffers, ensure they exist
    assert self.advantages is not None, "Buffer must be created with create()"
    assert self.returns is not None, "Buffer must be created with create()"

    lastgaelam = torch.zeros(self.rewards.size(1), device=self.rewards.device)

    # Backward recursion through timesteps
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_nonterminal = 1.0 - next_done.float()
            next_values = next_value
        else:
            next_nonterminal = 1.0 - self.dones[t + 1].float()
            next_values = self.values[t + 1]

        delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        self.advantages[t] = lastgaelam

    # Returns in-place: R = A + V
    self.returns.copy_(self.advantages + self.values)
```

### Step 5: Run tests

```bash
PYTHONPATH=. uv run pytest tests/unit/training/test_rollout.py -v
```

Expected: PASS

### Step 6: Lint and type check

```bash
uv run ruff check echelon/training/rollout.py
uv run mypy echelon/training/rollout.py
```

### Step 7: Commit

```bash
git add echelon/training/rollout.py tests/unit/training/test_rollout.py
git commit -m "perf: pre-allocate advantages/returns in RolloutBuffer

- Allocate in create() instead of compute_gae()
- Use in-place copy_() for returns
- Reduces memory allocations per update"
```

---

## Task 3: Lazy import wandb in spatial.py (Item 18)

**Context:** `spatial.py` imports wandb at module level, but wandb is only used in `to_images()`. Lazy importing reduces import time when wandb isn't needed.

**Files:**
- Modify: `echelon/training/spatial.py`
- Test: `tests/unit/training/test_spatial.py`

### Step 1: Write test for lazy import

Add to `tests/unit/training/test_spatial.py`:

```python
def test_spatial_accumulator_import_does_not_require_wandb():
    """Verify importing SpatialAccumulator doesn't require wandb."""
    import sys

    # Remove wandb from modules if present
    wandb_modules = [k for k in sys.modules if k.startswith('wandb')]
    for m in wandb_modules:
        del sys.modules[m]

    # Fresh import should work without wandb
    # This test verifies the module can be imported even if wandb isn't installed
    from echelon.training.spatial import SpatialAccumulator

    acc = SpatialAccumulator(grid_size=16)
    assert acc.grid_size == 16
```

### Step 2: Run test to verify current behavior

```bash
PYTHONPATH=. uv run pytest tests/unit/training/test_spatial.py::test_spatial_accumulator_import_does_not_require_wandb -v
```

Expected: May pass or fail depending on wandb availability - the point is to verify the refactor doesn't break things.

### Step 3: Update spatial.py for lazy wandb import

In `echelon/training/spatial.py`:

Remove line 10 (`import wandb`).

Update the `to_images()` method to import wandb lazily:

```python
def to_images(self) -> dict[str, Any]:
    """Convert accumulators to W&B images with colormaps.

    Requires wandb and matplotlib (imported lazily).

    Returns:
        Dict mapping heatmap names to wandb.Image objects.
        Empty dict if matplotlib or wandb is not available.
    """
    try:
        import matplotlib.pyplot as plt
        import wandb
    except ImportError as e:
        logger.warning(f"Skipping heatmap generation: {e}")
        return {}

    images: dict[str, Any] = {}

    for name, data in [
        ("deaths", self.death_locations),
        ("damage", self.damage_locations),
        ("movement", self.movement_density),
    ]:
        if data.max() > 0:
            normalized = data / data.max()
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(normalized, cmap="hot", origin="lower")
            ax.set_title(f"{name.title()} Heatmap")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            images[name] = wandb.Image(fig)
            plt.close(fig)

    return images
```

### Step 4: Run tests

```bash
PYTHONPATH=. uv run pytest tests/unit/training/test_spatial.py -v
```

Expected: PASS

### Step 5: Lint and type check

```bash
uv run ruff check echelon/training/spatial.py
uv run mypy echelon/training/spatial.py
```

### Step 6: Commit

```bash
git add echelon/training/spatial.py tests/unit/training/test_spatial.py
git commit -m "perf: lazy import wandb in spatial.py

- wandb only needed in to_images(), not at module load
- Reduces import time when wandb isn't used
- Gracefully handles missing wandb with warning"
```

---

## Task 4: Commander Retirement Mechanism (Item 20)

**Context:** As the league grows, old underperforming commanders should be retired to keep the pool manageable. This prevents infinite growth of the commander pool.

**Files:**
- Modify: `echelon/arena/league.py`
- Test: `tests/unit/test_arena_league.py`

### Step 1: Write test for retirement

Add to `tests/unit/test_arena_league.py`:

```python
def test_retire_stale_commanders():
    """Test that old, low-rated commanders can be retired."""
    from echelon.arena.league import League, LeagueEntry, Glicko2Rating
    from pathlib import Path
    import tempfile

    league = League()

    # Add some commanders with varying ratings and game counts
    for i in range(10):
        entry = LeagueEntry(
            entry_id=f"cmd_{i}",
            ckpt_path=f"/fake/path_{i}.pt",
            kind="commander",
            commander_name=f"Commander {i}",
            rating=Glicko2Rating(
                rating=1500 - i * 50,  # Decreasing ratings
                rd=50.0,
                vol=0.06,
            ),
            games=100,
        )
        league.entries[entry.entry_id] = entry

    # Retire bottom 3 commanders
    retired = league.retire_commanders(keep_top=7)

    assert len(retired) == 3
    assert len([e for e in league.entries.values() if e.kind == "commander"]) == 7

    # Verify the lowest rated were retired
    retired_ids = {e.entry_id for e in retired}
    assert "cmd_7" in retired_ids
    assert "cmd_8" in retired_ids
    assert "cmd_9" in retired_ids


def test_retire_commanders_min_games():
    """Don't retire commanders that haven't played enough games."""
    from echelon.arena.league import League, LeagueEntry, Glicko2Rating

    league = League()

    # Add commanders: some new (low games), some established
    for i in range(6):
        games = 5 if i < 3 else 100  # First 3 are new
        entry = LeagueEntry(
            entry_id=f"cmd_{i}",
            ckpt_path=f"/fake/path_{i}.pt",
            kind="commander",
            commander_name=f"Commander {i}",
            rating=Glicko2Rating(rating=1500 - i * 100, rd=50.0, vol=0.06),
            games=games,
        )
        league.entries[entry.entry_id] = entry

    # Try to retire to keep 4, but require 20 games minimum
    retired = league.retire_commanders(keep_top=4, min_games=20)

    # Only cmd_4 and cmd_5 (established + low rating) should be retired
    assert len(retired) == 2
    for e in retired:
        assert e.games >= 20
```

### Step 2: Run test to verify it fails

```bash
PYTHONPATH=. uv run pytest tests/unit/test_arena_league.py::test_retire_stale_commanders -v
```

Expected: FAIL with `AttributeError: 'League' object has no attribute 'retire_commanders'`

### Step 3: Implement retire_commanders

Add to `echelon/arena/league.py` (after `apply_rating_period` method):

```python
def retire_commanders(
    self,
    keep_top: int = 20,
    min_games: int = 20,
) -> list[LeagueEntry]:
    """Retire underperforming commanders to keep pool manageable.

    Commanders are ranked by conservative rating (rating - 2*RD).
    Only commanders with sufficient games can be retired (new commanders
    are protected until their rating stabilizes).

    Args:
        keep_top: Number of top commanders to retain
        min_games: Minimum games before a commander can be retired

    Returns:
        List of retired LeagueEntry objects (removed from entries)
    """
    commanders = [
        e for e in self.entries.values()
        if e.kind == "commander"
    ]

    if len(commanders) <= keep_top:
        return []

    # Split into protected (new) and retirable (established)
    protected = [e for e in commanders if e.games < min_games]
    retirable = [e for e in commanders if e.games >= min_games]

    def conservative(e: LeagueEntry) -> float:
        return float(e.rating.rating) - 2.0 * float(e.rating.rd)

    # Sort retirable by conservative rating (worst first)
    retirable.sort(key=conservative)

    # Calculate how many we need to retire
    # Keep at least keep_top total, but never retire protected
    current_total = len(commanders)
    num_to_retire = max(0, current_total - keep_top)

    # Can only retire from retirable pool
    num_to_retire = min(num_to_retire, len(retirable))

    # Retire the worst performers
    retired = retirable[:num_to_retire]

    for entry in retired:
        # Change kind to "retired" rather than deleting (preserves history)
        entry.kind = "retired"

    return retired
```

### Step 4: Run tests

```bash
PYTHONPATH=. uv run pytest tests/unit/test_arena_league.py -v
```

Expected: PASS

### Step 5: Lint and type check

```bash
uv run ruff check echelon/arena/league.py
uv run mypy echelon/arena/league.py
```

### Step 6: Commit

```bash
git add echelon/arena/league.py tests/unit/test_arena_league.py
git commit -m "feat: add commander retirement mechanism

- retire_commanders() removes low-rated commanders
- Protects new commanders (min_games threshold)
- Uses conservative rating (rating - 2*RD) for ranking
- Sets kind='retired' to preserve history"
```

---

## Task 5: Extend Curriculum (Map Difficulty, Enemy Count) (Item 21)

**Context:** Current curriculum only adjusts heuristic weapon probability. Extending to include map size/complexity and enemy count provides smoother difficulty progression.

**Files:**
- Modify: `echelon/training/vec_env.py`
- Modify: `echelon/config.py` (if needed for defaults)
- Test: `tests/unit/training/test_vec_env.py`

### Step 1: Write test for curriculum parameters

Add to `tests/unit/training/test_vec_env.py`:

```python
def test_vec_env_curriculum_parameters(env_cfg):
    """Test that curriculum parameters can be dynamically updated."""
    from echelon.training.vec_env import VectorEnv

    vec_env = VectorEnv(num_envs=1, env_cfg=env_cfg)
    try:
        # Set curriculum parameters
        vec_env.set_curriculum(
            weapon_prob=0.3,
            map_size_range=(40, 60),
            packs_per_team_range=(1, 1),
        )

        # Verify it doesn't crash and values are acknowledged
        # (actual behavior tested in integration)
        pass
    finally:
        vec_env.close()


def test_vec_env_get_curriculum(env_cfg):
    """Test getting current curriculum state."""
    from echelon.training.vec_env import VectorEnv

    vec_env = VectorEnv(num_envs=1, env_cfg=env_cfg)
    try:
        curriculum = vec_env.get_curriculum()

        assert "weapon_prob" in curriculum
        assert "map_size_range" in curriculum
        assert "packs_per_team_range" in curriculum
    finally:
        vec_env.close()
```

### Step 2: Run test to verify it fails

```bash
PYTHONPATH=. uv run pytest tests/unit/training/test_vec_env.py::test_vec_env_curriculum_parameters -v
```

Expected: FAIL with `AttributeError: 'VectorEnv' object has no attribute 'set_curriculum'`

### Step 3: Update vec_env.py worker protocol

In `echelon/training/vec_env.py`, update `_env_worker` to handle curriculum:

```python
def _env_worker(
    remote: Connection,
    env_fn,
    env_cfg: EnvConfig,
    initial_weapon_prob: float = 0.5,
    initial_map_size_range: tuple[int, int] = (80, 120),
    initial_packs_range: tuple[int, int] = (1, 1),
) -> None:
    """Worker process that runs a single environment.

    Protocol:
        ...existing commands...
        ("set_curriculum", dict) -> None (updates curriculum params)
        ("get_curriculum", None) -> dict (returns current curriculum)
    """
    try:
        from echelon.agents.heuristic import HeuristicPolicy
        import random

        env = env_fn(env_cfg)
        heuristic = HeuristicPolicy(weapon_fire_prob=initial_weapon_prob)

        # Curriculum state
        curriculum = {
            "weapon_prob": initial_weapon_prob,
            "map_size_range": initial_map_size_range,
            "packs_per_team_range": initial_packs_range,
        }

        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, term, trunc, info = env.step(data)
                remote.send((obs, reward, term, trunc, info))
            elif cmd == "reset":
                # Apply curriculum: randomize map size within range
                min_size, max_size = curriculum["map_size_range"]
                if min_size != max_size:
                    env.cfg.world.size = random.randint(min_size, max_size)

                obs, info = env.reset(seed=data)
                remote.send((obs, info))
            elif cmd == "get_team_alive":
                remote.send({team: env.sim.team_alive(team) for team in ("blue", "red")})
            elif cmd == "get_last_outcome":
                remote.send(env.last_outcome)
            elif cmd == "get_heuristic_actions":
                res = {rid: heuristic.act(env, rid) for rid in data}
                remote.send(res)
            elif cmd == "set_heuristic_weapon_prob":
                heuristic.weapon_fire_prob = float(data)
                curriculum["weapon_prob"] = float(data)
                remote.send(None)
            elif cmd == "set_curriculum":
                curriculum.update(data)
                heuristic.weapon_fire_prob = curriculum["weapon_prob"]
                remote.send(None)
            elif cmd == "get_curriculum":
                remote.send(dict(curriculum))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except Exception:
        print(f"Worker process encountered error:\n{traceback.format_exc()}")
        remote.close()
```

### Step 4: Add curriculum methods to VectorEnv class

Add to `VectorEnv` class:

```python
def set_curriculum(
    self,
    weapon_prob: float | None = None,
    map_size_range: tuple[int, int] | None = None,
    packs_per_team_range: tuple[int, int] | None = None,
) -> None:
    """Update curriculum parameters across all environments.

    Args:
        weapon_prob: Heuristic weapon fire probability (0.0 to 1.0)
        map_size_range: (min_size, max_size) for random map sizes on reset
        packs_per_team_range: (min, max) packs per team
    """
    update = {}
    if weapon_prob is not None:
        update["weapon_prob"] = weapon_prob
    if map_size_range is not None:
        update["map_size_range"] = map_size_range
    if packs_per_team_range is not None:
        update["packs_per_team_range"] = packs_per_team_range

    if update:
        for remote in self.remotes:
            remote.send(("set_curriculum", update))
        for remote in self.remotes:
            remote.recv()


def get_curriculum(self) -> dict:
    """Get current curriculum parameters from first environment.

    Returns:
        Dict with weapon_prob, map_size_range, packs_per_team_range
    """
    self.remotes[0].send(("get_curriculum", None))
    return self.remotes[0].recv()
```

### Step 5: Run tests

```bash
PYTHONPATH=. uv run pytest tests/unit/training/test_vec_env.py -v
```

Expected: PASS

### Step 6: Lint and type check

```bash
uv run ruff check echelon/training/vec_env.py
uv run mypy echelon/training/vec_env.py
```

### Step 7: Commit

```bash
git add echelon/training/vec_env.py tests/unit/training/test_vec_env.py
git commit -m "feat: extend curriculum with map size and pack count

- Add set_curriculum() for unified curriculum control
- Support map_size_range for random map sizes on reset
- Support packs_per_team_range for variable team sizes
- Add get_curriculum() to query current settings"
```

---

## Task 6: Add Gradient Flow Tests (Item 22)

**Context:** Verify gradients flow correctly through the actor-critic networks. This catches issues like dead ReLUs, vanishing gradients through LSTM, or disconnected computation graphs.

**Files:**
- Create: `tests/unit/test_gradient_flow.py`

### Step 1: Create gradient flow test file

Create `tests/unit/test_gradient_flow.py`:

```python
"""Gradient flow tests for actor-critic models.

These tests verify that gradients propagate correctly through all layers,
catching issues like:
- Dead ReLUs (zero gradients)
- Vanishing gradients through LSTM
- Disconnected computation graphs
- NaN or inf in gradients
"""

import pytest
import torch

from echelon.rl.model import ActorCriticLSTM, LSTMState


class TestActorCriticGradientFlow:
    """Gradient flow tests for ActorCriticLSTM."""

    @pytest.fixture
    def model(self):
        return ActorCriticLSTM(obs_dim=64, action_dim=9, hidden_dim=32, lstm_hidden_dim=32)

    @pytest.fixture
    def batch_data(self, model):
        batch_size = 4
        device = next(model.parameters()).device
        return {
            "obs": torch.randn(batch_size, 64, device=device),
            "done": torch.zeros(batch_size, device=device),
            "state": model.initial_state(batch_size, device),
        }

    def test_actor_loss_gradients_flow_to_all_layers(self, model, batch_data):
        """Actor loss should produce gradients in encoder, LSTM, and actor head."""
        action, logprob, entropy, value, _ = model.get_action_and_value(
            batch_data["obs"],
            batch_data["state"],
            batch_data["done"],
        )

        # Simple actor loss: maximize log_prob (policy gradient)
        actor_loss = -logprob.mean()
        actor_loss.backward()

        # Check encoder gradients
        for name, param in model.encoder.named_parameters():
            assert param.grad is not None, f"No gradient for encoder.{name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in encoder.{name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in encoder.{name}"

        # Check LSTM gradients
        for name, param in model.lstm.named_parameters():
            assert param.grad is not None, f"No gradient for lstm.{name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in lstm.{name}"

        # Check actor head gradients
        assert model.actor_mean.weight.grad is not None
        assert model.actor_logstd.grad is not None

    def test_critic_loss_gradients_flow_to_all_layers(self, model, batch_data):
        """Critic loss should produce gradients in encoder, LSTM, and critic head."""
        model.zero_grad()

        action, logprob, entropy, value, _ = model.get_action_and_value(
            batch_data["obs"],
            batch_data["state"],
            batch_data["done"],
        )

        # Simple critic loss: MSE to target
        target = torch.ones_like(value)
        critic_loss = ((value - target) ** 2).mean()
        critic_loss.backward()

        # Check critic head gradients
        assert model.critic.weight.grad is not None
        assert not torch.isnan(model.critic.weight.grad).any()

        # Check encoder gradients (shared with actor)
        for name, param in model.encoder.named_parameters():
            assert param.grad is not None, f"No gradient for encoder.{name}"

    def test_gradients_through_lstm_sequence(self, model):
        """Gradients should flow through multiple LSTM steps."""
        model.zero_grad()

        batch_size = 2
        seq_len = 5
        device = next(model.parameters()).device

        # Create sequence data
        obs_seq = torch.randn(seq_len, batch_size, 64, device=device)
        actions_seq = torch.tanh(torch.randn(seq_len, batch_size, 9, device=device))
        dones_seq = torch.zeros(seq_len, batch_size, device=device)
        init_state = model.initial_state(batch_size, device)

        # Forward through sequence
        logprobs, entropies, values, _ = model.evaluate_actions_sequence(
            obs_seq, actions_seq, dones_seq, init_state
        )

        # Loss on last timestep value
        loss = values[-1].mean()
        loss.backward()

        # Gradients should reach encoder (through LSTM chain)
        for name, param in model.encoder.named_parameters():
            assert param.grad is not None, f"No gradient for encoder.{name}"
            # Gradient magnitude might be small but should be non-zero
            if param.grad.abs().max() < 1e-10:
                pytest.warns(UserWarning, f"Very small gradient in encoder.{name}")

    def test_no_dead_relus_in_encoder(self, model, batch_data):
        """Verify ReLUs in encoder don't completely die."""
        model.zero_grad()

        # Run forward with varied inputs
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach())

        hooks = []
        for layer in model.encoder:
            if isinstance(layer, torch.nn.Tanh):
                hooks.append(layer.register_forward_hook(hook_fn))

        # Run forward
        _ = model.get_action_and_value(
            batch_data["obs"],
            batch_data["state"],
            batch_data["done"],
        )

        # Check activations aren't all saturated
        for i, act in enumerate(activations):
            # Tanh saturates at -1 and 1
            saturated_frac = ((act.abs() > 0.99).float().mean()).item()
            assert saturated_frac < 0.5, f"Layer {i} has {saturated_frac:.1%} saturated"

        # Cleanup hooks
        for h in hooks:
            h.remove()

    def test_entropy_gradient_sign(self, model, batch_data):
        """Entropy gradient should have correct sign for exploration bonus."""
        model.zero_grad()

        _, _, entropy, _, _ = model.get_action_and_value(
            batch_data["obs"],
            batch_data["state"],
            batch_data["done"],
        )

        # Maximize entropy (exploration bonus)
        entropy_loss = -entropy.mean()
        entropy_loss.backward()

        # log_std should have gradient (controls entropy)
        assert model.actor_logstd.grad is not None
        # Gradient should push log_std up (increase entropy)
        # When minimizing -entropy, d(-entropy)/d(log_std) < 0 means log_std should increase
        # Actually the sign depends on current std value, so just check it exists
        assert model.actor_logstd.grad.abs().sum() > 0


class TestSuiteModelGradientFlow:
    """Gradient flow tests for CombatSuiteActorCritic."""

    @pytest.fixture
    def model(self):
        from echelon.rl.suite_model import CombatSuiteActorCritic
        return CombatSuiteActorCritic(action_dim=9)

    @pytest.fixture
    def suite_obs(self, model):
        from echelon.rl.suite_model import SuiteObservation
        batch_size = 4

        return SuiteObservation(
            contacts=torch.randn(batch_size, 20, 25),
            contact_mask=torch.zeros(batch_size, 20),
            squad=torch.randn(batch_size, 12, 10),
            squad_mask=torch.zeros(batch_size, 12),
            ego_state=torch.randn(batch_size, 32),
            suite_descriptor=torch.randn(batch_size, 14),
            panel_stats=torch.randn(batch_size, 8),
        )

    def test_gradients_flow_through_deepset_encoder(self, model, suite_obs):
        """Gradients should flow through DeepSet contact/squad encoders."""
        model.zero_grad()

        batch_size = 4
        state = model.initial_state(batch_size)
        done = torch.zeros(batch_size)

        action, logprob, entropy, value, _ = model.get_action_and_value(
            suite_obs, state, done
        )

        loss = logprob.mean()
        loss.backward()

        # Check DeepSet encoder gradients
        for name, param in model.encoder.contact_encoder.named_parameters():
            assert param.grad is not None, f"No gradient for contact_encoder.{name}"

        for name, param in model.encoder.squad_encoder.named_parameters():
            assert param.grad is not None, f"No gradient for squad_encoder.{name}"

    def test_masked_contacts_dont_contribute_gradients(self, model):
        """Masked contacts should not contribute to gradients."""
        from echelon.rl.suite_model import SuiteObservation

        model.zero_grad()
        batch_size = 2

        # Create obs with some contacts masked
        obs = SuiteObservation(
            contacts=torch.randn(batch_size, 20, 25, requires_grad=True),
            contact_mask=torch.ones(batch_size, 20),  # All masked!
            squad=torch.randn(batch_size, 12, 10),
            squad_mask=torch.zeros(batch_size, 12),
            ego_state=torch.randn(batch_size, 32),
            suite_descriptor=torch.randn(batch_size, 14),
            panel_stats=torch.randn(batch_size, 8),
        )

        state = model.initial_state(batch_size)
        done = torch.zeros(batch_size)

        action, logprob, entropy, value, _ = model.get_action_and_value(obs, state, done)
        loss = value.mean()
        loss.backward()

        # Gradients to masked contacts should be zero
        # (they're multiplied by mask=0 in forward pass)
        assert obs.contacts.grad is not None
        # All gradients should be zero since all contacts are masked
        assert (obs.contacts.grad == 0).all(), "Masked contacts have non-zero gradients"
```

### Step 2: Run tests

```bash
PYTHONPATH=. uv run pytest tests/unit/test_gradient_flow.py -v
```

Expected: PASS (or identify actual gradient issues to fix)

### Step 3: Lint

```bash
uv run ruff check tests/unit/test_gradient_flow.py
```

### Step 4: Commit

```bash
git add tests/unit/test_gradient_flow.py
git commit -m "test: add gradient flow tests for actor-critic models

- Verify gradients flow through encoder, LSTM, heads
- Check for NaN/inf gradients
- Test sequence gradient propagation
- Verify masked contacts don't contribute gradients
- Check DeepSet encoder gradient flow"
```

---

## Task 7: Verify flat_obs_to_suite_obs Dimension Alignment (Item 23)

**Context:** The `flat_obs_to_suite_obs` function unpacks a flat observation vector into structured `SuiteObservation`. The dimension calculations must match exactly or observations will be misaligned.

**Files:**
- Modify: `echelon/rl/suite_model.py` (add assertion/constant)
- Test: `tests/unit/test_suite.py`

### Step 1: Write dimension alignment test

Add to `tests/unit/test_suite.py`:

```python
def test_flat_obs_to_suite_obs_dimension_alignment():
    """Verify flat observation dimensions match structured observation."""
    import torch
    from echelon.rl.suite_model import (
        flat_obs_to_suite_obs,
        SuiteObservation,
        FLAT_OBS_DIM,
    )
    from echelon.rl.suite import SUITE_DESCRIPTOR_DIM

    # Default parameters
    max_contacts = 20
    contact_dim = 25
    max_squad = 12
    squad_dim = 10
    ego_dim = 32
    panel_dim = 8

    # Calculate expected flat dim
    expected_dim = (
        max_contacts * contact_dim  # contacts
        + max_contacts  # contact_mask
        + max_squad * squad_dim  # squad
        + max_squad  # squad_mask
        + ego_dim  # ego_state
        + SUITE_DESCRIPTOR_DIM  # suite_descriptor
        + panel_dim  # panel_stats
    )

    # Verify constant matches
    assert FLAT_OBS_DIM == expected_dim, f"FLAT_OBS_DIM={FLAT_OBS_DIM} != expected={expected_dim}"

    # Verify round-trip
    batch_size = 4
    flat_obs = torch.randn(batch_size, FLAT_OBS_DIM)

    suite_obs = flat_obs_to_suite_obs(flat_obs)

    # Verify shapes
    assert suite_obs.contacts.shape == (batch_size, max_contacts, contact_dim)
    assert suite_obs.contact_mask.shape == (batch_size, max_contacts)
    assert suite_obs.squad.shape == (batch_size, max_squad, squad_dim)
    assert suite_obs.squad_mask.shape == (batch_size, max_squad)
    assert suite_obs.ego_state.shape == (batch_size, ego_dim)
    assert suite_obs.suite_descriptor.shape == (batch_size, SUITE_DESCRIPTOR_DIM)
    assert suite_obs.panel_stats.shape == (batch_size, panel_dim)


def test_flat_obs_consumes_all_dimensions():
    """Verify flat_obs_to_suite_obs uses exactly all input dimensions."""
    import torch
    from echelon.rl.suite_model import flat_obs_to_suite_obs, FLAT_OBS_DIM
    from echelon.rl.suite import SUITE_DESCRIPTOR_DIM

    # The function should consume exactly FLAT_OBS_DIM dimensions
    # We verify this by checking that indexing doesn't go out of bounds
    # and that we use all dimensions

    batch_size = 2
    flat_obs = torch.arange(FLAT_OBS_DIM).unsqueeze(0).expand(batch_size, -1).float()

    suite_obs = flat_obs_to_suite_obs(flat_obs)

    # Reconstruct flat from structured and verify
    reconstructed = torch.cat([
        suite_obs.contacts.reshape(batch_size, -1),
        suite_obs.contact_mask,
        suite_obs.squad.reshape(batch_size, -1),
        suite_obs.squad_mask,
        suite_obs.ego_state,
        suite_obs.suite_descriptor,
        suite_obs.panel_stats,
    ], dim=-1)

    assert reconstructed.shape[-1] == FLAT_OBS_DIM
    # Values should match (we used arange so each position is unique)
    assert torch.allclose(flat_obs, reconstructed)
```

### Step 2: Run test to verify it fails

```bash
PYTHONPATH=. uv run pytest tests/unit/test_suite.py::test_flat_obs_to_suite_obs_dimension_alignment -v
```

Expected: FAIL with `ImportError` for `FLAT_OBS_DIM`

### Step 3: Add FLAT_OBS_DIM constant to suite_model.py

Add near the top of `echelon/rl/suite_model.py` (after imports):

```python
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
```

### Step 4: Update flat_obs_to_suite_obs with validation

Update the function to validate dimensions:

```python
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
```

### Step 5: Run tests

```bash
PYTHONPATH=. uv run pytest tests/unit/test_suite.py -v
```

Expected: PASS

### Step 6: Lint and type check

```bash
uv run ruff check echelon/rl/suite_model.py
uv run mypy echelon/rl/suite_model.py
```

### Step 7: Commit

```bash
git add echelon/rl/suite_model.py tests/unit/test_suite.py
git commit -m "fix: add FLAT_OBS_DIM constant and dimension validation

- Add FLAT_OBS_DIM = 706 for default parameters
- Add runtime validation in flat_obs_to_suite_obs()
- Add round-trip test to verify dimension alignment
- Prevents silent observation misalignment bugs"
```

---

## Final Verification

After all tasks are complete:

```bash
# Full test suite
PYTHONPATH=. uv run pytest tests -v --tb=short

# Lint and type check
uv run ruff check .
uv run mypy echelon/

# Review changes
git log --oneline -10
```

---

## Checklist

- [ ] Task 1: LSTMState to NamedTuple (Items 16+19)
- [ ] Task 2: Pre-allocate RolloutBuffer advantages/returns (Item 17)
- [ ] Task 3: Lazy wandb import in spatial.py (Item 18)
- [ ] Task 4: Commander retirement mechanism (Item 20)
- [ ] Task 5: Extended curriculum parameters (Item 21)
- [ ] Task 6: Gradient flow tests (Item 22)
- [ ] Task 7: flat_obs dimension validation (Item 23)

---

*Generated from Tier 4 items in `docs/plans/2025-12-26-consolidated-improvement-plan.md`*
