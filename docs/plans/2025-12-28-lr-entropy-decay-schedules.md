# LR and Entropy Decay Schedules Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add linear decay schedules for learning rate and entropy coefficient to improve PPO training stability as opponent difficulty increases.

**Architecture:** Add schedule computation functions and CLI arguments to train_ppo.py. Decay is applied after each PPO update based on training progress (update / total_updates). Both LR and entropy decay linearly with configurable decay factors and floor values.

**Tech Stack:** PyTorch optimizer manipulation, argparse CLI

---

## Background (from DRL Specialist Review)

The training collapses when opponent lethality ramps up because:
1. **LR too conservative early** (1e-4) - slower escape from random init
2. **LR too aggressive late** - causes catastrophic forgetting during curriculum shift
3. **Fixed entropy** - needs high early (exploration) but low late (convergence)

Recommended schedule:
- LR: 3e-4 → 3e-5 (90% decay, floor at 1e-5)
- Entropy: 0.05 → 0.005 (90% decay, floor at 0.005)

---

### Task 1: Add Schedule Computation Functions

**Files:**
- Modify: `scripts/train_ppo.py:150-180` (after imports, before LRUModelCache)

**Step 1: Write the decay function**

Add after the imports section (around line 53):

```python
def compute_linear_decay(
    initial: float,
    decay_factor: float,
    progress: float,
    floor: float,
) -> float:
    """Compute linearly decayed value with floor.

    Args:
        initial: Starting value
        decay_factor: Fraction to decay by end (0.9 = decay to 10% of initial)
        progress: Training progress in [0, 1]
        floor: Minimum value (never decay below this)

    Returns:
        Decayed value: max(floor, initial * (1 - decay_factor * progress))
    """
    decayed = initial * (1.0 - decay_factor * progress)
    return max(floor, decayed)
```

**Step 2: Verify syntax**

Run: `uv run python -c "exec(open('scripts/train_ppo.py').read().split('def main')[0])"`
Expected: No syntax errors

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(training): add compute_linear_decay helper function"
```

---

### Task 2: Add CLI Arguments for Decay Schedules

**Files:**
- Modify: `scripts/train_ppo.py:561-565` (after --random-formations)

**Step 1: Add the CLI arguments**

After the `--random-formations` argument (around line 565), add:

```python
    # Learning rate schedule
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.0,
        help="LR decay factor (0.9 = decay to 10%% of initial by end). 0 disables.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-5,
        help="Minimum learning rate floor (default: 1e-5)",
    )

    # Entropy coefficient schedule
    parser.add_argument(
        "--ent-decay",
        type=float,
        default=0.0,
        help="Entropy decay factor (0.9 = decay to 10%% of initial by end). 0 disables.",
    )
    parser.add_argument(
        "--ent-min",
        type=float,
        default=0.005,
        help="Minimum entropy coefficient floor (default: 0.005)",
    )
```

**Step 2: Verify CLI parsing**

Run: `uv run python scripts/train_ppo.py --help | grep -A2 "lr-decay\|ent-decay"`
Expected: Shows all four new arguments with help text

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(training): add --lr-decay, --lr-min, --ent-decay, --ent-min CLI args"
```

---

### Task 3: Apply LR Decay in Training Loop

**Files:**
- Modify: `scripts/train_ppo.py:1441-1450` (after PPO update, before metrics extraction)

**Step 1: Add LR decay logic**

After `metrics = trainer.update(buffer, init_state)` (line 1441), add:

```python
        # Apply learning rate decay schedule
        if args.lr_decay > 0:
            progress = (update - start_update) / max(1, end_update - start_update)
            new_lr = compute_linear_decay(
                initial=args.lr,
                decay_factor=args.lr_decay,
                progress=progress,
                floor=args.lr_min,
            )
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = new_lr
```

**Step 2: Verify no syntax errors**

Run: `uv run python -c "import scripts.train_ppo"`
Expected: No errors (may show warnings, that's OK)

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(training): apply LR decay schedule after each PPO update"
```

---

### Task 4: Apply Entropy Decay in Training Loop

**Files:**
- Modify: `scripts/train_ppo.py` (immediately after LR decay block from Task 3)

**Step 1: Add entropy decay logic**

After the LR decay block, add:

```python
        # Apply entropy coefficient decay schedule
        if args.ent_decay > 0:
            progress = (update - start_update) / max(1, end_update - start_update)
            new_ent = compute_linear_decay(
                initial=args.ent_coef,
                decay_factor=args.ent_decay,
                progress=progress,
                floor=args.ent_min,
            )
            trainer.config.ent_coef = new_ent
```

**Step 2: Verify no syntax errors**

Run: `uv run python -c "import scripts.train_ppo"`
Expected: No errors

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(training): apply entropy decay schedule after each PPO update"
```

---

### Task 5: Add Entropy Coefficient to W&B Logging

**Files:**
- Modify: `scripts/train_ppo.py:1739` (in wandb_metrics dict)

**Step 1: Add entropy coefficient metric**

Find the line `"train/learning_rate": trainer.optimizer.param_groups[0]["lr"],` and add after it:

```python
                "train/entropy_coef": trainer.config.ent_coef,
```

**Step 2: Verify W&B metrics structure**

Run: `uv run ruff check scripts/train_ppo.py`
Expected: No errors

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(training): log entropy_coef to W&B for schedule monitoring"
```

---

### Task 6: Write Unit Tests for Decay Function

**Files:**
- Create: `tests/unit/training/test_schedules.py`

**Step 1: Write the test file**

```python
"""Tests for training schedule functions."""

import sys
from pathlib import Path

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))

from train_ppo import compute_linear_decay


def test_decay_at_zero_progress():
    """At progress=0, value equals initial."""
    result = compute_linear_decay(
        initial=3e-4, decay_factor=0.9, progress=0.0, floor=1e-5
    )
    assert result == 3e-4


def test_decay_at_full_progress():
    """At progress=1, value is initial * (1 - decay_factor)."""
    result = compute_linear_decay(
        initial=3e-4, decay_factor=0.9, progress=1.0, floor=1e-5
    )
    assert abs(result - 3e-5) < 1e-10  # 3e-4 * 0.1 = 3e-5


def test_decay_respects_floor():
    """Decay never goes below floor."""
    result = compute_linear_decay(
        initial=1e-4, decay_factor=0.99, progress=1.0, floor=5e-5
    )
    # Without floor: 1e-4 * 0.01 = 1e-6, but floor is 5e-5
    assert result == 5e-5


def test_decay_at_half_progress():
    """At progress=0.5, value is halfway decayed."""
    result = compute_linear_decay(
        initial=0.05, decay_factor=0.9, progress=0.5, floor=0.005
    )
    # 0.05 * (1 - 0.9 * 0.5) = 0.05 * 0.55 = 0.0275
    assert abs(result - 0.0275) < 1e-10


def test_zero_decay_factor_returns_initial():
    """With decay_factor=0, value stays at initial."""
    result = compute_linear_decay(
        initial=3e-4, decay_factor=0.0, progress=1.0, floor=1e-5
    )
    assert result == 3e-4
```

**Step 2: Run tests**

Run: `PYTHONPATH=. uv run pytest tests/unit/training/test_schedules.py -v`
Expected: All 5 tests pass

**Step 3: Commit**

```bash
git add tests/unit/training/test_schedules.py
git commit -m "test: add unit tests for compute_linear_decay schedule function"
```

---

### Task 7: Update Docstring with Schedule Examples

**Files:**
- Modify: `scripts/train_ppo.py:3-9` (docstring at top of file)

**Step 1: Update docstring**

Replace the usage section:

```python
"""PPO training script for Echelon.

Usage:
    uv run python scripts/train_ppo.py --total-steps 1000000 --num-envs 8
    uv run python scripts/train_ppo.py --size 100 --updates 200
    uv run python scripts/train_ppo.py --wandb --wandb-run-name "experiment-01"
    uv run python scripts/train_ppo.py --resume latest --total-steps 500000

    # With curriculum schedules (recommended for long runs):
    uv run python scripts/train_ppo.py --lr 3e-4 --lr-decay 0.9 --ent-coef 0.05 --ent-decay 0.9
"""
```

**Step 2: Verify**

Run: `head -12 scripts/train_ppo.py`
Expected: Shows updated docstring

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "docs: add schedule usage examples to train_ppo.py docstring"
```

---

### Task 8: Final Verification

**Files:** None (verification only)

**Step 1: Run full lint check**

Run: `uv run ruff check scripts/train_ppo.py && uv run mypy scripts/train_ppo.py --ignore-missing-imports`
Expected: No errors

**Step 2: Run all training tests**

Run: `PYTHONPATH=. uv run pytest tests/unit/training/ -v`
Expected: All tests pass

**Step 3: Dry-run with schedules enabled**

Run: `timeout 10 uv run python scripts/train_ppo.py --games 10 --lr 3e-4 --lr-decay 0.9 --ent-coef 0.05 --ent-decay 0.9 --num-envs 1 --size 30 2>&1 | head -20`
Expected: Script starts without errors, shows training progress

**Step 4: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address any issues from final verification"
```

---

## Summary

After completing all tasks, the training script supports:

```bash
# DRL Specialist recommended settings:
uv run python scripts/train_ppo.py \
    --games 2000000 \
    --lr 3e-4 --lr-decay 0.9 --lr-min 1e-5 \
    --ent-coef 0.05 --ent-decay 0.9 --ent-min 0.005 \
    --opfor-weapon-start 0.15 --opfor-ramp-updates 500 \
    --random-formations \
    --wandb
```

This gives:
- LR: 3e-4 → 3e-5 over training (10x reduction)
- Entropy: 0.05 → 0.005 over training (10x reduction)
- Both logged to W&B for monitoring
