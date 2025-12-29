# Game-Based Curriculum Ramp Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add game-based opponent curriculum ramping as an alternative to update-based ramping, making curriculum pacing robust to batch size and rollout length changes.

**Architecture:** Add `--opfor-ramp-games` CLI argument that, when set, computes curriculum progress from `episodes / opfor_ramp_games` instead of `update / opfor_ramp_updates`. This decouples curriculum from training hyperparameters.

**Tech Stack:** Python, argparse, pytest

---

## Background

Current curriculum ramp uses updates:
```python
progress = min(1.0, (update - 1) / args.opfor_ramp_updates)
```

Problem: With `--games 2000000` and `--opfor-ramp-updates 3000`, the ramp completes at ~4% of training (3000/80000 updates). The user expects the ramp to cover a meaningful fraction of training.

Solution: Allow specifying ramp in terms of games completed, which is invariant to `--num-envs`, `--rollout-steps`, etc.

---

## Task 1: Add CLI Argument

**Files:**
- Modify: `scripts/train_ppo.py:579-584`

**Step 1: Add the new argument after `--opfor-ramp-updates`**

Locate lines 579-584 in `scripts/train_ppo.py`:
```python
    parser.add_argument(
        "--opfor-ramp-updates",
        type=int,
        default=0,
        help="Updates to ramp from start to end (0 = instant full lethality)",
    )
```

Add immediately after:
```python
    parser.add_argument(
        "--opfor-ramp-games",
        type=int,
        default=0,
        help="Games to ramp from start to end (overrides --opfor-ramp-updates if set)",
    )
```

**Step 2: Verify syntax**

Run: `uv run python scripts/train_ppo.py --help | grep -A2 opfor-ramp`

Expected output should show both `--opfor-ramp-updates` and `--opfor-ramp-games`.

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(training): add --opfor-ramp-games CLI argument"
```

---

## Task 2: Update Curriculum Print Statement

**Files:**
- Modify: `scripts/train_ppo.py:807-811`

**Step 1: Update the print statement to handle both modes**

Replace lines 807-811:
```python
    if args.train_mode == "heuristic" and args.opfor_ramp_updates > 0:
        print(
            f"Opponent curriculum: weapon prob {args.opfor_weapon_start:.0%} → "
            f"{args.opfor_weapon_end:.0%} over {args.opfor_ramp_updates} updates"
        )
```

With:
```python
    if args.train_mode == "heuristic" and (args.opfor_ramp_updates > 0 or args.opfor_ramp_games > 0):
        if args.opfor_ramp_games > 0:
            ramp_desc = f"{args.opfor_ramp_games:,} games"
        else:
            ramp_desc = f"{args.opfor_ramp_updates:,} updates"
        print(
            f"Opponent curriculum: weapon prob {args.opfor_weapon_start:.0%} → "
            f"{args.opfor_weapon_end:.0%} over {ramp_desc}"
        )
```

**Step 2: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(training): update curriculum print for game-based ramp"
```

---

## Task 3: Implement Game-Based Ramp Logic

**Files:**
- Modify: `scripts/train_ppo.py:1076-1084`

**Step 1: Update the curriculum progress calculation**

Replace lines 1076-1084:
```python
        # Update opponent curriculum (heuristic mode only)
        if args.train_mode == "heuristic" and args.opfor_ramp_updates > 0:
            progress = min(1.0, (update - 1) / args.opfor_ramp_updates)
            current_weapon_prob = args.opfor_weapon_start + progress * (
                args.opfor_weapon_end - args.opfor_weapon_start
            )
            venv.set_heuristic_weapon_prob(current_weapon_prob)
        else:
            current_weapon_prob = args.opfor_weapon_end if args.train_mode == "heuristic" else 1.0
```

With:
```python
        # Update opponent curriculum (heuristic mode only)
        if args.train_mode == "heuristic":
            if args.opfor_ramp_games > 0:
                # Game-based ramp (preferred - robust to batch size changes)
                progress = min(1.0, episodes / args.opfor_ramp_games)
                current_weapon_prob = args.opfor_weapon_start + progress * (
                    args.opfor_weapon_end - args.opfor_weapon_start
                )
                venv.set_heuristic_weapon_prob(current_weapon_prob)
            elif args.opfor_ramp_updates > 0:
                # Update-based ramp (legacy)
                progress = min(1.0, (update - 1) / args.opfor_ramp_updates)
                current_weapon_prob = args.opfor_weapon_start + progress * (
                    args.opfor_weapon_end - args.opfor_weapon_start
                )
                venv.set_heuristic_weapon_prob(current_weapon_prob)
            else:
                current_weapon_prob = args.opfor_weapon_end
        else:
            current_weapon_prob = 1.0
```

**Step 2: Run lint check**

Run: `uv run ruff check scripts/train_ppo.py`

Expected: No errors

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(training): implement game-based curriculum ramp"
```

---

## Task 4: Write Unit Test for Game-Based Ramp

**Files:**
- Create: `tests/unit/training/test_curriculum_ramp.py`

**Step 1: Create the test file**

```python
"""Unit tests for curriculum ramp calculations."""

import pytest


def compute_curriculum_progress(
    episodes: int,
    update: int,
    opfor_ramp_games: int,
    opfor_ramp_updates: int,
) -> float:
    """Compute curriculum progress (extracted logic for testing).

    This mirrors the logic in train_ppo.py for curriculum ramping.
    """
    if opfor_ramp_games > 0:
        return min(1.0, episodes / opfor_ramp_games)
    elif opfor_ramp_updates > 0:
        return min(1.0, (update - 1) / opfor_ramp_updates)
    else:
        return 1.0  # No ramp = instant full difficulty


def compute_weapon_prob(
    progress: float,
    weapon_start: float,
    weapon_end: float,
) -> float:
    """Compute weapon probability from progress."""
    return weapon_start + progress * (weapon_end - weapon_start)


class TestGameBasedRamp:
    """Tests for game-based curriculum ramp."""

    def test_game_ramp_at_zero_games(self):
        """At 0 games, progress should be 0."""
        progress = compute_curriculum_progress(
            episodes=0, update=1, opfor_ramp_games=1000, opfor_ramp_updates=0
        )
        assert progress == 0.0

    def test_game_ramp_at_half(self):
        """At 500/1000 games, progress should be 0.5."""
        progress = compute_curriculum_progress(
            episodes=500, update=100, opfor_ramp_games=1000, opfor_ramp_updates=0
        )
        assert progress == 0.5

    def test_game_ramp_at_completion(self):
        """At 1000/1000 games, progress should be 1.0."""
        progress = compute_curriculum_progress(
            episodes=1000, update=500, opfor_ramp_games=1000, opfor_ramp_updates=0
        )
        assert progress == 1.0

    def test_game_ramp_caps_at_one(self):
        """Progress should cap at 1.0 even if games exceed target."""
        progress = compute_curriculum_progress(
            episodes=2000, update=1000, opfor_ramp_games=1000, opfor_ramp_updates=0
        )
        assert progress == 1.0

    def test_game_ramp_overrides_update_ramp(self):
        """Game ramp takes priority over update ramp when both set."""
        # With game ramp set, update value should be ignored
        progress = compute_curriculum_progress(
            episodes=500, update=1, opfor_ramp_games=1000, opfor_ramp_updates=100
        )
        # If update-based were used, progress would be 0.0 (update=1)
        # But game-based should give 0.5
        assert progress == 0.5


class TestUpdateBasedRamp:
    """Tests for update-based curriculum ramp (legacy)."""

    def test_update_ramp_at_start(self):
        """At update 1, progress should be 0."""
        progress = compute_curriculum_progress(
            episodes=0, update=1, opfor_ramp_games=0, opfor_ramp_updates=100
        )
        assert progress == 0.0

    def test_update_ramp_at_half(self):
        """At update 51/100, progress should be 0.5."""
        progress = compute_curriculum_progress(
            episodes=1000, update=51, opfor_ramp_games=0, opfor_ramp_updates=100
        )
        assert progress == 0.5

    def test_update_ramp_caps_at_one(self):
        """Progress should cap at 1.0."""
        progress = compute_curriculum_progress(
            episodes=5000, update=200, opfor_ramp_games=0, opfor_ramp_updates=100
        )
        assert progress == 1.0


class TestNoRamp:
    """Tests for no curriculum ramp (instant full difficulty)."""

    def test_no_ramp_returns_one(self):
        """With no ramp configured, progress should be 1.0."""
        progress = compute_curriculum_progress(
            episodes=0, update=1, opfor_ramp_games=0, opfor_ramp_updates=0
        )
        assert progress == 1.0


class TestWeaponProbCalculation:
    """Tests for weapon probability calculation from progress."""

    def test_weapon_prob_at_start(self):
        """At progress 0, should return start value."""
        prob = compute_weapon_prob(0.0, weapon_start=0.15, weapon_end=1.0)
        assert prob == 0.15

    def test_weapon_prob_at_end(self):
        """At progress 1, should return end value."""
        prob = compute_weapon_prob(1.0, weapon_start=0.15, weapon_end=1.0)
        assert prob == 1.0

    def test_weapon_prob_at_half(self):
        """At progress 0.5, should interpolate linearly."""
        prob = compute_weapon_prob(0.5, weapon_start=0.2, weapon_end=0.8)
        assert prob == pytest.approx(0.5)

    def test_weapon_prob_custom_range(self):
        """Test with custom start/end values."""
        prob = compute_weapon_prob(0.25, weapon_start=0.1, weapon_end=0.5)
        # 0.1 + 0.25 * (0.5 - 0.1) = 0.1 + 0.25 * 0.4 = 0.1 + 0.1 = 0.2
        assert prob == pytest.approx(0.2)
```

**Step 2: Run the tests**

Run: `PYTHONPATH=. uv run pytest tests/unit/training/test_curriculum_ramp.py -v`

Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/unit/training/test_curriculum_ramp.py
git commit -m "test: add unit tests for curriculum ramp calculations"
```

---

## Task 5: Update Docstring

**Files:**
- Modify: `scripts/train_ppo.py:1-50` (module docstring area)

**Step 1: Find the docstring section with schedule examples**

Look for the docstring that contains usage examples and add game-based ramp example.

Run: `grep -n "opfor-weapon-start\|Schedule usage" scripts/train_ppo.py | head -20`

**Step 2: Add example to docstring**

Find the schedule usage examples section and add:

```python
# Game-based curriculum (recommended for long training runs):
#   --opfor-weapon-start 0.15 --opfor-ramp-games 500000 --games 2000000
#   (Ramps over 25% of training, invariant to batch size)
```

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "docs: add game-based curriculum example to train_ppo.py"
```

---

## Task 6: Integration Smoke Test

**Files:**
- No files modified, just verification

**Step 1: Verify game-based ramp works end-to-end**

Run a short training with game-based ramp:
```bash
timeout 60 uv run python scripts/train_ppo.py \
    --games 100 \
    --game-length 50 \
    --packs-per-team 1 \
    --size 40 \
    --mode partial \
    --num-envs 4 \
    --opfor-weapon-start 0.1 \
    --opfor-ramp-games 50 \
    --run-dir /tmp/test-game-ramp
```

Expected: Training starts and prints `Opponent curriculum: weapon prob 10% → 100% over 50 games`

**Step 2: Verify curriculum updates during training**

Check that `curriculum/opfor_weapon_prob` increases as games complete (visible in logs or W&B).

**Step 3: Clean up test run**

```bash
rm -rf /tmp/test-game-ramp
```

---

## Task 7: Final Commit and Summary

**Step 1: Run full test suite**

Run: `PYTHONPATH=. uv run pytest tests/unit -x -q`

Expected: All tests pass

**Step 2: Run lint and type check**

Run: `uv run ruff check . && uv run mypy echelon/`

Expected: No errors

**Step 3: Create summary commit if needed**

If any fixups were made, squash or amend as appropriate.

---

## Usage After Implementation

```bash
# Game-based curriculum (recommended)
uv run python scripts/train_ppo.py \
    --games 2000000 \
    --opfor-weapon-start 0.15 \
    --opfor-ramp-games 500000 \  # Ramp over 25% of training
    ...

# Update-based curriculum (legacy, still works)
uv run python scripts/train_ppo.py \
    --games 2000000 \
    --opfor-weapon-start 0.15 \
    --opfor-ramp-updates 40000 \
    ...
```

The key advantage: `--opfor-ramp-games 500000` means "ramp over 500k games" regardless of `--num-envs`, `--rollout-steps`, or any other hyperparameter that affects updates-per-game.
