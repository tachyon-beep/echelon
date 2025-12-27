# Self-Play with Lieutenant Heuristic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable self-play training where the policy trains against past versions of itself plus a permanent heuristic baseline ("Lieutenant Heuristic").

**Architecture:** Add `kind="heuristic"` as a new LeagueEntry type. When the arena samples the heuristic as opponent, use `venv.get_heuristic_actions()` instead of model inference. The heuristic has a normal Glicko-2 rating that updates from matches, but never retires.

**Tech Stack:** Python, PyTorch, existing arena infrastructure (Glicko-2, PFSP sampling, LRU cache)

---

## Task 1: Add Heuristic Entry Support to League

**Files:**
- Modify: `echelon/arena/league.py:176-179` (top_commanders)
- Modify: `echelon/arena/league.py:312-356` (retire_commanders)
- Modify: `echelon/arena/league.py` (add new method)
- Test: `tests/unit/test_arena_league.py`

### Step 1: Write failing test for heuristic entry creation

Add to `tests/unit/test_arena_league.py`:

```python
class TestHeuristicEntry:
    """Test heuristic baseline support."""

    def test_add_heuristic_creates_entry(self):
        """add_heuristic creates a heuristic entry with fixed ID."""
        league = League()
        entry = league.add_heuristic()

        assert entry.entry_id == "heuristic"
        assert entry.kind == "heuristic"
        assert entry.commander_name == "Lieutenant Heuristic"
        assert entry.ckpt_path == ""
        assert entry.rating.rating == league.cfg.rating0

    def test_add_heuristic_idempotent(self):
        """Calling add_heuristic twice returns same entry."""
        league = League()
        entry1 = league.add_heuristic()
        entry1.rating = Glicko2Rating(rating=1600.0, rd=100.0, vol=0.05)
        entry2 = league.add_heuristic()

        assert entry1 is entry2
        assert entry2.rating.rating == 1600.0  # Preserved

    def test_heuristic_included_in_top_commanders(self):
        """top_commanders includes heuristic entries."""
        league = League()
        league.add_heuristic()

        top = league.top_commanders(10)
        assert len(top) == 1
        assert top[0].kind == "heuristic"

    def test_heuristic_never_retired(self):
        """retire_commanders skips heuristic entries."""
        league = League()
        heur = league.add_heuristic()
        heur.games = 100  # Would normally qualify for retirement
        heur.rating = Glicko2Rating(rating=800.0, rd=50.0, vol=0.05)  # Low rating

        # Add some regular commanders
        for i in range(25):
            entry = LeagueEntry(
                entry_id=f"cmd_{i}",
                ckpt_path=f"/fake/path_{i}.pt",
                kind="commander",
                commander_name=f"Commander {i}",
                rating=Glicko2Rating(rating=1500.0 + i * 10, rd=50.0, vol=0.05),
                games=50,
            )
            league.entries[entry.entry_id] = entry

        retired = league.retire_commanders(keep_top=10, min_games=20)

        # Heuristic should NOT be retired even though it has lowest rating
        assert heur.kind == "heuristic"
        assert "heuristic" not in [e.entry_id for e in retired]

    def test_heuristic_serialization_roundtrip(self):
        """Heuristic entry survives save/load cycle."""
        import tempfile
        from pathlib import Path

        league = League()
        league.add_heuristic()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "league.json"
            league.save(path)
            loaded = League.load(path)

        assert "heuristic" in loaded.entries
        entry = loaded.entries["heuristic"]
        assert entry.kind == "heuristic"
        assert entry.commander_name == "Lieutenant Heuristic"
```

### Step 2: Run test to verify it fails

```bash
PYTHONPATH=. uv run pytest tests/unit/test_arena_league.py::TestHeuristicEntry -v
```

Expected: FAIL with `AttributeError: 'League' object has no attribute 'add_heuristic'`

### Step 3: Implement add_heuristic method

Add to `echelon/arena/league.py` after `upsert_checkpoint` method (around line 175):

```python
    def add_heuristic(self) -> LeagueEntry:
        """Add or get the permanent heuristic baseline entry.

        The heuristic ("Lieutenant Heuristic") is a special commander that:
        - Always exists in the pool
        - Never retires
        - Uses venv.get_heuristic_actions() instead of model inference
        - Has normal Glicko-2 rating that updates from matches

        Returns:
            The heuristic LeagueEntry (created or existing)
        """
        entry_id = "heuristic"
        existing = self.entries.get(entry_id)
        if existing is not None:
            return existing

        entry = LeagueEntry(
            entry_id=entry_id,
            ckpt_path="",
            kind="heuristic",
            commander_name="Lieutenant Heuristic",
            rating=Glicko2Rating(
                rating=float(self.cfg.rating0),
                rd=float(self.cfg.rd0),
                vol=float(self.cfg.vol0),
            ),
        )
        self.entries[entry_id] = entry
        return entry
```

### Step 4: Update top_commanders to include heuristic

Modify `echelon/arena/league.py:176-179`:

```python
    def top_commanders(self, n: int) -> list[LeagueEntry]:
        # Include both commanders and heuristic entries
        commanders = [e for e in self.entries.values() if e.kind in ("commander", "heuristic")]
        commanders.sort(key=lambda e: float(e.rating.rating), reverse=True)
        return commanders[: max(0, int(n))]
```

### Step 5: Update retire_commanders to skip heuristic

Modify `echelon/arena/league.py:330` - change the filter:

```python
        commanders = [e for e in self.entries.values() if e.kind == "commander"]
```

This already excludes heuristic (kind="heuristic"), so no change needed. But add a comment for clarity:

```python
        # Only retire regular commanders - heuristic entries are permanent
        commanders = [e for e in self.entries.values() if e.kind == "commander"]
```

### Step 6: Run tests to verify they pass

```bash
PYTHONPATH=. uv run pytest tests/unit/test_arena_league.py::TestHeuristicEntry -v
```

Expected: All 5 tests PASS

### Step 7: Run full league test suite

```bash
PYTHONPATH=. uv run pytest tests/unit/test_arena_league.py -v
```

Expected: All tests PASS

### Step 8: Commit

```bash
git add echelon/arena/league.py tests/unit/test_arena_league.py
git commit -m "feat(arena): add Lieutenant Heuristic as permanent baseline

Add kind='heuristic' as new LeagueEntry type for the permanent baseline:
- add_heuristic() creates/gets the heuristic entry
- top_commanders() includes heuristic entries in pool
- retire_commanders() never retires heuristic entries
- Normal Glicko-2 rating that updates from matches

Lieutenant Heuristic provides a stable training signal as policies
learn to beat themselves."
```

---

## Task 2: Add Bootstrap Command to Arena CLI

**Files:**
- Modify: `scripts/arena.py:255-289` (add new subcommand)
- Test: Manual CLI test

### Step 1: Add cmd_bootstrap function

Add to `scripts/arena.py` after `cmd_add` (around line 98):

```python
def cmd_bootstrap(args: argparse.Namespace) -> None:
    """Bootstrap a new league with Lieutenant Heuristic."""
    league_path = Path(args.league)

    if league_path.exists() and not args.force:
        raise SystemExit(f"League already exists: {league_path} (use --force to overwrite)")

    league = League()
    heuristic = league.add_heuristic()
    league.save(league_path)

    print(f"ok: bootstrapped {league_path}")
    print(f"  - {heuristic.commander_name} (rating={heuristic.rating.rating:.0f})")
```

### Step 2: Add bootstrap subparser

Add to `scripts/arena.py` in `main()` after `p_init` (around line 263):

```python
    p_boot = sub.add_parser("bootstrap", help="Initialize league with Lieutenant Heuristic")
    p_boot.add_argument("--force", action="store_true", help="Overwrite existing league")
    p_boot.set_defaults(func=cmd_bootstrap)
```

### Step 3: Test the command

```bash
# Create a test league
uv run python scripts/arena.py --league /tmp/test_league.json bootstrap

# Verify it works
uv run python scripts/arena.py --league /tmp/test_league.json list
```

Expected output:
```
ok: bootstrapped /tmp/test_league.json
  - Lieutenant Heuristic (rating=1500)
```

### Step 4: Commit

```bash
git add scripts/arena.py
git commit -m "feat(arena): add bootstrap command for Lieutenant Heuristic

New CLI command: arena.py bootstrap
- Creates new league with Lieutenant Heuristic as initial commander
- Use --force to overwrite existing league"
```

---

## Task 3: Wire Heuristic Detection in Training Loop

**Files:**
- Modify: `scripts/train_ppo.py:955-968` (arena_sample_opponent)
- Modify: `scripts/train_ppo.py:1074-1091` (red team action selection)

### Step 1: Add heuristic detection flag

Modify `scripts/train_ppo.py` - add a flag to track if current opponent is heuristic. Find the arena variables (around line 923):

```python
    arena_opponent_id: str | None = None
    arena_opponent: ActorCriticLSTM | None = None
    arena_opponent_is_heuristic: bool = False  # NEW
    arena_lstm_state = None
```

### Step 2: Update arena_sample_opponent to detect heuristic

Modify `scripts/train_ppo.py:955-968`:

```python
    def arena_sample_opponent(reset_hidden: bool) -> None:
        nonlocal arena_opponent_id, arena_opponent, arena_lstm_state, arena_done, arena_opponent_is_heuristic
        entry = arena_rng.choice(arena_pool)
        arena_opponent_id = entry.entry_id

        # Check if this is the heuristic baseline
        if entry.kind == "heuristic":
            arena_opponent_is_heuristic = True
            arena_opponent = None  # No model to load
            # Still reset LSTM state tracking for consistency
            if reset_hidden:
                arena_lstm_state = None
                arena_done = torch.ones(num_envs * len(red_ids), device=opponent_device)
            return

        arena_opponent_is_heuristic = False
        cached = arena_cache.get(arena_opponent_id)
        if cached is None:
            cached = arena_load_model(entry.ckpt_path)
            arena_cache.put(arena_opponent_id, cached)
        arena_opponent = cached
        if reset_hidden:
            arena_lstm_state = arena_opponent.initial_state(
                batch_size=num_envs * len(red_ids), device=opponent_device
            )
            arena_done = torch.ones(num_envs * len(red_ids), device=opponent_device)
```

### Step 3: Update red team action selection to use heuristic when appropriate

Modify `scripts/train_ppo.py:1073-1091`. The current code:

```python
            # Red team (opponent)
            if args.train_mode == "arena":
                obs_r_many = stack_obs_many(next_obs_dicts, red_ids)
                ...
            elif args.train_mode == "heuristic":
                heuristic_acts_list = venv.get_heuristic_actions(red_ids)
                ...
```

Change to:

```python
            # Red team (opponent)
            if args.train_mode == "arena":
                if arena_opponent_is_heuristic:
                    # Lieutenant Heuristic - use heuristic actions
                    heuristic_acts_list = venv.get_heuristic_actions(red_ids)
                    for env_idx in range(num_envs):
                        all_actions_dicts[env_idx].update(heuristic_acts_list[env_idx])
                else:
                    # Neural network opponent - use model inference
                    obs_r_many = stack_obs_many(next_obs_dicts, red_ids)
                    obs_r_torch = torch.from_numpy(obs_r_many).to(opponent_device)
                    with torch.no_grad():
                        assert arena_opponent is not None
                        assert arena_lstm_state is not None
                        act_r, _, _, _, arena_lstm_state = arena_opponent.get_action_and_value(
                            obs_r_torch, arena_lstm_state, arena_done
                        )
                    act_r_np = act_r.detach().cpu().numpy()
                    for env_idx in range(num_envs):
                        for i, rid in enumerate(red_ids):
                            all_actions_dicts[env_idx][rid] = act_r_np[env_idx * len(red_ids) + i]
            elif args.train_mode == "heuristic":
                heuristic_acts_list = venv.get_heuristic_actions(red_ids)
                for env_idx in range(num_envs):
                    all_actions_dicts[env_idx].update(heuristic_acts_list[env_idx])
```

### Step 4: Test with a smoke run

```bash
# Bootstrap the league
uv run python scripts/arena.py --league runs/arena/league.json bootstrap --force

# Run a short training in arena mode
uv run python scripts/train_ppo.py \
    --train-mode arena \
    --arena-league runs/arena/league.json \
    --updates 2 \
    --rollout-steps 64 \
    --packs-per-team 1 \
    --size 50
```

Expected: Training runs without errors, using Lieutenant Heuristic as opponent.

### Step 5: Commit

```bash
git add scripts/train_ppo.py
git commit -m "feat(train): wire heuristic detection in arena mode

When arena samples Lieutenant Heuristic as opponent:
- Sets arena_opponent_is_heuristic flag
- Uses venv.get_heuristic_actions() instead of model inference
- No model loaded (arena_opponent = None)

This allows self-play training to include the heuristic baseline."
```

---

## Task 4: Add Opponent Logging

**Files:**
- Modify: `scripts/train_ppo.py` (status line and logging)

### Step 1: Add opponent info to status line

Find the status line (around line 1445):

```python
        opfor_str = f" | opfor {current_weapon_prob:.0%}" if args.train_mode == "heuristic" else ""
```

Change to:

```python
        if args.train_mode == "heuristic":
            opfor_str = f" | opfor {current_weapon_prob:.0%}"
        elif args.train_mode == "arena" and arena_opponent_id:
            # Show current arena opponent
            opp_name = "Lt. Heuristic" if arena_opponent_is_heuristic else arena_opponent_id[:12]
            opfor_str = f" | vs {opp_name}"
        else:
            opfor_str = ""
```

### Step 2: Test the logging

```bash
uv run python scripts/train_ppo.py \
    --train-mode arena \
    --arena-league runs/arena/league.json \
    --updates 2 \
    --rollout-steps 64 \
    --packs-per-team 1 \
    --size 50
```

Expected: Status line shows `| vs Lt. Heuristic`

### Step 3: Commit

```bash
git add scripts/train_ppo.py
git commit -m "feat(train): show arena opponent in status line

Arena mode now displays current opponent:
- 'vs Lt. Heuristic' for heuristic baseline
- 'vs <entry_id>' for neural network opponents"
```

---

## Task 5: Integration Test

**Files:**
- Create: `tests/integration/test_arena_selfplay.py`

### Step 1: Write integration test

```python
"""Integration test for arena self-play with Lieutenant Heuristic."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from echelon.arena.league import League
from echelon.config import EnvConfig, WorldConfig
from echelon.env import EchelonEnv


class TestArenaSelfPlay:
    """Test arena self-play integration."""

    @pytest.fixture
    def league_with_heuristic(self) -> tuple[Path, League]:
        """Create a temporary league with Lieutenant Heuristic."""
        tmpdir = tempfile.mkdtemp()
        league_path = Path(tmpdir) / "league.json"

        league = League()
        league.add_heuristic()
        league.save(league_path)

        return league_path, league

    def test_heuristic_in_arena_pool(self, league_with_heuristic):
        """Lieutenant Heuristic appears in arena pool."""
        league_path, league = league_with_heuristic

        # Reload to verify persistence
        loaded = League.load(league_path)

        pool = loaded.top_commanders(10) + loaded.recent_candidates(5)
        assert len(pool) == 1
        assert pool[0].entry_id == "heuristic"
        assert pool[0].kind == "heuristic"
        assert pool[0].commander_name == "Lieutenant Heuristic"

    def test_heuristic_rating_updates(self, league_with_heuristic):
        """Heuristic rating can be updated from match results."""
        from echelon.arena.glicko2 import GameResult, Glicko2Rating

        league_path, league = league_with_heuristic
        heuristic = league.entries["heuristic"]
        initial_rating = heuristic.rating.rating

        # Simulate heuristic losing a match
        fake_opponent_rating = Glicko2Rating(rating=1500.0, rd=100.0, vol=0.06)
        results = {
            "heuristic": [GameResult(opponent=fake_opponent_rating, score=0.0)]  # Loss
        }
        league.apply_rating_period(results)

        # Rating should decrease after a loss
        assert heuristic.rating.rating < initial_rating
        assert heuristic.games == 1
```

### Step 2: Run integration test

```bash
PYTHONPATH=. uv run pytest tests/integration/test_arena_selfplay.py -v
```

Expected: All tests PASS

### Step 3: Commit

```bash
git add tests/integration/test_arena_selfplay.py
git commit -m "test(arena): add integration tests for self-play

Tests Lieutenant Heuristic:
- Appears in arena pool after bootstrap
- Rating updates from match results"
```

---

## Task 6: Documentation Update

**Files:**
- Modify: `CLAUDE.md` (add arena mode docs)

### Step 1: Add arena mode documentation

Add to `CLAUDE.md` under the Training section:

```markdown
### Arena Self-Play Training

Train against past versions of the policy plus a permanent heuristic baseline:

```bash
# Bootstrap league with Lieutenant Heuristic
uv run python scripts/arena.py bootstrap

# Train in arena mode
uv run python scripts/train_ppo.py --train-mode arena --arena-league runs/arena/league.json

# Add trained checkpoints to league
uv run python scripts/arena.py add runs/train/best.pt --kind commander
```

The arena uses PFSP (Prioritized Fictitious Self-Play) to sample opponents by rating similarity.
Lieutenant Heuristic provides a stable baseline that never retires.
```

### Step 2: Commit

```bash
git add CLAUDE.md
git commit -m "docs: add arena self-play training instructions"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add heuristic entry support to League | league.py, test_arena_league.py |
| 2 | Add bootstrap CLI command | arena.py |
| 3 | Wire heuristic detection in training loop | train_ppo.py |
| 4 | Add opponent logging | train_ppo.py |
| 5 | Integration test | test_arena_selfplay.py |
| 6 | Documentation | CLAUDE.md |

**Total: 6 tasks, ~200 lines of code changes**
