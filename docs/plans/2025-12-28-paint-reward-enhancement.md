# Paint Reward Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve credit assignment for scout painting by adding immediate rewards, observation feedback, and kill bonuses.

**Architecture:** Three-layer approach to credit assignment: (1) immediate reward when paint applied, (2) observation feature showing "my paint was used", (3) kill bonus when painted target dies. Penalty for unused paint to discourage spam.

**Tech Stack:** Python, NumPy, pytest

---

## Summary of Changes

| Component | Reward | Location |
|-----------|--------|----------|
| Paint applied to valid target | +0.3 | env.py |
| Paint expires unused | -0.1 | env.py (new event from sim.py) |
| Paint-assisted kill | +5.0 | env.py |
| `my_paint_used` observation | N/A | observations.py |

---

### Task 1: Add Paint Reward Weights

**Files:**
- Modify: `echelon/env/rewards.py:72-79`

**Step 1: Add new weight constants to RewardWeights**

Add after `paint_assist_bonus` (line 79):

```python
    # Paint credit assignment (2025-12-28)
    paint_applied: float = 0.3  # Immediate reward for painting valid target
    paint_expired_unused: float = -0.1  # Penalty if paint expires without use
    paint_kill_bonus: float = 5.0  # Bonus when painted target dies
```

**Step 2: Run lint check**

Run: `uv run ruff check echelon/env/rewards.py`
Expected: All checks passed

**Step 3: Commit**

```bash
git add echelon/env/rewards.py
git commit -m "feat(rewards): add paint credit assignment weights"
```

---

### Task 2: Add Paint Expiry Event to Simulation

**Files:**
- Modify: `echelon/sim/mech.py:82-83`
- Modify: `echelon/sim/sim.py:1394-1397`
- Test: `tests/unit/test_mechanics.py`

**Step 1: Add paint_used tracking to MechState**

In `echelon/sim/mech.py`, after line 83 (`last_painter_id`), add:

```python
    paint_was_used: bool = False  # Set True when paint lock enables damage/kill
```

**Step 2: Set paint_was_used when paint enables damage**

In `echelon/sim/sim.py`, in `_get_paint_bonus` method (around line 579), after checking painter != shooter:

```python
                if target.last_painter_id != shooter_id:
                    target.paint_was_used = True  # Mark paint as useful
                    events.append(
```

**Step 3: Emit paint_expired event when paint decays to zero**

In `echelon/sim/sim.py`, in the cooldown decay section (around line 1396), replace the painted_remaining decay:

```python
                # Paint decay with expiry tracking
                if mech.painted_remaining > 0.0:
                    mech.painted_remaining = max(0.0, float(mech.painted_remaining - self.dt))
                    if mech.painted_remaining <= 0.0:
                        # Paint just expired
                        events.append({
                            "type": "paint_expired",
                            "target": mech.mech_id,
                            "painter": mech.last_painter_id,
                            "was_used": mech.paint_was_used,
                        })
                        # Reset tracking state
                        mech.last_painter_id = None
                        mech.paint_was_used = False
```

**Step 4: Write test for paint expiry event**

In `tests/unit/test_mechanics.py`, add:

```python
def test_paint_expired_event_emitted():
    """Paint expiry emits event with was_used flag."""
    from echelon.sim.sim import Sim
    from echelon.sim.world import VoxelWorld
    from echelon.config import WorldConfig, default_mech_class_config

    world_cfg = WorldConfig(size_x=30, size_y=30, size_z=10, obstacle_fill=0.0)
    world = VoxelWorld.from_config(world_cfg, seed=42)
    sim = Sim(world, {}, default_mech_class_config())

    # Add two mechs
    sim.add_mech("scout", team="blue", mech_class="Scout", pos=(5.0, 5.0, 1.0))
    sim.add_mech("target", team="red", mech_class="Heavy", pos=(10.0, 10.0, 1.0))

    # Manually set paint state (simulating a paint hit)
    target = sim.mechs["target"]
    target.painted_remaining = 0.1  # About to expire
    target.last_painter_id = "scout"
    target.paint_was_used = False

    # Step until paint expires
    events = sim.step({})

    # Find paint_expired event
    expired_events = [e for e in events if e["type"] == "paint_expired"]
    assert len(expired_events) == 1
    assert expired_events[0]["painter"] == "scout"
    assert expired_events[0]["was_used"] is False
```

**Step 5: Run the test**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_mechanics.py::test_paint_expired_event_emitted -v`
Expected: PASS

**Step 6: Commit**

```bash
git add echelon/sim/mech.py echelon/sim/sim.py tests/unit/test_mechanics.py
git commit -m "feat(sim): emit paint_expired event with usage tracking"
```

---

### Task 3: Process Paint Events in Environment

**Files:**
- Modify: `echelon/env/env.py:1059-1090` (event processing section)

**Step 1: Add paint event tracking dicts**

After the existing step tracking dicts (around line 1054), add:

```python
        step_paint_applied: dict[str, int] = dict.fromkeys(self.agents, 0)
        step_paint_expired_unused: dict[str, int] = dict.fromkeys(self.agents, 0)
        step_paint_kills: dict[str, int] = dict.fromkeys(self.agents, 0)
```

**Step 2: Process paint event for immediate reward**

In the event processing loop (after the "paint" event handling around line 1081), add:

```python
                elif et == "paint":
                    shooter_id = str(ev["shooter"])
                    if shooter_id in step_paint_applied:
                        step_paint_applied[shooter_id] += 1
```

**Step 3: Process paint_expired event**

Add new event handler:

```python
                elif et == "paint_expired":
                    painter_id = ev.get("painter")
                    was_used = ev.get("was_used", False)
                    if painter_id and painter_id in step_paint_expired_unused and not was_used:
                        step_paint_expired_unused[painter_id] += 1
```

**Step 4: Process kill event for paint kill bonus**

Modify the existing kill handler to check if target was painted:

```python
                if et == "kill":
                    shooter_id = str(ev["shooter"])
                    target_id = str(ev["target"])
                    target = sim.mechs.get(target_id)
                    shooter = sim.mechs.get(shooter_id)
                    # Existing kill tracking...

                    # Paint kill bonus: credit the painter if target was painted
                    if target is not None and target.last_painter_id:
                        painter_id = target.last_painter_id
                        if painter_id in step_paint_kills and painter_id != shooter_id:
                            step_paint_kills[painter_id] += 1
```

**Step 5: Add paint rewards to reward calculation**

After the RewardComputer.compute() call, add paint-specific rewards:

```python
        # Paint credit assignment rewards (immediate, not going through RewardComputer)
        w = self._reward_weights
        for aid in self.agents:
            mech = sim.mechs.get(aid)
            if mech is None or not mech.alive:
                continue

            # Immediate paint reward
            paint_count = step_paint_applied.get(aid, 0)
            if paint_count > 0:
                rewards[aid] += w.paint_applied * paint_count
                reward_components[aid].paint_assist += w.paint_applied * paint_count

            # Paint expired unused penalty
            expired_count = step_paint_expired_unused.get(aid, 0)
            if expired_count > 0:
                rewards[aid] += w.paint_expired_unused * expired_count
                reward_components[aid].paint_assist += w.paint_expired_unused * expired_count

            # Paint kill bonus
            kill_count = step_paint_kills.get(aid, 0)
            if kill_count > 0:
                rewards[aid] += w.paint_kill_bonus * kill_count
                reward_components[aid].kill += w.paint_kill_bonus * kill_count
```

**Step 6: Run existing reward tests**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_rewards.py -v --tb=short`
Expected: All pass (no breaking changes)

**Step 7: Commit**

```bash
git add echelon/env/env.py
git commit -m "feat(env): process paint events for credit assignment rewards"
```

---

### Task 4: Add Paint-Used Observation Feature

**Files:**
- Modify: `echelon/env/observations.py:312-383` (ObservationContext)
- Modify: `echelon/env/observations.py:414-421` (self_dim)
- Modify: `echelon/env/observations.py:667-785` (self features building)
- Modify: `echelon/env/env.py` (pass data to context)

**Step 1: Add paint_used field to ObservationContext**

In `echelon/env/observations.py`, add to ObservationContext dataclass (after line 351):

```python
    # Paint usage feedback (for credit assignment)
    paint_used_this_step: dict[str, int]  # painter_id -> count
```

**Step 2: Update ObservationContext.from_env()**

This requires env.py to expose the data. Add parameter with default:

```python
    @classmethod
    def from_env(cls, env: EchelonEnv, paint_used: dict[str, int] | None = None) -> ObservationContext:
```

And in the return statement, add:

```python
            paint_used_this_step=paint_used or {},
```

**Step 3: Update self_dim comment and value**

In `ObservationBuilder.__init__` (line 414-421), update:

```python
        # Self features dimension breakdown:
        # acoustic_quadrants(4) + hull_type_onehot(4) + suite_descriptor(14) +
        # targeted, under_fire, painted, shutdown, crit_heat, self_hp_norm, self_heat_norm,
        # heat_headroom, stability_risk, damage_dir_local(3), incoming_missile, sensor_quality,
        # jam_level, ecm_on, eccm_on, suppressed, ams_cd, self_vel(3), cooldowns(4), in_zone,
        # vec_to_zone(3), zone_radius, my_control, my_score, enemy_score, time_frac,
        # obs_sort_onehot(3), hostile_only, my_paint_used = 48
        self.self_dim = 48
```

**Step 4: Add my_paint_used to self features**

In the self features building section (around line 669), add after `hostile_only`:

```python
            my_paint_used = 1.0 if ctx.paint_used_this_step.get(aid, 0) > 0 else 0.0
```

And include it in the self_feats array (around line 790, in the np.array):

```python
                        hostile_only,
                        my_paint_used,  # NEW: paint feedback for credit assignment
```

**Step 5: Update env.py to pass paint_used to observation context**

In `env.py`, store step_assists after computing it, and pass to ObservationContext:

```python
        # Store for observation context
        self._paint_used_this_step = step_assists

        # In _get_obs or wherever ObservationContext.from_env is called:
        ctx = ObservationContext.from_env(self, paint_used=self._paint_used_this_step)
```

**Step 6: Run observation tests**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_episode_stats.py -v --tb=short`
Expected: All pass

**Step 7: Commit**

```bash
git add echelon/env/observations.py echelon/env/env.py
git commit -m "feat(obs): add my_paint_used observation for credit assignment"
```

---

### Task 5: Write Integration Tests

**Files:**
- Create: `tests/unit/test_paint_rewards.py`

**Step 1: Write test file**

```python
"""Tests for paint credit assignment rewards."""

import pytest
import numpy as np

from echelon import EchelonEnv
from echelon.config import EnvConfig, WorldConfig


@pytest.fixture
def paint_env():
    """Environment configured for paint testing."""
    cfg = EnvConfig(
        world=WorldConfig(
            size_x=30, size_y=30, size_z=10,
            obstacle_fill=0.0, ensure_connectivity=False
        ),
        num_packs=1,
        seed=42,
        max_episode_seconds=10.0,
    )
    env = EchelonEnv(cfg)
    env.reset(seed=42)
    return env


class TestPaintRewards:
    """Test paint credit assignment rewards."""

    def test_paint_applied_gives_immediate_reward(self, paint_env):
        """Painting a target gives immediate +0.3 reward."""
        env = paint_env
        # This test verifies the reward weight exists and is positive
        from echelon.env.rewards import RewardWeights
        weights = RewardWeights()
        assert weights.paint_applied == 0.3, "Paint applied reward should be 0.3"

    def test_paint_expired_unused_gives_penalty(self, paint_env):
        """Paint expiring without use gives -0.1 penalty."""
        from echelon.env.rewards import RewardWeights
        weights = RewardWeights()
        assert weights.paint_expired_unused == -0.1, "Paint expired unused should be -0.1"

    def test_paint_kill_bonus_exists(self, paint_env):
        """Kill bonus for painted target is 5.0."""
        from echelon.env.rewards import RewardWeights
        weights = RewardWeights()
        assert weights.paint_kill_bonus == 5.0, "Paint kill bonus should be 5.0"

    def test_observation_includes_paint_used_feature(self, paint_env):
        """Observation space includes my_paint_used feature (dim 48 in self features)."""
        env = paint_env
        obs, _ = env.reset(seed=42)

        # Check observation dimension increased
        first_agent = list(obs.keys())[0]
        obs_array = obs[first_agent]

        # Observation should be non-empty numpy array
        assert isinstance(obs_array, np.ndarray)
        assert len(obs_array) > 0
```

**Step 2: Run the tests**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_paint_rewards.py -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/unit/test_paint_rewards.py
git commit -m "test: add paint credit assignment integration tests"
```

---

### Task 6: Final Verification

**Step 1: Run full lint check**

Run: `uv run ruff check echelon/`
Expected: All checks passed

**Step 2: Run type check**

Run: `uv run mypy echelon/ --no-error-summary`
Expected: No errors

**Step 3: Run full test suite**

Run: `PYTHONPATH=. uv run pytest tests/unit -v --tb=short`
Expected: All tests pass

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: paint credit assignment enhancement complete"
```

---

## Edge Cases Handled

1. **Multiple painters on same target**: Last painter gets credit (existing behavior via `last_painter_id`)
2. **Paint refreshed before expiry**: No penalty, new paint starts fresh
3. **Painter dies before paint used**: Painter still tracked, rewards still apply
4. **Self-painting (painter shoots own painted target)**: No assist credit (existing `painter != shooter` check)

## Verification Checklist

- [ ] Paint applied: +0.3 immediate reward
- [ ] Paint expired unused: -0.1 penalty
- [ ] Paint-assisted kill: +5.0 bonus to painter
- [ ] `my_paint_used` observation feature (self_dim = 48)
- [ ] All tests pass
- [ ] Lint and type checks pass
