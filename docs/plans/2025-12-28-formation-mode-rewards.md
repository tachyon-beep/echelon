# Formation Mode Reward Modulation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train individual mech policies to respond to formation commands (close/standard/loose) that modulate zone reward importance, preparing for future commander training.

**Architecture:** Add FormationMode enum with reward multipliers. Pass formation mode through EnvConfig, expose in observations as one-hot, and apply multipliers during reward computation. Training randomly samples formation modes so policies learn all three postures.

**Tech Stack:** Python dataclasses, NumPy, existing reward/observation infrastructure

---

## Background

Current training shows zone rewards dominating (51% of return). This is correct for "hold objective" scenarios, but we need policies that can also:
- **CLOSE**: Tight zone control (heavies holding)
- **STANDARD**: Balanced posture (current behavior)
- **LOOSE**: Maneuver freedom (scouts flanking, lights harassing)

This is Phase 1 of hierarchical RL: train subordinates to be *responsive* to commands before training a commander to *issue* them.

---

### Task 1: Add FormationMode Enum and Multipliers

**Files:**
- Modify: `echelon/env/rewards.py:1-30`
- Test: `tests/unit/test_rewards.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_rewards.py after imports

class TestFormationModeMultipliers:
    """Verify formation mode reward multipliers."""

    def test_formation_mode_enum_values(self):
        """FormationMode enum has expected values."""
        from echelon.env.rewards import FormationMode

        assert FormationMode.CLOSE == 0
        assert FormationMode.STANDARD == 1
        assert FormationMode.LOOSE == 2

    def test_formation_multipliers_exist(self):
        """Each formation mode has multipliers defined."""
        from echelon.env.rewards import FORMATION_MULTIPLIERS, FormationMode

        for mode in FormationMode:
            assert mode in FORMATION_MULTIPLIERS
            mult = FORMATION_MULTIPLIERS[mode]
            assert "zone" in mult
            assert "out_death" in mult
            assert "approach" in mult

    def test_close_amplifies_zone_rewards(self):
        """CLOSE mode has zone multiplier > 1."""
        from echelon.env.rewards import FORMATION_MULTIPLIERS, FormationMode

        assert FORMATION_MULTIPLIERS[FormationMode.CLOSE]["zone"] > 1.0
        assert FORMATION_MULTIPLIERS[FormationMode.CLOSE]["out_death"] > 1.0

    def test_loose_reduces_zone_rewards(self):
        """LOOSE mode has zone multiplier < 1."""
        from echelon.env.rewards import FORMATION_MULTIPLIERS, FormationMode

        assert FORMATION_MULTIPLIERS[FormationMode.LOOSE]["zone"] < 1.0
        assert FORMATION_MULTIPLIERS[FormationMode.LOOSE]["out_death"] < 1.0

    def test_standard_is_neutral(self):
        """STANDARD mode has multipliers of 1.0."""
        from echelon.env.rewards import FORMATION_MULTIPLIERS, FormationMode

        mult = FORMATION_MULTIPLIERS[FormationMode.STANDARD]
        assert mult["zone"] == 1.0
        assert mult["out_death"] == 1.0
        assert mult["approach"] == 1.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_rewards.py::TestFormationModeMultipliers -v`
Expected: FAIL with "cannot import name 'FormationMode'"

**Step 3: Write minimal implementation**

```python
# Add to echelon/env/rewards.py after imports, before RewardWeights

from enum import IntEnum


class FormationMode(IntEnum):
    """Formation posture commanded by squad leader.

    Modulates reward weights to train policies that respond to tactical commands:
    - CLOSE: Tight zone control, penalize straying
    - STANDARD: Balanced posture (default)
    - LOOSE: Maneuver freedom, reduced zone pressure
    """

    CLOSE = 0
    STANDARD = 1
    LOOSE = 2


# Reward multipliers per formation mode
# Applied to: zone_tick, out_zone_death_mult, approach
FORMATION_MULTIPLIERS: dict[FormationMode, dict[str, float]] = {
    FormationMode.CLOSE: {"zone": 2.0, "out_death": 2.5, "approach": 1.5},
    FormationMode.STANDARD: {"zone": 1.0, "out_death": 1.0, "approach": 1.0},
    FormationMode.LOOSE: {"zone": 0.3, "out_death": 0.3, "approach": 0.5},
}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_rewards.py::TestFormationModeMultipliers -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add echelon/env/rewards.py tests/unit/test_rewards.py
git commit -m "feat(rewards): add FormationMode enum and multipliers"
```

---

### Task 2: Add Formation Mode to EnvConfig

**Files:**
- Modify: `echelon/config.py`
- Modify: `echelon/env/env.py`
- Test: `tests/unit/test_rewards.py`

**Step 1: Write the failing test**

```python
# Add to TestFormationModeMultipliers class

def test_env_config_has_formation_mode(self):
    """EnvConfig accepts formation_mode parameter."""
    from echelon.config import EnvConfig, WorldConfig
    from echelon.env.rewards import FormationMode

    cfg = EnvConfig(
        world=WorldConfig(size_x=30, size_y=30, size_z=10),
        formation_mode=FormationMode.CLOSE,
    )
    assert cfg.formation_mode == FormationMode.CLOSE

def test_env_config_defaults_to_standard(self):
    """EnvConfig defaults to STANDARD formation."""
    from echelon.config import EnvConfig, WorldConfig
    from echelon.env.rewards import FormationMode

    cfg = EnvConfig(world=WorldConfig(size_x=30, size_y=30, size_z=10))
    assert cfg.formation_mode == FormationMode.STANDARD
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_rewards.py::TestFormationModeMultipliers::test_env_config_has_formation_mode -v`
Expected: FAIL with "unexpected keyword argument 'formation_mode'"

**Step 3: Write minimal implementation**

```python
# In echelon/config.py, add import at top
from echelon.env.rewards import FormationMode

# In EnvConfig dataclass, add field (after existing fields, before methods)
    formation_mode: FormationMode = FormationMode.STANDARD
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_rewards.py::TestFormationModeMultipliers::test_env_config_has_formation_mode tests/unit/test_rewards.py::TestFormationModeMultipliers::test_env_config_defaults_to_standard -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add echelon/config.py tests/unit/test_rewards.py
git commit -m "feat(config): add formation_mode to EnvConfig"
```

---

### Task 3: Add Formation Mode to Observations

**Files:**
- Modify: `echelon/env/observations.py`
- Test: `tests/unit/test_observations.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_observations.py

class TestFormationModeObservation:
    """Verify formation mode appears in observations."""

    def test_formation_mode_in_observation(self, obs_env):
        """Formation mode one-hot is in self-features."""
        from echelon.env.rewards import FormationMode

        env = obs_env
        # Default is STANDARD
        env.reset(seed=0)

        obs = env._obs()
        scout_obs = obs["blue_0"]

        # self_dim increased by 3 (one-hot for formation)
        # Formation mode is in self-features, after my_paint_used
        self_features = scout_obs[-51:]  # 48 + 3 = 51

        # STANDARD = [0, 1, 0]
        formation_one_hot = self_features[-3:]
        assert formation_one_hot[0] == 0.0  # not CLOSE
        assert formation_one_hot[1] == 1.0  # STANDARD
        assert formation_one_hot[2] == 0.0  # not LOOSE

    def test_formation_mode_close_encoding(self, obs_env):
        """CLOSE formation encodes as [1, 0, 0]."""
        from echelon.config import EnvConfig, WorldConfig
        from echelon.env.rewards import FormationMode
        from echelon import EchelonEnv

        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=0,
            formation_mode=FormationMode.CLOSE,
        )
        env = EchelonEnv(cfg)
        env.reset(seed=0)

        obs = env._obs()
        scout_obs = obs["blue_0"]
        formation_one_hot = scout_obs[-3:]

        assert formation_one_hot[0] == 1.0  # CLOSE
        assert formation_one_hot[1] == 0.0
        assert formation_one_hot[2] == 0.0

    def test_formation_mode_loose_encoding(self, obs_env):
        """LOOSE formation encodes as [0, 0, 1]."""
        from echelon.config import EnvConfig, WorldConfig
        from echelon.env.rewards import FormationMode
        from echelon import EchelonEnv

        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=15),
            num_packs=1,
            seed=0,
            formation_mode=FormationMode.LOOSE,
        )
        env = EchelonEnv(cfg)
        env.reset(seed=0)

        obs = env._obs()
        scout_obs = obs["blue_0"]
        formation_one_hot = scout_obs[-3:]

        assert formation_one_hot[0] == 0.0
        assert formation_one_hot[1] == 0.0
        assert formation_one_hot[2] == 1.0  # LOOSE
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_observations.py::TestFormationModeObservation -v`
Expected: FAIL with assertion error (formation mode not in obs yet)

**Step 3: Write minimal implementation**

```python
# In echelon/env/observations.py

# 1. Add import at top
from echelon.env.rewards import FormationMode

# 2. Update self_dim from 48 to 51
self_dim = 51  # Was 48, added 3 for formation mode one-hot

# 3. Add formation_mode to ObservationContext dataclass
    formation_mode: FormationMode = FormationMode.STANDARD

# 4. In from_env() class method, pass formation_mode
    formation_mode=env.config.formation_mode,

# 5. In _build_self_features(), add formation one-hot at end (before return)
        # Formation mode one-hot (for commander responsiveness)
        formation_close = 1.0 if ctx.formation_mode == FormationMode.CLOSE else 0.0
        formation_standard = 1.0 if ctx.formation_mode == FormationMode.STANDARD else 0.0
        formation_loose = 1.0 if ctx.formation_mode == FormationMode.LOOSE else 0.0

# 6. Add to self_features array (at end, before the closing bracket)
            formation_close,
            formation_standard,
            formation_loose,
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_observations.py::TestFormationModeObservation -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add echelon/env/observations.py tests/unit/test_observations.py
git commit -m "feat(obs): add formation mode one-hot to observations"
```

---

### Task 4: Apply Formation Multipliers to Rewards

**Files:**
- Modify: `echelon/env/rewards.py`
- Modify: `echelon/env/env.py`
- Test: `tests/unit/test_rewards.py`

**Step 1: Write the failing test**

```python
# Add to TestFormationModeMultipliers class

def test_close_formation_amplifies_zone_reward(self):
    """CLOSE formation gives higher zone rewards."""
    from echelon.config import EnvConfig, WorldConfig
    from echelon.env.rewards import FormationMode
    from echelon import EchelonEnv
    from echelon.gen.objective import capture_zone_params
    import numpy as np

    # Standard formation
    cfg_std = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=15, obstacle_fill=0.0),
        num_packs=1,
        seed=0,
        formation_mode=FormationMode.STANDARD,
    )
    env_std = EchelonEnv(cfg_std)
    env_std.reset(seed=0)

    # Close formation
    cfg_close = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=15, obstacle_fill=0.0),
        num_packs=1,
        seed=0,
        formation_mode=FormationMode.CLOSE,
    )
    env_close = EchelonEnv(cfg_close)
    env_close.reset(seed=0)

    # Position blue_0 in zone for both
    zone_cx, zone_cy, _ = capture_zone_params(
        env_std.world.meta, size_x=env_std.world.size_x, size_y=env_std.world.size_y
    )

    for env in [env_std, env_close]:
        for aid in env.agents:
            m = env.sim.mechs[aid]
            if aid == "blue_0":
                m.pos[0], m.pos[1] = zone_cx, zone_cy
            else:
                m.pos[0], m.pos[1] = 5.0, 5.0
            m.vel[:] = 0.0

    actions = {aid: np.zeros(9, dtype=np.float32) for aid in env_std.agents}

    _, rewards_std, _, _, _ = env_std.step(actions)
    _, rewards_close, _, _, _ = env_close.step(actions)

    # CLOSE should give higher zone reward (2x multiplier)
    assert rewards_close["blue_0"] > rewards_std["blue_0"] * 1.5, (
        f"CLOSE should amplify zone reward: close={rewards_close['blue_0']}, std={rewards_std['blue_0']}"
    )

def test_loose_formation_reduces_zone_reward(self):
    """LOOSE formation gives lower zone rewards."""
    from echelon.config import EnvConfig, WorldConfig
    from echelon.env.rewards import FormationMode
    from echelon import EchelonEnv
    from echelon.gen.objective import capture_zone_params
    import numpy as np

    # Standard formation
    cfg_std = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=15, obstacle_fill=0.0),
        num_packs=1,
        seed=0,
        formation_mode=FormationMode.STANDARD,
    )
    env_std = EchelonEnv(cfg_std)
    env_std.reset(seed=0)

    # Loose formation
    cfg_loose = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=15, obstacle_fill=0.0),
        num_packs=1,
        seed=0,
        formation_mode=FormationMode.LOOSE,
    )
    env_loose = EchelonEnv(cfg_loose)
    env_loose.reset(seed=0)

    # Position blue_0 in zone for both
    zone_cx, zone_cy, _ = capture_zone_params(
        env_std.world.meta, size_x=env_std.world.size_x, size_y=env_std.world.size_y
    )

    for env in [env_std, env_loose]:
        for aid in env.agents:
            m = env.sim.mechs[aid]
            if aid == "blue_0":
                m.pos[0], m.pos[1] = zone_cx, zone_cy
            else:
                m.pos[0], m.pos[1] = 5.0, 5.0
            m.vel[:] = 0.0

    actions = {aid: np.zeros(9, dtype=np.float32) for aid in env_std.agents}

    _, rewards_std, _, _, _ = env_std.step(actions)
    _, rewards_loose, _, _, _ = env_loose.step(actions)

    # LOOSE should give lower zone reward (0.3x multiplier)
    assert rewards_loose["blue_0"] < rewards_std["blue_0"] * 0.5, (
        f"LOOSE should reduce zone reward: loose={rewards_loose['blue_0']}, std={rewards_std['blue_0']}"
    )
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_rewards.py::TestFormationModeMultipliers::test_close_formation_amplifies_zone_reward -v`
Expected: FAIL (rewards are same because multipliers not applied yet)

**Step 3: Write minimal implementation**

```python
# In echelon/env/rewards.py, add formation_mode to StepContext dataclass
    formation_mode: FormationMode = FormationMode.STANDARD

# In echelon/env/rewards.py, modify RewardComputer._compute_agent_reward()
# After getting base weights, apply formation multipliers

def _compute_agent_reward(
    self,
    aid: str,
    team: str,
    ctx: StepContext,
    teammates_in_zone: dict[str, int],
) -> RewardComponents:
    """Compute reward components for a single agent."""
    w = self.weights
    comp = RewardComponents()

    # Get formation multipliers
    fm = FORMATION_MULTIPLIERS[ctx.formation_mode]

    # (1) Approach shaping: PBRS-compliant distance-based reward
    # Apply formation multiplier to approach
    d0 = ctx.dist_to_zone_before.get(aid)
    d1 = ctx.dist_to_zone_after.get(aid)
    if d0 is not None and d1 is not None and ctx.max_xy > 0.0:
        phi0 = -d0 / ctx.max_xy
        phi1 = -d1 / ctx.max_xy
        comp.approach = w.approach * fm["approach"] * (w.shaping_gamma * phi1 - phi0)

    # ... (rest unchanged until zone reward)

    # (2) Zone control reward - apply formation multiplier
    team_tick = ctx.blue_tick if team == "blue" else ctx.red_tick
    if ctx.in_zone_by_agent.get(aid, False):
        n_in_zone = max(1, teammates_in_zone.get(team, 1))
        comp.zone = w.zone_tick * fm["zone"] * team_tick / math.sqrt(n_in_zone)

    # ... (rest unchanged until death penalty)

    # (4) Death penalty: zone-dependent with formation multiplier
    if ctx.step_deaths.get(aid, False):
        if agent_in_zone:
            comp.death = w.death * w.in_zone_death_mult
        else:
            # Out-of-zone death: apply formation multiplier
            comp.death = w.death * w.out_zone_death_mult * fm["out_death"]
```

```python
# In echelon/env/env.py, pass formation_mode to StepContext
# In step() method, when creating reward_ctx:

reward_ctx = StepContext(
    # ... existing fields ...
    formation_mode=self.config.formation_mode,
)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_rewards.py::TestFormationModeMultipliers::test_close_formation_amplifies_zone_reward tests/unit/test_rewards.py::TestFormationModeMultipliers::test_loose_formation_reduces_zone_reward -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add echelon/env/rewards.py echelon/env/env.py tests/unit/test_rewards.py
git commit -m "feat(rewards): apply formation multipliers to zone/approach/death rewards"
```

---

### Task 5: Add Formation Mode Randomization for Training

**Files:**
- Modify: `echelon/training/vec_env.py`
- Modify: `scripts/train_ppo.py`
- Test: Manual verification with training run

**Step 1: Add formation mode setter to VectorEnv**

```python
# In echelon/training/vec_env.py, add method to VectorEnv class

def set_formation_mode(self, mode: FormationMode) -> None:
    """Set formation mode for all environments.

    Used during training to cycle through formation modes
    so policy learns to respond to all three.
    """
    for env in self.envs:
        env.config.formation_mode = mode
```

**Step 2: Add formation cycling to training loop**

```python
# In scripts/train_ppo.py, add import
from echelon.env.rewards import FormationMode

# In main(), after creating vec_env, add formation cycling logic
# Cycle formation every N updates to expose policy to all modes

FORMATION_CYCLE_UPDATES = 3  # Change formation every 3 updates

# In training loop, before rollout collection:
if update % FORMATION_CYCLE_UPDATES == 0:
    formation_idx = (update // FORMATION_CYCLE_UPDATES) % 3
    formation = FormationMode(formation_idx)
    vec_env.set_formation_mode(formation)
    if wandb_run:
        wandb.log({"train/formation_mode": formation_idx}, step=global_step)
```

**Step 3: Run short training to verify**

Run: `PYTHONPATH=. uv run python scripts/train_ppo.py --updates 10 --num-envs 2 --packs-per-team 1 --size 40`
Expected: Training runs, formation mode cycles visible in output

**Step 4: Commit**

```bash
git add echelon/training/vec_env.py scripts/train_ppo.py
git commit -m "feat(training): add formation mode cycling during training"
```

---

### Task 6: Final Verification and Documentation

**Files:**
- Run full test suite
- Update CLAUDE.md if needed

**Step 1: Run full test suite**

Run: `PYTHONPATH=. uv run pytest tests/unit -v --tb=short`
Expected: All tests pass

**Step 2: Run lint and type check**

Run: `uv run ruff check echelon/ && uv run mypy echelon/`
Expected: No errors

**Step 3: Commit any final fixes**

```bash
git add -A
git commit -m "chore: formation mode final cleanup"
```

---

## Summary

After completing all tasks:

1. **FormationMode enum**: CLOSE (0), STANDARD (1), LOOSE (2)
2. **Multipliers**: zone, out_death, approach per mode
3. **Observation**: 3-dim one-hot in self-features (self_dim 48→51)
4. **Rewards**: Zone tick, approach, out-of-zone death scaled by formation
5. **Training**: Cycles through formations so policy learns all three

**Next phase**: Train commander policy to output formation mode based on tactical situation.
