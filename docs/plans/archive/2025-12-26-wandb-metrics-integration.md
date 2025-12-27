# W&B Comprehensive Metrics Integration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add comprehensive observability to Echelon training via W&B with scalar metrics, histograms, spatial heatmaps, and episode tables.

**Architecture:** Three-tier metric collection: (1) per-step accumulation in env.py, (2) per-episode aggregation in train_ppo.py, (3) per-update W&B logging with strided expensive metrics. Spatial data accumulated in grid accumulators, rendered as heatmaps.

**Tech Stack:** W&B (wandb), NumPy, matplotlib (for heatmap rendering)

---

## Phase 1: Core PPO Metrics (Already Computed, Just Wire Up)

### Task 1.1: Add Missing PPO Metrics to W&B

**Files:**
- Modify: `scripts/train_ppo.py:1000-1010` (extract metrics)
- Modify: `scripts/train_ppo.py:1270-1305` (W&B logging block)

**Step 1: Extract approx_kl and clipfrac from PPO metrics**

In train_ppo.py around line 1006, after extracting existing metrics, add:

```python
# Extract metrics
pg_loss = metrics["pg_loss"]
v_loss = metrics["vf_loss"]
entropy_loss = metrics["entropy"]
grad_norm = metrics["grad_norm"]
approx_kl = metrics["approx_kl"]  # ADD THIS
clipfrac = metrics["clipfrac"]    # ADD THIS
loss = metrics["loss"]
```

**Step 2: Add to W&B logging block**

In the wandb_metrics dict (around line 1270), add:

```python
wandb_metrics = {
    # ... existing metrics ...
    "train/approx_kl": approx_kl,
    "train/clipfrac": clipfrac,
    "train/learning_rate": trainer.optimizer.param_groups[0]["lr"],
}
```

**Step 3: Run smoke test**

```bash
timeout 60 uv run python scripts/train_ppo.py --total-steps 5000 --wandb --wandb-run-name "test-metrics" 2>&1 | head -50
```

**Step 4: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(metrics): add approx_kl, clipfrac, learning_rate to W&B"
```

---

### Task 1.2: Add Advantage and Value Statistics

**Files:**
- Modify: `scripts/train_ppo.py` (after GAE computation, before PPO update)

**Step 1: Compute advantage/value statistics after GAE**

After the `compute_gae_and_returns` call (around line 985), add:

```python
# Compute advantage and value statistics for logging
adv_mean = float(buffer.advantages.mean().item())
adv_std = float(buffer.advantages.std().item())
val_mean = float(buffer.values.mean().item())
val_std = float(buffer.values.std().item())

# Explained variance: how well values predict returns
# ev = 1 - Var(returns - values) / Var(returns)
with torch.no_grad():
    returns_var = buffer.returns.var()
    residual_var = (buffer.returns - buffer.values).var()
    explained_var = 1.0 - (residual_var / (returns_var + 1e-8))
    explained_var = float(explained_var.item())
```

**Step 2: Add to W&B logging**

```python
wandb_metrics.update({
    "policy/advantage_mean": adv_mean,
    "policy/advantage_std": adv_std,
    "policy/value_mean": val_mean,
    "policy/value_std": val_std,
    "policy/explained_variance": explained_var,
})
```

**Step 3: Run and verify**

```bash
timeout 60 uv run python scripts/train_ppo.py --total-steps 5000 --no-wandb 2>&1 | grep -E "adv|val"
```

**Step 4: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(metrics): add advantage/value statistics and explained variance"
```

---

### Task 1.3: Add Action Saturation Metrics

**Files:**
- Modify: `scripts/train_ppo.py` (in rollout collection loop)

**Step 1: Track action statistics during rollout**

After collecting actions, accumulate statistics. Add near the action collection (around line 850):

```python
# Track action saturation (|a| > 0.95 indicates policy hitting bounds)
action_abs = np.abs(actions_flat)
saturation_count = (action_abs > 0.95).sum()
total_actions = action_abs.size
```

Accumulate across the update:

```python
# Before rollout loop (around line 780)
update_action_sum = np.zeros(action_dim, dtype=np.float64)
update_action_sq_sum = np.zeros(action_dim, dtype=np.float64)
update_action_count = 0
update_saturation_count = 0

# Inside rollout step (after getting actions)
update_action_sum += actions_flat.sum(axis=0)
update_action_sq_sum += (actions_flat ** 2).sum(axis=0)
update_action_count += actions_flat.shape[0]
update_saturation_count += (np.abs(actions_flat) > 0.95).sum()
```

**Step 2: Compute and log statistics**

After rollout, before PPO update:

```python
# Action statistics
action_mean = update_action_sum / max(update_action_count, 1)
action_var = (update_action_sq_sum / max(update_action_count, 1)) - (action_mean ** 2)
action_std = np.sqrt(np.maximum(action_var, 0))
saturation_rate = update_saturation_count / max(update_action_count * action_dim, 1)
```

Add to W&B:

```python
wandb_metrics.update({
    "policy/action_mean_norm": float(np.linalg.norm(action_mean)),
    "policy/action_std_mean": float(action_std.mean()),
    "policy/saturation_rate": saturation_rate,
})
```

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(metrics): add action mean/std and saturation rate"
```

---

## Phase 2: Episode-Level Statistics

### Task 2.1: Add Return Distribution Statistics

**Files:**
- Modify: `scripts/train_ppo.py` (where episodic_returns is aggregated)

**Step 1: Compute return percentiles**

After computing `avg_return` (around line 1020), add:

```python
# Return distribution statistics
if len(recent_returns) >= 5:
    returns_arr = np.array(recent_returns)
    return_median = float(np.median(returns_arr))
    return_p10 = float(np.percentile(returns_arr, 10))
    return_p90 = float(np.percentile(returns_arr, 90))
    return_std = float(np.std(returns_arr))
else:
    return_median = avg_return
    return_p10 = avg_return
    return_p90 = avg_return
    return_std = 0.0
```

**Step 2: Add to W&B**

```python
wandb_metrics.update({
    "returns/mean": avg_return,
    "returns/median": return_median,
    "returns/p10": return_p10,
    "returns/p90": return_p90,
    "returns/std": return_std,
})
```

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(metrics): add return distribution statistics (median, p10, p90, std)"
```

---

### Task 2.2: Add Combat Statistics from Episode Stats

**Files:**
- Modify: `scripts/train_ppo.py` (episode completion handling)

**Step 1: Extract episode stats on episode end**

When episode ends (around line 920), extract stats from infos:

```python
if ep_over:
    # ... existing code ...

    # Extract combat stats from episode
    ep_stats = {}
    for bid in blue_ids:
        agent_info = infos_list[env_idx].get(bid, {})
        if "outcome" in agent_info:
            ep_stats = agent_info["outcome"].get("stats", {})
            break

    # Store for aggregation
    episodic_combat_stats.append({
        "damage_blue": ep_stats.get("damage_blue", 0.0),
        "damage_red": ep_stats.get("damage_red", 0.0),
        "kills_blue": ep_stats.get("kills_blue", 0.0),
        "kills_red": ep_stats.get("kills_red", 0.0),
        "assists_blue": ep_stats.get("assists_blue", 0.0),
        "deaths_blue": sum(1 for bid in blue_ids if not infos_list[env_idx].get(bid, {}).get("alive", True)),
    })
```

**Step 2: Initialize the tracking list**

Near other episodic tracking (around line 635):

```python
episodic_combat_stats: list[dict[str, float]] = []
```

**Step 3: Aggregate and log**

In the metrics computation section:

```python
# Combat statistics
recent_combat = episodic_combat_stats[-window:]
if recent_combat:
    avg_damage_blue = float(np.mean([s["damage_blue"] for s in recent_combat]))
    avg_damage_red = float(np.mean([s["damage_red"] for s in recent_combat]))
    damage_ratio = avg_damage_blue / max(avg_damage_red, 1.0)

    avg_kills = float(np.mean([s["kills_blue"] for s in recent_combat]))
    avg_deaths = float(np.mean([s["deaths_blue"] for s in recent_combat]))
    avg_assists = float(np.mean([s["assists_blue"] for s in recent_combat]))

    # Kill participation = (kills + assists) / total_team_kills
    total_kills = avg_kills + float(np.mean([s["kills_red"] for s in recent_combat]))
    kill_participation = (avg_kills + avg_assists) / max(total_kills, 1.0)

    # Survival rate = agents alive at end / total agents
    survival_rate = 1.0 - (avg_deaths / len(blue_ids))
else:
    damage_ratio = 1.0
    kill_participation = 0.0
    survival_rate = 1.0
    avg_damage_blue = 0.0

wandb_metrics.update({
    "combat/damage_dealt": avg_damage_blue,
    "combat/damage_ratio": damage_ratio,
    "combat/kill_participation": kill_participation,
    "combat/survival_rate": survival_rate,
})
```

**Step 4: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(metrics): add combat statistics (damage ratio, kill participation, survival)"
```

---

### Task 2.3: Add Zone Control Metrics

**Files:**
- Modify: `echelon/env/env.py` (add zone tracking to _episode_stats)
- Modify: `scripts/train_ppo.py` (aggregate zone metrics)

**Step 1: Add zone tracking in env.py**

In `_episode_stats` initialization (around line 383), add:

```python
self._episode_stats = {
    # ... existing ...
    "zone_ticks_blue": 0.0,
    "zone_ticks_red": 0.0,
    "contested_ticks": 0.0,
    "first_zone_entry_step": -1.0,  # -1 means never entered
}
self._step_in_episode = 0
```

In the zone control computation section of `step()` (around line 2050), add:

```python
# Track zone presence for metrics
blue_in_zone = any(in_zone_by_agent.get(bid, False) for bid in self.blue_ids)
red_in_zone = any(in_zone_by_agent.get(rid, False) for rid in self.red_ids)

if blue_in_zone:
    self._episode_stats["zone_ticks_blue"] += 1.0
if red_in_zone:
    self._episode_stats["zone_ticks_red"] += 1.0
if blue_in_zone and red_in_zone:
    self._episode_stats["contested_ticks"] += 1.0
if blue_in_zone and self._episode_stats["first_zone_entry_step"] < 0:
    self._episode_stats["first_zone_entry_step"] = float(self._step_in_episode)

self._step_in_episode += 1
```

**Step 2: Reset step counter in reset()**

In `reset()`, add:

```python
self._step_in_episode = 0
```

**Step 3: Aggregate in train_ppo.py**

Add zone metrics to episodic_combat_stats extraction:

```python
episodic_zone_stats.append({
    "zone_ticks_blue": ep_stats.get("zone_ticks_blue", 0.0),
    "zone_ticks_red": ep_stats.get("zone_ticks_red", 0.0),
    "contested_ticks": ep_stats.get("contested_ticks", 0.0),
    "first_zone_entry": ep_stats.get("first_zone_entry_step", -1.0),
    "episode_length": ep_len,
})
```

Compute and log:

```python
recent_zone = episodic_zone_stats[-window:]
if recent_zone:
    # Zone control margin (blue - red normalized by episode length)
    margins = [(z["zone_ticks_blue"] - z["zone_ticks_red"]) / max(z["episode_length"], 1)
               for z in recent_zone]
    zone_margin = float(np.mean(margins))

    # Contested ratio
    contested_ratios = [z["contested_ticks"] / max(z["episode_length"], 1) for z in recent_zone]
    contested_ratio = float(np.mean(contested_ratios))

    # Time to first zone entry (normalized)
    entries = [z["first_zone_entry"] / max(z["episode_length"], 1)
               for z in recent_zone if z["first_zone_entry"] >= 0]
    time_to_zone = float(np.mean(entries)) if entries else 1.0
else:
    zone_margin = 0.0
    contested_ratio = 0.0
    time_to_zone = 1.0

wandb_metrics.update({
    "zone/control_margin": zone_margin,
    "zone/contested_ratio": contested_ratio,
    "zone/time_to_entry": time_to_zone,
})
```

**Step 4: Run tests**

```bash
PYTHONPATH=. uv run pytest tests/unit/test_rewards.py -v --tb=short
```

**Step 5: Commit**

```bash
git add echelon/env/env.py scripts/train_ppo.py
git commit -m "feat(metrics): add zone control metrics (margin, contested ratio, time to entry)"
```

---

## Phase 3: Coordination Metrics (Strided - Every 10 Updates)

### Task 3.1: Add Pack Dispersion Tracking

**Files:**
- Modify: `echelon/env/env.py` (add dispersion accumulator)
- Modify: `scripts/train_ppo.py` (aggregate and log)

**Step 1: Add dispersion tracking in env.py**

In `_episode_stats` initialization:

```python
self._episode_stats.update({
    "pack_dispersion_sum": 0.0,
    "pack_dispersion_count": 0.0,
    "centroid_zone_dist_sum": 0.0,
})
```

Add helper method to EchelonEnv class:

```python
def _compute_pack_dispersion(self, team: str) -> float:
    """Compute mean pairwise distance between pack members."""
    sim = self.sim
    if sim is None:
        return 0.0

    ids = self.blue_ids if team == "blue" else self.red_ids
    positions = []
    for mid in ids:
        m = sim.mechs.get(mid)
        if m is not None and m.alive:
            positions.append(m.pos[:2])  # XY only

    if len(positions) < 2:
        return 0.0

    # Mean pairwise distance
    total_dist = 0.0
    count = 0
    for i, p1 in enumerate(positions):
        for p2 in positions[i+1:]:
            total_dist += float(np.linalg.norm(p1 - p2))
            count += 1

    return total_dist / max(count, 1)
```

In `step()`, after zone tracking:

```python
# Coordination metrics (computed every step, accumulated)
dispersion = self._compute_pack_dispersion("blue")
self._episode_stats["pack_dispersion_sum"] += dispersion
self._episode_stats["pack_dispersion_count"] += 1.0

# Centroid to zone distance
blue_positions = [sim.mechs[bid].pos[:2] for bid in self.blue_ids
                  if sim.mechs[bid].alive]
if blue_positions:
    centroid = np.mean(blue_positions, axis=0)
    zone_center = np.array([zone_cx, zone_cy])
    centroid_dist = float(np.linalg.norm(centroid - zone_center))
    self._episode_stats["centroid_zone_dist_sum"] += centroid_dist
```

**Step 2: Aggregate in train_ppo.py (strided)**

```python
# Coordination metrics - only compute every 10 updates
if update % 10 == 0:
    recent_coord = episodic_coord_stats[-window:]
    if recent_coord:
        avg_dispersion = float(np.mean([
            s["pack_dispersion_sum"] / max(s["pack_dispersion_count"], 1)
            for s in recent_coord
        ]))
        avg_centroid_dist = float(np.mean([
            s["centroid_zone_dist_sum"] / max(s["pack_dispersion_count"], 1)
            for s in recent_coord
        ]))
    else:
        avg_dispersion = 0.0
        avg_centroid_dist = 0.0

    wandb_metrics.update({
        "coordination/pack_dispersion": avg_dispersion,
        "coordination/centroid_zone_dist": avg_centroid_dist,
    })
```

**Step 3: Commit**

```bash
git add echelon/env/env.py scripts/train_ppo.py
git commit -m "feat(metrics): add coordination metrics (pack dispersion, centroid distance)"
```

---

### Task 3.2: Add Focus Fire Concentration

**Files:**
- Modify: `echelon/env/env.py` (track damage per target)

**Step 1: Track damage distribution in env.py**

In `_episode_stats` initialization:

```python
self._damage_by_target: dict[str, float] = {}  # target_id -> damage received
```

In event processing (laser_hit, projectile_hit handlers):

```python
# Track damage by target for focus fire metric
target_id = ev["target"]
dmg = float(ev["damage"])
self._damage_by_target[target_id] = self._damage_by_target.get(target_id, 0.0) + dmg
```

At episode end, compute concentration:

```python
# Focus fire concentration: damage on top target / total damage
if self._damage_by_target:
    total_dmg = sum(self._damage_by_target.values())
    max_target_dmg = max(self._damage_by_target.values())
    self._episode_stats["focus_fire_concentration"] = max_target_dmg / max(total_dmg, 1.0)
else:
    self._episode_stats["focus_fire_concentration"] = 0.0
```

Reset in `reset()`:

```python
self._damage_by_target = {}
```

**Step 2: Log in train_ppo.py**

```python
# Inside strided coordination block
wandb_metrics["coordination/focus_fire"] = float(np.mean([
    s.get("focus_fire_concentration", 0.0) for s in recent_coord
]))
```

**Step 3: Commit**

```bash
git add echelon/env/env.py scripts/train_ppo.py
git commit -m "feat(metrics): add focus fire concentration metric"
```

---

## Phase 4: Perception Metrics (Strided - Every 10 Updates)

### Task 4.1: Add Contact Count and Filter Usage

**Files:**
- Modify: `echelon/env/env.py` (track in observation building)

**Step 1: Track contact counts in env.py**

In `_episode_stats` initialization:

```python
self._episode_stats.update({
    "visible_contacts_sum": 0.0,
    "visible_contacts_count": 0.0,
    "hostile_filter_on_count": 0.0,
})
```

In observation building (where contacts are processed), track counts:

```python
# Track visible contact count for metrics
visible_count = sum(1 for rel, _, _ in selected[:max_contact_count] if rel <= self.config.sensor_range)
self._episode_stats["visible_contacts_sum"] += visible_count
self._episode_stats["visible_contacts_count"] += 1.0

# Track hostile filter usage
if self._contact_filter_hostile.get(aid, False):
    self._episode_stats["hostile_filter_on_count"] += 1.0
```

**Step 2: Aggregate and log**

```python
# Perception metrics (strided with coordination)
if update % 10 == 0:
    wandb_metrics.update({
        "perception/visible_contacts": float(np.mean([
            s["visible_contacts_sum"] / max(s["visible_contacts_count"], 1)
            for s in recent_coord
        ])),
        "perception/hostile_filter_usage": float(np.mean([
            s["hostile_filter_on_count"] / max(s["visible_contacts_count"], 1)
            for s in recent_coord
        ])),
    })
```

**Step 3: Commit**

```bash
git add echelon/env/env.py scripts/train_ppo.py
git commit -m "feat(metrics): add perception metrics (contact count, filter usage)"
```

---

### Task 4.2: Add EWAR Usage Tracking

**Files:**
- Modify: `echelon/env/env.py` (track ECM/ECCM activation)

**Step 1: Track EWAR in env.py**

In `_episode_stats` initialization:

```python
self._episode_stats.update({
    "ecm_on_ticks": 0.0,
    "eccm_on_ticks": 0.0,
    "scout_ticks": 0.0,  # denominator for scout-only metrics
})
```

In `step()`, after EWAR toggles are applied:

```python
# Track EWAR usage (scouts only)
for mid in self.blue_ids:
    m = sim.mechs.get(mid)
    if m is not None and m.alive and m.spec.name == "scout":
        self._episode_stats["scout_ticks"] += 1.0
        if m.ecm_on:
            self._episode_stats["ecm_on_ticks"] += 1.0
        if m.eccm_on:
            self._episode_stats["eccm_on_ticks"] += 1.0
```

**Step 2: Log**

```python
wandb_metrics.update({
    "perception/ecm_usage": float(np.mean([
        s["ecm_on_ticks"] / max(s["scout_ticks"], 1) for s in recent_coord
    ])),
    "perception/eccm_usage": float(np.mean([
        s["eccm_on_ticks"] / max(s["scout_ticks"], 1) for s in recent_coord
    ])),
})
```

**Step 3: Commit**

```bash
git add echelon/env/env.py scripts/train_ppo.py
git commit -m "feat(metrics): add EWAR usage metrics (ECM, ECCM)"
```

---

## Phase 5: Histograms and Distributions

### Task 5.1: Add Return and Advantage Histograms

**Files:**
- Modify: `scripts/train_ppo.py` (W&B logging section)

**Step 1: Add histogram logging**

In the W&B logging block:

```python
# Histograms (every update)
if len(recent_returns) >= 10:
    wandb_metrics["distributions/returns"] = wandb.Histogram(recent_returns)

# Advantage histogram (from buffer)
if buffer.advantages is not None:
    adv_sample = buffer.advantages.cpu().numpy().flatten()
    if len(adv_sample) > 100:
        # Sample to avoid huge histograms
        adv_sample = np.random.choice(adv_sample, 1000, replace=False)
    wandb_metrics["distributions/advantages"] = wandb.Histogram(adv_sample)
```

**Step 2: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(metrics): add return and advantage histograms"
```

---

### Task 5.2: Add Action Distribution Histograms

**Files:**
- Modify: `scripts/train_ppo.py`

**Step 1: Track action samples during rollout**

Near rollout tracking initialization:

```python
update_action_samples: list[np.ndarray] = []  # Store sampled actions for histograms
```

During rollout:

```python
# Sample actions for histogram (every 10th step to limit memory)
if step % 10 == 0:
    update_action_samples.append(actions_flat.copy())
```

**Step 2: Create histograms**

```python
# Action histograms (strided - every 5 updates)
if update % 5 == 0 and update_action_samples:
    all_actions = np.vstack(update_action_samples)
    action_names = ["forward", "strafe", "vertical", "yaw", "primary",
                    "vent", "secondary", "tertiary", "special"]
    for i, name in enumerate(action_names[:min(len(action_names), all_actions.shape[1])]):
        wandb_metrics[f"distributions/action_{name}"] = wandb.Histogram(all_actions[:, i])
```

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(metrics): add per-action histograms"
```

---

## Phase 6: Spatial Heatmaps (Strided - Every 50 Updates)

### Task 6.1: Create SpatialAccumulator Class

**Files:**
- Create: `echelon/training/spatial.py`
- Test: `tests/unit/training/test_spatial.py`

**Step 1: Write failing test**

```python
# tests/unit/training/test_spatial.py
import numpy as np
import pytest

from echelon.training.spatial import SpatialAccumulator


class TestSpatialAccumulator:
    def test_record_death(self):
        acc = SpatialAccumulator(grid_size=16)
        acc.record_death(50.0, 50.0, world_size=(100, 100))

        # Should be in center cell (8, 8)
        assert acc.death_locations[8, 8] == 1.0

    def test_record_multiple(self):
        acc = SpatialAccumulator(grid_size=16)
        acc.record_death(50.0, 50.0, world_size=(100, 100))
        acc.record_death(50.0, 50.0, world_size=(100, 100))

        assert acc.death_locations[8, 8] == 2.0

    def test_to_heatmap_image(self):
        acc = SpatialAccumulator(grid_size=16)
        acc.record_death(50.0, 50.0, world_size=(100, 100))

        images = acc.to_images()
        assert "deaths" in images
        # Should be a wandb.Image or numpy array
```

**Step 2: Run test to verify failure**

```bash
PYTHONPATH=. uv run pytest tests/unit/training/test_spatial.py -v
```

**Step 3: Implement SpatialAccumulator**

```python
# echelon/training/spatial.py
"""Spatial data accumulation for heatmap visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import wandb


class SpatialAccumulator:
    """Accumulates 2D spatial events for heatmap generation."""

    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size
        self.death_locations = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.damage_locations = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.movement_density = np.zeros((grid_size, grid_size), dtype=np.float32)

    def _to_grid(self, x: float, y: float, world_size: tuple[float, float]) -> tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int(np.clip(x / world_size[0] * self.grid_size, 0, self.grid_size - 1))
        gy = int(np.clip(y / world_size[1] * self.grid_size, 0, self.grid_size - 1))
        return gx, gy

    def record_death(self, x: float, y: float, world_size: tuple[float, float]) -> None:
        gx, gy = self._to_grid(x, y, world_size)
        self.death_locations[gy, gx] += 1.0

    def record_damage(self, x: float, y: float, damage: float, world_size: tuple[float, float]) -> None:
        gx, gy = self._to_grid(x, y, world_size)
        self.damage_locations[gy, gx] += damage

    def record_position(self, x: float, y: float, world_size: tuple[float, float]) -> None:
        gx, gy = self._to_grid(x, y, world_size)
        self.movement_density[gy, gx] += 1.0

    def reset(self) -> None:
        self.death_locations.fill(0.0)
        self.damage_locations.fill(0.0)
        self.movement_density.fill(0.0)

    def to_images(self) -> dict[str, "wandb.Image"]:
        """Convert accumulators to W&B images with colormaps."""
        import wandb

        try:
            import matplotlib.pyplot as plt

            images = {}

            for name, data in [
                ("deaths", self.death_locations),
                ("damage", self.damage_locations),
                ("movement", self.movement_density),
            ]:
                if data.max() > 0:
                    # Normalize to [0, 1]
                    normalized = data / data.max()

                    # Create figure
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(normalized, cmap="hot", origin="lower")
                    ax.set_title(f"{name.title()} Heatmap")
                    plt.colorbar(im, ax=ax)
                    plt.tight_layout()

                    images[name] = wandb.Image(fig)
                    plt.close(fig)

            return images

        except ImportError:
            # Fallback without matplotlib
            return {}
```

**Step 4: Run tests**

```bash
PYTHONPATH=. uv run pytest tests/unit/training/test_spatial.py -v
```

**Step 5: Commit**

```bash
git add echelon/training/spatial.py tests/unit/training/test_spatial.py
git commit -m "feat(metrics): add SpatialAccumulator for heatmap generation"
```

---

### Task 6.2: Integrate Spatial Tracking into Training

**Files:**
- Modify: `scripts/train_ppo.py`

**Step 1: Initialize accumulator**

```python
from echelon.training.spatial import SpatialAccumulator

# After env setup
spatial_acc = SpatialAccumulator(grid_size=32)
world_size = (env_cfg.world.size_x, env_cfg.world.size_y)
```

**Step 2: Record events during rollout**

In episode completion handling, extract death locations:

```python
# Record spatial data from episode
for bid in blue_ids:
    agent_info = infos_list[env_idx].get(bid, {})
    if not agent_info.get("alive", True):
        # Agent died - get position from last known state
        # This requires storing positions, or extracting from outcome
        pass  # Will be filled in from outcome data

# Simpler approach: extract from outcome stats
if "outcome" in agent_info:
    outcome = agent_info["outcome"]
    # Record deaths from events
    for event in outcome.get("events", []):
        if event.get("type") == "kill":
            x, y = event.get("pos", (0, 0))[:2]
            spatial_acc.record_death(x, y, world_size)
```

**Step 3: Log heatmaps (strided)**

```python
# Spatial heatmaps - every 50 updates
if update % 50 == 0:
    heatmap_images = spatial_acc.to_images()
    for name, img in heatmap_images.items():
        wandb_metrics[f"spatial/{name}"] = img
    spatial_acc.reset()
```

**Step 4: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(metrics): integrate spatial heatmaps into training loop"
```

---

## Phase 7: Episode Tables (Sampled)

### Task 7.1: Add Per-Agent Episode Breakdown Table

**Files:**
- Modify: `scripts/train_ppo.py`

**Step 1: Create table on episode end**

```python
# Sample 1 episode per update for detailed breakdown
if len(episodic_returns) % 10 == 0 and wandb_run is not None:
    # Build per-agent table
    columns = ["agent_id", "role", "alive", "damage_dealt", "damage_taken",
               "kills", "reward_total"]
    table_data = []

    for bid in blue_ids:
        agent_info = infos_list[env_idx].get(bid, {})
        rc = agent_info.get("reward_components", {})
        role = _role_for_agent(bid)

        table_data.append([
            bid,
            role,
            agent_info.get("alive", False),
            rc.get("damage", 0.0) * 200,  # Unnormalize
            0.0,  # damage_taken - would need tracking
            0.0,  # kills - would need per-agent tracking
            sum(rc.values()),
        ])

    episode_table = wandb.Table(columns=columns, data=table_data)
    wandb_run.log({"episodes/agent_breakdown": episode_table}, step=global_step)
```

**Step 2: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(metrics): add per-agent episode breakdown table"
```

---

## Phase 8: Cleanup and Documentation

### Task 8.1: Organize Metrics into Helper Functions

**Files:**
- Modify: `scripts/train_ppo.py`

**Step 1: Extract metric computation into functions**

```python
def compute_return_stats(recent_returns: list[float]) -> dict[str, float]:
    """Compute return distribution statistics."""
    if len(recent_returns) < 5:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "std": 0.0}

    arr = np.array(recent_returns)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "std": float(np.std(arr)),
    }


def compute_combat_stats(recent_combat: list[dict], num_agents: int) -> dict[str, float]:
    """Compute combat statistics from recent episodes."""
    if not recent_combat:
        return {"damage_ratio": 1.0, "kill_participation": 0.0, "survival_rate": 1.0}

    # ... implementation ...
```

**Step 2: Use helpers in main loop**

```python
# Replace inline computation with helper calls
return_stats = compute_return_stats(recent_returns)
combat_stats = compute_combat_stats(recent_combat, len(blue_ids))

wandb_metrics.update({f"returns/{k}": v for k, v in return_stats.items()})
wandb_metrics.update({f"combat/{k}": v for k, v in combat_stats.items()})
```

**Step 3: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "refactor(metrics): extract metric computation into helper functions"
```

---

### Task 8.2: Add W&B Dashboard Configuration

**Files:**
- Create: `wandb_config.yaml` (optional, for workspace setup)

**Step 1: Document recommended dashboard panels**

Create a markdown section in the plan or a separate doc describing recommended W&B dashboard layout:

```markdown
## Recommended W&B Dashboard Layout

### Row 1: Training Progress
- Line: train/avg_return, returns/p10, returns/p90
- Line: train/win_rate_blue, train/win_rate_red
- Line: train/sps

### Row 2: Policy Health
- Line: train/entropy, policy/approx_kl
- Line: policy/clipfrac, policy/saturation_rate
- Line: policy/explained_variance

### Row 3: Combat
- Line: combat/damage_ratio, combat/survival_rate
- Line: combat/kill_participation
- Bar: reward/* components

### Row 4: Coordination (Strided)
- Line: coordination/pack_dispersion
- Line: coordination/focus_fire
- Line: zone/control_margin

### Row 5: Distributions
- Histogram: distributions/returns
- Histogram: distributions/advantages
- Histogram: distributions/action_forward

### Row 6: Spatial (Strided)
- Image: spatial/deaths
- Image: spatial/damage
- Image: spatial/movement
```

**Step 2: Commit documentation**

```bash
git add docs/
git commit -m "docs: add W&B dashboard layout recommendations"
```

---

## Summary

**Total Tasks:** 15 across 8 phases

**Metrics Added:**

| Namespace | Count | Stride |
|-----------|-------|--------|
| train/ | 3 new (approx_kl, clipfrac, lr) | Every update |
| policy/ | 6 (advantage, value, saturation) | Every update |
| returns/ | 5 (mean, median, p10, p90, std) | Every update |
| combat/ | 4 (damage_ratio, kill_part, survival, dealt) | Every update |
| zone/ | 3 (margin, contested, time_to_entry) | Every update |
| coordination/ | 3 (dispersion, centroid, focus_fire) | Every 10 |
| perception/ | 4 (contacts, filter, ecm, eccm) | Every 10 |
| distributions/ | 12 (returns, advantages, 9 actions) | Every 5 |
| spatial/ | 3 heatmaps | Every 50 |
| episodes/ | 1 table | Sampled |

**Files Modified:**
- `echelon/env/env.py` - Add tracking accumulators
- `scripts/train_ppo.py` - Add aggregation and W&B logging
- `echelon/training/spatial.py` - New file for heatmaps

**Estimated Implementation Time:** 2-3 hours following TDD approach
