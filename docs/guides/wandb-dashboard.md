# W&B Dashboard Configuration Guide

This guide documents how to set up an effective Weights & Biases dashboard for monitoring Echelon training runs.

## Quick Start

1. Enable W&B logging when training:
   ```bash
   uv run python scripts/train_ppo.py --wandb --wandb-run-name "my-experiment"
   ```

2. Visit your W&B project page and create a new dashboard workspace.

3. Follow the panel layout recommendations below.

---

## Recommended Dashboard Layout

### Row 1: Training Progress

The primary indicators of training health and performance.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Line | `train/avg_return`, `returns/p10`, `returns/p90` | Episode returns with spread (p10-p90 shows variance) |
| Line | `train/win_rate_blue`, `train/win_rate_red` | Win rates for learning team vs opponent |
| Line | `train/sps` | Steps per second (training throughput) |

**Key signals:**
- `avg_return` trending upward = learning
- `win_rate_blue` > 0.5 = outperforming opponent
- Narrow p10-p90 spread = consistent performance

### Row 2: Policy Health

Indicators of PPO algorithm health and stability.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Line | `train/entropy`, `train/approx_kl` | Exploration vs policy update magnitude |
| Line | `train/clipfrac`, `policy/saturation_rate` | PPO clipping rate, action space saturation |
| Line | `policy/explained_variance` | Value function quality (target: > 0.5) |

**Key signals:**
- `entropy` slowly declining = healthy exploration decay
- `approx_kl` spikes = policy updating too aggressively (consider lower LR)
- `clipfrac` > 0.2 = clipping too much (increase clip range or lower LR)
- `saturation_rate` > 0.1 = actions hitting bounds too often
- `explained_variance` < 0 = value function not learning

### Row 3: Value & Advantage Statistics

Detailed policy learning diagnostics.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Line | `policy/advantage_mean`, `policy/advantage_std` | GAE advantage distribution |
| Line | `policy/value_mean`, `policy/value_std` | Critic predictions |
| Line | `policy/action_mean_norm`, `policy/action_std_mean` | Action distribution shape |

**Key signals:**
- `advantage_mean` near 0 = proper normalization
- `advantage_std` stable = consistent value estimation
- High `action_std_mean` = more exploration

### Row 4: Combat Performance

Domain-specific metrics showing learned combat behavior.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Line | `combat/damage_ratio`, `combat/survival_rate` | Combat efficiency |
| Line | `combat/kill_participation`, `combat/damage_dealt` | Team effectiveness |
| Bar | `reward/*` components | Reward signal breakdown |

**Key signals:**
- `damage_ratio` > 1.0 = dealing more than receiving
- `survival_rate` trending up = learning to stay alive
- `kill_participation` near 1.0 = whole team contributing

### Row 5: Zone Control

Objective-based metrics for tactical gameplay.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Line | `zone/control_margin` | Blue control - Red control (normalized) |
| Line | `zone/contested_ratio` | Time both teams in zone |
| Line | `zone/time_to_entry` | How quickly blue reaches zone (lower = faster) |

**Key signals:**
- `control_margin` > 0 = blue controlling objective
- Low `time_to_entry` = learned to move toward objective

### Row 6: Coordination (Strided - Every 10 Updates)

Pack-level tactical coordination metrics.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Line | `coordination/pack_dispersion` | Mean distance between pack members |
| Line | `coordination/centroid_zone_dist` | Pack centroid distance from zone |
| Line | `coordination/focus_fire` | Concentration of damage on single targets |

**Key signals:**
- Moderate `pack_dispersion` = not too spread, not clumped
- Low `centroid_zone_dist` = staying near objective
- High `focus_fire` > 0.5 = team focusing targets

### Row 7: Perception (Strided - Every 10 Updates)

Sensor and electronic warfare usage.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Line | `perception/visible_contacts` | Average visible enemies |
| Line | `perception/hostile_filter_usage` | Contact filter activation rate |
| Line | `perception/ecm_usage`, `perception/eccm_usage` | Scout EWAR activation rates |

**Key signals:**
- Stable `visible_contacts` = consistent situational awareness
- `ecm_usage` vs `eccm_usage` balance = strategic EWAR usage

### Row 8: Reward Components

Detailed breakdown of what behaviors the policy is optimizing.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Stacked Area | `reward_pct/*` | Percentage contribution of each reward component |
| Line | `reward/damage`, `reward/kills` | Combat rewards |
| Line | `reward/zone_presence`, `reward/zone_control` | Objective rewards |
| Line | `reward/survival`, `reward/formation` | Tactical rewards |

**Key signals:**
- Balanced `reward_pct/*` = multi-objective optimization
- Dominant single component = potential reward hacking

### Row 9: Per-Role Performance

Compare learning across mech classes.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Line | `train/reward_heavy`, `train/reward_medium` | Heavy/Medium class returns |
| Line | `train/reward_light`, `train/reward_scout` | Light/Scout class returns |
| Grouped Bar | `actions/*/primary`, `actions/*/secondary` | Weapon usage by role |

**Key signals:**
- All roles improving = balanced team composition
- One role lagging = may need role-specific rewards

### Row 10: Action Distributions (Strided - Every 5 Updates)

Per-action activation distributions.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Histogram | `distributions/action_forward`, `distributions/action_strafe` | Movement actions |
| Histogram | `distributions/action_yaw`, `distributions/action_vertical` | Orientation/vertical |
| Histogram | `distributions/action_primary`, `distributions/action_secondary` | Weapon activations |
| Histogram | `distributions/action_vent`, `distributions/action_tertiary` | Utility actions |

**Key signals:**
- Bimodal distributions = learned to activate/deactivate
- All near 0 = not using that action
- All saturated at bounds = action space issues

### Row 11: Return & Advantage Distributions

Histogram views of key statistics.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Histogram | `distributions/returns` | Episode return distribution |
| Histogram | `distributions/advantages` | GAE advantage distribution |

**Key signals:**
- Returns histogram shifting right = improving
- Advantages centered near 0 = proper normalization
- Heavy tails = high variance episodes

### Row 12: Spatial Heatmaps (Strided - Every 50 Updates)

Visualize where events occur on the map.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Image | `spatial/deaths` | Death location heatmap |
| Image | `spatial/damage` | Damage dealt heatmap |
| Image | `spatial/movement` | Agent movement density |

**Key signals:**
- Deaths clustered at zone = contesting objective
- Movement density around zone = learned positioning
- Damage at zone center = effective engagement

### Row 13: Evaluation & Arena

Periodic evaluation and arena match results.

| Panel Type | Metrics | Description |
|------------|---------|-------------|
| Line | `eval/win_rate`, `eval/mean_hp_margin` | Heuristic opponent evaluation |
| Line | `arena/rating`, `arena/rd` | Glicko-2 rating and deviation |
| Line | `arena/games`, `arena/promoted` | Arena matches played and promotion status |
| Table | `episodes/agent_breakdown` | Per-agent episode statistics |

---

## Metric Reference

### Logging Frequency

| Category | Stride | Metrics |
|----------|--------|---------|
| Core | Every update | `train/*`, `policy/*`, `returns/*`, `combat/*`, `zone/*`, `reward/*` |
| Distributions | Every 5 updates | `distributions/action_*` |
| Coordination | Every 10 updates | `coordination/*`, `perception/*` |
| Spatial | Every 50 updates | `spatial/*` |
| Evaluation | Per eval interval | `eval/*` |
| Arena | Per arena match | `arena/*` |
| Tables | Sampled | `episodes/*` |

### Full Metric List

#### train/
| Metric | Description |
|--------|-------------|
| `update` | PPO update number |
| `loss` | Total PPO loss |
| `pg_loss` | Policy gradient loss |
| `v_loss` | Value function loss |
| `entropy` | Policy entropy |
| `grad_norm` | Gradient norm |
| `approx_kl` | Approximate KL divergence |
| `clipfrac` | PPO clip fraction |
| `learning_rate` | Current learning rate |
| `sps` | Steps per second |
| `global_step` | Total environment steps |
| `episodes` | Total episodes completed |
| `avg_return` | Mean episode return |
| `avg_len` | Mean episode length |
| `win_rate_blue` | Blue team win rate |
| `win_rate_red` | Red team win rate |
| `reward_{role}` | Per-role average return (heavy, medium, light, scout) |

#### policy/
| Metric | Description |
|--------|-------------|
| `advantage_mean` | Mean GAE advantage |
| `advantage_std` | Standard deviation of advantages |
| `value_mean` | Mean value predictions |
| `value_std` | Standard deviation of values |
| `explained_variance` | Value function explained variance |
| `action_mean_norm` | L2 norm of mean action |
| `action_std_mean` | Mean of action standard deviations |
| `saturation_rate` | Fraction of actions at bounds |

#### returns/
| Metric | Description |
|--------|-------------|
| `mean` | Mean return |
| `median` | Median return |
| `p10` | 10th percentile return |
| `p90` | 90th percentile return |
| `std` | Return standard deviation |

#### combat/
| Metric | Description |
|--------|-------------|
| `damage_dealt` | Average damage dealt by blue |
| `damage_ratio` | Blue damage / Red damage |
| `kill_participation` | (Kills + Assists) / Total kills |
| `survival_rate` | Fraction of blue team alive at end |

#### zone/
| Metric | Description |
|--------|-------------|
| `control_margin` | (Blue ticks - Red ticks) / Episode length |
| `contested_ratio` | Contested ticks / Episode length |
| `time_to_entry` | Steps to first zone entry / Episode length |

#### coordination/
| Metric | Description |
|--------|-------------|
| `pack_dispersion` | Mean pairwise distance between pack members |
| `centroid_zone_dist` | Pack centroid distance from zone center |
| `focus_fire` | Max target damage / Total damage |

#### perception/
| Metric | Description |
|--------|-------------|
| `visible_contacts` | Average visible enemy contacts |
| `hostile_filter_usage` | Hostile contact filter activation rate |
| `ecm_usage` | Scout ECM activation rate |
| `eccm_usage` | Scout ECCM activation rate |

#### reward/
| Metric | Description |
|--------|-------------|
| `{component}` | Average reward from each component |

#### reward_pct/
| Metric | Description |
|--------|-------------|
| `{component}` | Percentage contribution of each reward component |

#### actions/
| Metric | Description |
|--------|-------------|
| `{role}/{slot}` | Activation rate for action slot by role |

#### distributions/
| Metric | Description |
|--------|-------------|
| `returns` | Histogram of recent episode returns |
| `advantages` | Histogram of GAE advantages |
| `action_{name}` | Histogram of action values (forward, strafe, vertical, yaw, primary, vent, secondary, tertiary, special) |

#### spatial/
| Metric | Description |
|--------|-------------|
| `deaths` | Death location heatmap |
| `damage` | Damage dealt heatmap |
| `movement` | Agent movement density heatmap |

#### eval/
| Metric | Description |
|--------|-------------|
| `win_rate` | Win rate vs heuristic opponent |
| `mean_hp_margin` | Average HP margin at game end |
| `episodes` | Number of evaluation episodes |

#### episodes/
| Metric | Description |
|--------|-------------|
| `agent_breakdown` | Table with per-agent statistics |

#### arena/
| Metric | Description |
|--------|-------------|
| `rating` | Glicko-2 rating |
| `rd` | Glicko-2 rating deviation |
| `games` | Number of arena matches played |
| `promoted` | Whether policy was promoted to league (1/0) |

---

## Troubleshooting

### Common Issues

**Metrics not appearing:**
- Ensure `--wandb` flag is passed
- Check `--wandb-mode` is `online` or `offline`
- Verify W&B project exists and you have access

**Histograms empty:**
- Histograms require minimum samples (10 for returns, 100 for advantages)
- Action histograms only logged every 5 updates

**Spatial heatmaps not showing:**
- Requires matplotlib installed
- Only logged every 50 updates
- Need events (deaths/damage) to occur

**High-frequency noise in metrics:**
- Use smoothing in W&B panel settings
- Consider larger rolling window

---

## Creating Custom Dashboards

### W&B Workspace Setup

1. Go to your project's Workspace tab
2. Click "Add panel" to add new visualizations
3. Use the search bar to find metrics by prefix (e.g., `combat/`)
4. Group related panels into sections

### Recommended Panel Settings

**Line Charts:**
- Smoothing: 0.6-0.8 for noisy metrics
- X-axis: `train/global_step` for consistent comparison
- Show min/max: Enable for return percentiles

**Histograms:**
- Bin count: 30-50 for good resolution
- Show mean line: Enable for quick reference

**Heatmaps:**
- Color scale: Hot or Inferno
- Origin: Lower (matches game coordinates)

---

## Command Line Options

```bash
# Basic W&B logging
uv run python scripts/train_ppo.py --wandb

# Full configuration
uv run python scripts/train_ppo.py \
    --wandb \
    --wandb-project "echelon" \
    --wandb-entity "my-team" \
    --wandb-run-name "experiment-01" \
    --wandb-group "ppo-ablation" \
    --wandb-tags "baseline,v1.0" \
    --wandb-watch gradients \
    --wandb-watch-freq 500 \
    --wandb-artifacts best

# Resume a previous run
uv run python scripts/train_ppo.py \
    --wandb \
    --wandb-id "abc123" \
    --wandb-resume must \
    --resume latest
```

| Option | Description |
|--------|-------------|
| `--wandb` | Enable W&B logging |
| `--wandb-mode` | `disabled`, `offline`, `online` |
| `--wandb-project` | W&B project name |
| `--wandb-entity` | W&B team/user |
| `--wandb-run-name` | Human-readable run name |
| `--wandb-group` | Group for comparing runs |
| `--wandb-tags` | Comma-separated tags |
| `--wandb-id` | Resume specific run |
| `--wandb-resume` | `allow`, `must`, `never` |
| `--wandb-watch` | Log gradients: `off`, `gradients`, `all` |
| `--wandb-watch-freq` | Gradient logging frequency |
| `--wandb-artifacts` | Save artifacts: `none`, `best`, `all` |
