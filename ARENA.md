# Arena Self-Play System

The arena enables self-play training where policies train against past versions of themselves plus a permanent heuristic baseline. This creates an ever-evolving curriculum that prevents overfitting to a single opponent.

## Quick Start

```bash
# Bootstrap league with Lieutenant Heuristic
uv run python scripts/arena.py bootstrap

# Add a trained checkpoint as opponent
uv run python scripts/arena.py add models/v1/best.pt --kind commander

# Train in arena mode
uv run python scripts/train_ppo.py --train-mode arena --arena-league runs/arena/league.json

# View league standings
uv run python scripts/arena.py list
```

## Core Concepts

### Lieutenant Heuristic

A permanent baseline opponent that never retires. Uses rule-based AI (`echelon/agents/heuristic.py`) instead of neural network inference.

- **Always available**: Present in every league from bootstrap
- **No checkpoint file**: Uses `venv.get_heuristic_actions()` directly
- **Rating updates**: Glicko-2 rating changes based on match results
- **Stable anchor**: Provides consistent difficulty reference as policies evolve

### League Entries

Each opponent in the league is a `LeagueEntry` with:

| Field | Description |
|-------|-------------|
| `entry_id` | Unique identifier (checkpoint hash or "heuristic") |
| `ckpt_path` | Path to checkpoint file (empty for heuristic) |
| `kind` | `"candidate"`, `"commander"`, or `"heuristic"` |
| `commander_name` | Display name (e.g., "Lieutenant Heuristic", "Iron Viper #a3f2") |
| `rating` | Glicko-2 rating object |
| `games` | Total games played |

### Entry Kinds

- **`heuristic`**: Lieutenant Heuristic - permanent, never retires
- **`commander`**: Promoted policies - proven strong, available as opponents
- **`candidate`**: New policies being evaluated - not yet proven
- **`retired`**: Former commanders removed from active pool

## Glicko-2 Rating System

The arena uses [Glicko-2](http://www.glicko.net/glicko/glicko2.pdf) for skill estimation:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rating` | 1500 | Skill estimate (higher = stronger) |
| `rd` | 350 | Rating deviation (uncertainty) |
| `vol` | 0.06 | Volatility (consistency) |

**Conservative rating** = `rating - 2*RD` (95% lower bound, used for matchmaking)

### Rating Updates

After matches, ratings update based on:
- Expected vs actual outcome
- Opponent's rating and RD
- Your own RD (uncertain ratings change more)

Ratings are updated in "rating periods" - batches of games processed together.

## PFSP Opponent Sampling

**Prioritized Fictitious Self-Play** samples opponents by skill similarity:

```
weight(opponent) = exp(-(rating_diff)² / (2 * sigma²))
```

- Opponents with similar ratings are more likely to be sampled
- `sigma=200` by default (controls how much to prefer similar opponents)
- Creates natural curriculum: face opponents at your level

### Cold-Start Warmup

New policies (< 20 games) are protected:
- Only matched against opponents within ±200 rating initially
- Range expands progressively if pool is too small
- Prevents being crushed before rating stabilizes

## CLI Commands

### `arena.py bootstrap`

Initialize a new league with Lieutenant Heuristic.

```bash
uv run python scripts/arena.py bootstrap
uv run python scripts/arena.py bootstrap --force  # Overwrite existing
```

### `arena.py add <checkpoint>`

Add a policy checkpoint to the league.

```bash
uv run python scripts/arena.py add runs/train/best.pt --kind candidate
uv run python scripts/arena.py add runs/train/best.pt --kind commander
uv run python scripts/arena.py add runs/train/best.pt --kind commander --name "My Policy v1"
```

### `arena.py list`

Show all league entries with ratings.

```bash
uv run python scripts/arena.py list
```

### `arena.py eval-candidate <checkpoint>`

Evaluate a candidate against the league pool.

```bash
uv run python scripts/arena.py eval-candidate runs/train/best.pt --matches 20
```

**Note**: This command only evaluates against neural network opponents (not Lieutenant Heuristic) because it requires loading checkpoint files.

### `arena.py init`

Create an empty league (without Lieutenant Heuristic).

```bash
uv run python scripts/arena.py init
```

## Training Integration

### Arena Mode

Enable with `--train-mode arena`:

```bash
uv run python scripts/train_ppo.py \
    --train-mode arena \
    --arena-league runs/arena/league.json \
    ...
```

### How It Works

1. **Opponent sampling**: Each rollout samples an opponent from the pool using PFSP
2. **Action generation**:
   - If heuristic: `venv.get_heuristic_actions(red_ids)`
   - If neural network: Forward pass through cached opponent model
3. **Status display**: Shows current opponent (`| vs Lt. Heuristic` or `| vs best:3491115...`)

### Relevant Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-mode` | `heuristic` | `heuristic` or `arena` |
| `--arena-league` | `runs/arena/league.json` | Path to league file |
| `--arena-top-k` | 20 | Number of top commanders in pool |
| `--arena-candidate-k` | 5 | Number of recent candidates in pool |
| `--arena-refresh-episodes` | 100 | Refresh pool every N episodes |

## Architecture

### Files

| File | Purpose |
|------|---------|
| `echelon/arena/glicko2.py` | Glicko-2 rating algorithm |
| `echelon/arena/league.py` | League management and persistence |
| `echelon/arena/match.py` | Match execution between two policies |
| `scripts/arena.py` | CLI tool |
| `scripts/train_ppo.py` | Training loop with arena integration |

### League Persistence

Leagues are stored as JSON:

```json
{
  "schema_version": 1,
  "glicko2": {
    "tau": 0.5,
    "epsilon": 1e-6,
    "rating0": 1500.0,
    "rd0": 350.0,
    "vol0": 0.06
  },
  "env_signature": { ... },
  "entries": [ ... ]
}
```

The `env_signature` prevents mixing policies trained with incompatible environment configurations.

### Model Caching

An LRU cache (default size 20) prevents OOM when loading many opponent models:

```python
arena_cache = LRUModelCache(max_size=20)
```

Evicted models are moved to CPU before deletion to free GPU memory.

## Workflow Recommendations

### Initial Training

1. Train against heuristic until policy is competitive:
   ```bash
   uv run python scripts/train_ppo.py --train-mode heuristic ...
   ```

2. Save the best checkpoint:
   ```bash
   cp runs/train/best.pt models/v1.pt
   ```

3. Bootstrap arena with this checkpoint:
   ```bash
   uv run python scripts/arena.py bootstrap
   uv run python scripts/arena.py add models/v1.pt --kind commander
   ```

### Ongoing Training

1. Train in arena mode
2. Periodically save checkpoints to stable locations
3. Add strong checkpoints as commanders:
   ```bash
   cp runs/arena-train/best.pt models/v2.pt
   uv run python scripts/arena.py add models/v2.pt --kind commander
   ```

### Managing League Size

The `retire_commanders()` method removes low-rated commanders:

```python
league.retire_commanders(keep_top=20, min_games=20)
```

- Keeps top 20 commanders by conservative rating
- Only retires commanders with 20+ games (rating has stabilized)
- Lieutenant Heuristic is never retired

## Possible Enhancements

### Near-Term

1. **Automatic checkpoint promotion**
   - Periodically evaluate training policy against league
   - Auto-add as commander if it achieves top-k rating
   - Retire weak commanders automatically

2. **Match replay storage**
   - Save replays of arena matches for analysis
   - Identify failure modes against specific opponents

3. **Per-opponent statistics**
   - Track win rate against each opponent
   - Identify which opponents cause the most trouble

4. **Rating-based curriculum**
   - Adjust opponent sampling based on training progress
   - Start with weaker opponents, increase difficulty over time

### Medium-Term

5. **Diverse opponent pool seeding**
   - Add checkpoints from different training stages
   - Include policies with different hyperparameters
   - Prevent convergence to single playstyle

6. **Asymmetric self-play**
   - Train with different team compositions
   - Create specialists vs generalists

7. **Population-based training integration**
   - Multiple training runs in parallel
   - Share checkpoints between runs
   - Evolve hyperparameters based on arena performance

8. **Match quality metrics**
   - Track game diversity (not just win/loss)
   - Measure tactical variety
   - Detect degenerate equilibria

### Long-Term

9. **Hierarchical leagues**
   - Bronze/Silver/Gold tiers
   - Promotion/relegation between tiers
   - Different sampling strategies per tier

10. **Multi-objective ratings**
    - Separate ratings for different skills (combat, zone control, survival)
    - Sample opponents based on skill gaps

11. **Human-in-the-loop**
    - Allow human players as league opponents
    - Calibrate ratings against human skill levels

12. **Transfer learning**
    - Train on smaller maps, transfer to larger
    - Use arena to validate transfer success

## Debugging

### Common Issues

**"arena league has no eligible opponents"**
- League is empty or only has heuristic
- Add at least one checkpoint: `arena.py add <ckpt> --kind commander`

**"checkpoint env_cfg does not match league env_signature"**
- Checkpoint was trained with different environment settings
- Use `--force` carefully or start fresh league

**AssertionError on arena_lstm_state**
- Fixed in commit `acac872`
- Occurs when switching from heuristic to neural network opponent
- LSTM state wasn't initialized after heuristic

### Inspecting League State

```bash
# Pretty-print league
cat runs/arena/league.json | python -m json.tool

# Check specific entry
uv run python -c "
from echelon.arena.league import League
from pathlib import Path
l = League.load(Path('runs/arena/league.json'))
print(l.entries['heuristic'].rating)
"
```

## References

- [Glicko-2 Paper](http://www.glicko.net/glicko/glicko2.pdf) - Rating algorithm
- [AlphaStar Nature Paper](https://www.nature.com/articles/s41586-019-1724-z) - PFSP and league training
- [OpenAI Five](https://openai.com/research/openai-five) - Self-play at scale
