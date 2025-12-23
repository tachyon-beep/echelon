# Deep RL Arena Review: Self-Play and League Training

**Date**: 2025-12-23
**Reviewer**: Claude Code (Sonnet 4.5)
**Files Reviewed**:
- `/home/john/echelon/echelon/arena/league.py`
- `/home/john/echelon/echelon/arena/match.py`
- `/home/john/echelon/echelon/arena/glicko2.py`
- `/home/john/echelon/scripts/arena.py`
- `/home/john/echelon/scripts/train_ppo.py` (arena integration)

---

## Summary

The self-play infrastructure implements a league-based training system with Glicko-2 ratings. The implementation is **mostly solid** but has **critical bugs** in opponent sampling, rating updates, and LSTM state management. The Glicko-2 math appears correct, but the integration with training has exploitability issues that could harm diversity and convergence.

**Key Findings**:
- 1 Critical Bug: Opponent never resampled during training rollouts
- 1 Critical Bug: Double-update vulnerability in rating periods
- 3 Significant Issues: LSTM state management, diversity mechanisms, match termination
- 5 Improvement Opportunities: Population-based enhancements

---

## Bugs Found

### BUG-1: Opponent Never Resampled During Training (CRITICAL)
**Location**: `/home/john/echelon/scripts/train_ppo.py:766-778,869-871`

**Issue**: When `train_mode="arena"`, the opponent policy is sampled once at initialization (line 586) and **never resampled during rollout collection**. The code has a comment acknowledging this:

```python
# Line 869-871
# Currently we sample one opponent for the WHOLE vector.
# If an episode ends, we'll just keep the current opponent until the next update
# or refresh cycle, but we reset its LSTM state above.
```

**Impact**:
- The learning agent plays against the **same frozen opponent** for the entire rollout (default: 128 steps across all envs)
- This creates a distribution mismatch: training data comes from playing a single opponent, but evaluation/arena-submit samples from the full pool
- Limits exploration of the opponent strategy space
- Violates self-play best practices where opponent diversity during rollouts improves robustness

**Expected Behavior**: Sample a new opponent from the pool after each episode ends (lines 853-858), or at minimum, sample different opponents per parallel environment.

**Severity**: HIGH - Degrades self-play quality, potentially leading to overfitting to single opponents.

---

### BUG-2: Double Rating Update Vulnerability (CRITICAL)
**Location**: `/home/john/echelon/scripts/arena.py:170-175,194-195` and `/home/john/echelon/scripts/train_ppo.py:1196-1201,1220-1221`

**Issue**: When updating ratings, **both players in a match have their ratings added to the results dictionary**:

```python
# arena.py:170-175 (same pattern in train_ppo.py:1196-1201)
results.setdefault(candidate_entry.entry_id, []).append(
    GameResult(opponent=opp_entry.rating, score=cand_score)
)
results.setdefault(opp_entry.entry_id, []).append(
    GameResult(opponent=candidate_entry.rating, score=opp_score)
)
```

Then `league.apply_rating_period(results)` is called (line 194), which updates **all entries in the results dict** (line 207-214 in `league.py`).

**Problem**: If the same opponent appears in multiple matches within one rating period, AND that opponent's entry is also being updated in the league (not just used as a frozen opponent), its rating will be updated based on **stale opponent ratings**. Worse, if you're updating commanders who played each other, you get circular dependency issues.

**Glicko-2 Violation**: The algorithm expects all games in a rating period to use the **same starting ratings** for all players. Updating one player mid-period then using their new rating for another player's calculation is incorrect.

**Example Scenario**:
1. Candidate plays Opponent A (match 1): uses Opponent A's rating at start
2. Candidate plays Opponent B (match 2): uses Opponent B's rating at start
3. `apply_rating_period` updates Opponent A's rating based on match 1
4. Then updates Opponent B's rating, but if Opponent A and B played each other in **another** evaluation session that overlaps, the ratings become inconsistent

**Expected Behavior**: Either:
- Only update the candidate's rating and leave opponents frozen (common in evaluation)
- If updating all players, ensure no match reuse and all opponent ratings are captured at period start
- Use a two-phase update: compute all new ratings first, then commit atomically

**Severity**: CRITICAL for multi-agent tournaments. MEDIUM for single-candidate evaluation if opponents are truly frozen.

---

### BUG-3: LSTM State Not Reset When Opponent Resampled
**Location**: `/home/john/echelon/scripts/train_ppo.py:867-871`

**Issue**: When `arena_refresh_pool()` is called (line 868), the pool of available opponents is reloaded, but there's **no call to resample the opponent or reset its LSTM state**. The code only resets LSTM state when an episode ends (line 864), not when the pool refreshes.

**Problem**: If the pool composition changes (new commanders added, ratings updated), the current opponent might still be valid, but its cached LSTM state could be stale or the opponent policy object might not be the latest version.

**Expected Behavior**: After `_arena_refresh_pool()`, call `_arena_sample_opponent(reset_hidden=True)` to ensure the opponent policy and state are fresh.

**Current Mitigation**: The refresh only happens every N episodes (default: 20), and episode resets handle LSTM state, so this is low-severity in practice.

**Severity**: LOW (edge case), but indicates incomplete refresh logic.

---

## Issues

### ISSUE-1: No Per-Environment Opponent Diversity
**Location**: `/home/john/echelon/scripts/train_ppo.py:766-778`

**Issue**: All parallel environments (default: 4) play against the **same opponent policy instance** with a **shared LSTM state** (line 771-772). This is efficient but reduces opponent diversity.

**Impact**:
- When collecting a rollout batch, all 4 environments see the same opponent behavior
- The opponent's LSTM state is updated sequentially across all envs, which is not parallelizable and may cause order-dependent behavior
- Reduces effective diversity compared to sampling different opponents per environment

**Best Practice (Population-Based Training)**: Each parallel environment should sample its own opponent from the pool to maximize strategy diversity.

**Trade-off**: Sampling per-env increases memory (need multiple opponent model instances) and complexity. Current approach prioritizes efficiency.

**Severity**: MEDIUM - Limits diversity but is a design choice, not a bug.

---

### ISSUE-2: Conservative Score Metric May Be Too Aggressive
**Location**: `/home/john/echelon/echelon/arena/league.py:189-190`

**Issue**: When deciding if a candidate promotes to commander, the ranking uses:
```python
def conservative(e: LeagueEntry) -> float:
    return float(e.rating.rating) - 2.0 * float(e.rating.rd)
```

This is `μ - 2σ` (mean minus 2 standard deviations), which represents approximately the **2.5th percentile** of the rating distribution.

**Problem**:
- Very conservative: A candidate with high rating but high uncertainty (e.g., rating=1700, rd=200 → conservative=1300) will rank poorly
- New candidates start with high RD (default: 350), so they need to play many games to reduce RD below ~100 to be competitive
- May prevent strong-but-undertested candidates from promoting

**Glicko-2 Guidance**: The original paper suggests using `μ - 2σ` for conservative estimates, but in practice, many systems use `μ - σ` or even just `μ` for promotion decisions, reserving conservative scores for matchmaking.

**Recommendation**: Consider making the conservatism factor configurable (e.g., `k * rd` where `k ∈ [1.0, 2.0]`) or using different metrics for promotion vs matchmaking.

**Severity**: LOW-MEDIUM - May slow down discovery of strong candidates, but is defensible.

---

### ISSUE-3: No Opponent Sampling Strategy (Uniform Random Only)
**Location**: `/home/john/echelon/scripts/train_ppo.py:575` and `/home/john/echelon/scripts/arena.py:135`

**Issue**: Opponents are sampled uniformly at random from the pool:
```python
entry = arena_rng.choice(arena_pool)  # train_ppo.py:575
opp_entry = rng.choice(pool)          # arena.py:135
```

**Problem**:
- No prioritization of diverse or challenging opponents
- No curriculum learning (starting with weaker opponents and progressing to stronger)
- No prioritized sampling based on rating uncertainty or expected information gain

**Best Practices (DRL Self-Play)**:
- **PFSP (Prioritized Fictitious Self-Play)**: Sample opponents proportional to their win rate against the current policy (focus on "just-right" difficulty)
- **Rating-based sampling**: Sample closer-rated opponents more often for better gradient signal
- **Uncertainty sampling**: Prioritize opponents with high RD to gather more informative results
- **Curriculum**: Start with weaker opponents, gradually increase difficulty

**Current Approach**: Uniform sampling is simple and unbiased but may be inefficient.

**Severity**: MEDIUM - Significant opportunity for improvement, but uniform sampling is a valid baseline.

---

### ISSUE-4: Match Termination Logic Uses Three Different Checks
**Location**: `/home/john/echelon/echelon/arena/match.py:101-107`

**Issue**: The match termination condition is:
```python
if max_steps is not None and steps >= int(max_steps):
    break

blue_alive = env.sim.team_alive("blue")
red_alive = env.sim.team_alive("red")
if any(truncations.values()) or (not blue_alive) or (not red_alive):
    break
```

**Redundancy**: The `team_alive` check is redundant with the `truncations` check, because the environment should set truncations when a team is eliminated (see `env.py:1320-1369`).

**Verification**: Checking `env.py:1320-1369`, the environment **does** set truncations/terminations when teams are eliminated, so the explicit `team_alive` check is defensive but unnecessary.

**Minor Issue**: `env.last_outcome` is only set when the episode ends via `truncations` or team elimination (lines 1369, 1382), so if `max_steps` triggers first, `last_outcome` will be `None` and line 109 falls back to `{"winner": "draw", "hp": env.team_hp()}`. This is correct but relies on fallback.

**Severity**: LOW - Defensive coding is fine, but indicates lack of clarity on termination contract.

---

### ISSUE-5: No Diversity Maintenance Mechanism
**Location**: `/home/john/echelon/echelon/arena/league.py` (entire file)

**Issue**: The league tracks commanders and candidates but has **no mechanism to enforce diversity** in the commander pool. Possible collapse scenarios:
- All commanders converge to similar strategies (e.g., all learn to camp)
- Strong strategies dominate, weak but diverse strategies get pruned
- No explicit diversity objective (style, behavioral, or architectural)

**Best Practices (Population-Based Training)**:
- **Behavioral diversity**: Measure strategy similarity (e.g., action distributions, state visitation) and penalize duplicates
- **Archive best-response chains**: Keep snapshots of past strategies to prevent cycling
- **Quality-Diversity algorithms**: Use MAP-Elites or similar to maintain diverse high-performers

**Current Approach**: Relying solely on Glicko-2 ratings and top-K promotion may lead to monoculture.

**Mitigation**: The `recent_candidates` mechanism (line 177-181) provides some temporal diversity, but no explicit diversity metric.

**Severity**: MEDIUM - Long-term risk, but not immediately critical for initial experiments.

---

## Improvement Opportunities

### OPP-1: Add Opponent Sampling Per Episode
**Priority**: HIGH

Modify `train_ppo.py` to resample opponent when an episode ends:

```python
# After line 871 (inside the episode-end block)
if args.train_mode == "arena":
    if args.arena_refresh_episodes > 0 and episodes % int(args.arena_refresh_episodes) == 0:
        _arena_refresh_pool()
    # RESAMPLE opponent for the next episode
    _arena_sample_opponent(reset_hidden=False)  # LSTM reset handled by arena_done
```

**Benefit**: Increases opponent diversity during training, better match distribution for self-play.

---

### OPP-2: Implement Two-Phase Rating Updates
**Priority**: CRITICAL (if updating opponents)

Modify `league.apply_rating_period` to capture opponent ratings at period start:

```python
def apply_rating_period(self, results: dict[str, list[GameResult]]) -> None:
    # Phase 1: Snapshot all current ratings
    old_ratings = {eid: self.entries[eid].rating for eid in results.keys() if eid in self.entries}

    # Phase 2: Compute all new ratings using old snapshots
    new_ratings = {}
    for entry_id, games in results.items():
        entry = self.entries.get(entry_id)
        if entry is None:
            continue
        # Ensure all opponent ratings in games use old values (already captured in GameResult)
        new_ratings[entry_id] = rate(old_ratings[entry_id], games, cfg=self.cfg)

    # Phase 3: Commit atomically
    for entry_id, new_rating in new_ratings.items():
        self.entries[entry_id].rating = new_rating
        self.entries[entry_id].games += len(results[entry_id])
```

**Benefit**: Ensures correct Glicko-2 rating period semantics, prevents circular dependency bugs.

---

### OPP-3: Add Prioritized Opponent Sampling
**Priority**: MEDIUM

Implement PFSP or rating-based sampling:

```python
def _arena_sample_opponent_weighted(learning_policy_rating: Glicko2Rating) -> LeagueEntry:
    # Compute win probability against each opponent
    weights = []
    for entry in arena_pool:
        win_prob = expected_score(learning_policy_rating, entry.rating, cfg=league.cfg)
        # PFSP: prioritize ~50% win rate opponents
        priority = 1.0 - abs(win_prob - 0.5) * 2.0  # Peak at 0.5, decay to 0 at 0.0/1.0
        weights.append(max(0.01, priority))  # Avoid zero weights

    # Sample proportional to priority
    total = sum(weights)
    probs = [w / total for w in weights]
    return arena_rng.choices(arena_pool, weights=probs, k=1)[0]
```

**Benefit**: Focus training on "goldilocks zone" opponents for faster learning.

---

### OPP-4: Add Per-Environment Opponent Sampling
**Priority**: MEDIUM

Modify the vectorized training loop to sample different opponents per environment:

```python
# Replace single arena_opponent with per-env opponents
arena_opponents: list[ActorCriticLSTM] = []
arena_lstm_states: list[LSTMState] = []

def _arena_sample_opponents_per_env():
    nonlocal arena_opponents, arena_lstm_states
    arena_opponents = []
    arena_lstm_states = []
    for _ in range(num_envs):
        entry = arena_rng.choice(arena_pool)
        if entry.entry_id not in arena_cache:
            arena_cache[entry.entry_id] = _arena_load_model(entry.ckpt_path)
        model = arena_cache[entry.entry_id]
        arena_opponents.append(model)
        arena_lstm_states.append(model.initial_state(batch_size=len(red_ids), device=opponent_device))

# In rollout loop (line 766-778), iterate over per-env opponents
for env_idx in range(num_envs):
    obs_r = torch.from_numpy(stack_obs(next_obs_dicts[env_idx], red_ids)).to(opponent_device)
    act_r, _, _, _, arena_lstm_states[env_idx] = arena_opponents[env_idx].get_action_and_value(
        obs_r, arena_lstm_states[env_idx], arena_done[env_idx]
    )
    # ... (rest of loop)
```

**Benefit**: Maximum opponent diversity per rollout batch.

**Cost**: Increased memory usage (N opponent models in memory simultaneously).

---

### OPP-5: Add Behavioral Diversity Metrics
**Priority**: LOW (future work)

Track behavioral features (e.g., damage dealt, positioning heatmaps, weapon usage) in `LeagueEntry.meta` and use for diversity-aware matchmaking or pruning.

**Example**:
```python
# In LeagueEntry.meta
{
    "avg_damage_per_episode": 450.2,
    "avg_distance_to_center": 12.3,
    "weapon_usage": {"laser": 0.6, "missile": 0.3, "melee": 0.1},
    "behavioral_embedding": [0.1, 0.3, -0.5, ...]  # learned via autoencoder
}
```

Use cosine similarity or KL divergence to avoid promoting near-duplicates.

**Benefit**: Prevents strategy monoculture, maintains exploration.

---

### OPP-6: Implement Rating Uncertainty Decay for Inactive Entries
**Priority**: LOW

Glicko-2 increases RD (rating deviation) for players who haven't played recently to reflect increased uncertainty. The code handles this when `results` is empty (line 125-128 in `glicko2.py`), but there's no periodic decay for inactive commanders.

**Recommendation**: Periodically call `apply_rating_period` with empty results for all commanders who haven't played in N episodes.

**Benefit**: More accurate confidence intervals for inactive policies.

---

### OPP-7: Add Match Checkpointing and Resume
**Priority**: LOW

Currently, if `arena.py eval-candidate` crashes mid-evaluation, all matches are lost. Consider checkpointing results after each match to allow resume.

**Benefit**: Robustness for long evaluation runs.

---

### OPP-8: Visualize Rating History and Matchmaking Matrix
**Priority**: LOW

Add logging/plotting for:
- Rating evolution over time (with confidence intervals)
- Win rate matrix (who beats whom)
- Opponent sampling distribution (are some opponents over-sampled?)

**Benefit**: Better debugging and insights into league dynamics.

---

## Recommendations Summary

### Immediate Fixes (Before Production Use)
1. **Fix OPP-1**: Resample opponent after each episode during training
2. **Fix BUG-2**: Implement two-phase rating updates if opponents are updated
3. **Review BUG-3**: Add opponent resample after pool refresh (or document why not needed)

### Short-Term Enhancements (Next Iteration)
4. Implement OPP-3 (prioritized opponent sampling) for faster convergence
5. Add validation tests for Glicko-2 rating updates (ensure rating conservation, no inflation)
6. Add per-environment opponent sampling (OPP-4) if memory permits

### Long-Term Research (Future Work)
7. Explore behavioral diversity metrics (OPP-5)
8. Implement curriculum learning (weak → strong opponents)
9. Add rating history visualization and diagnostics

---

## Testing Gaps

**No unit tests found** for:
- `league.py`: Promotion logic, rating period updates, pool sampling
- `glicko2.py`: Rating calculations (should verify against reference implementation)
- `match.py`: Match termination conditions, outcome parsing
- `arena.py`: End-to-end league evaluation workflow

**Recommended Tests**:
1. **Glicko-2 Math**: Verify against official examples from the Glicko-2 paper
2. **Promotion Logic**: Test edge cases (ties, high-uncertainty candidates, empty pool)
3. **Rating Conservation**: Ensure sum of rating changes in a match ≈ 0 (Elo property; Glicko-2 is similar but not exact)
4. **Opponent Sampling**: Verify no duplicate opponents in single rating period
5. **LSTM State Management**: Ensure resets happen correctly across episode boundaries

---

## Glicko-2 Implementation Quality

The `glicko2.py` implementation appears **correct**:
- Properly implements Step 1-6 from the Glicko-2 paper (Glickman 2013)
- Uses Illinois algorithm for sigma convergence (line 101-111), as specified
- Handles edge cases (no games → RD increase, v_inv near zero)
- Floating-point precision looks adequate (epsilon=1e-6 by default)

**Minor Concern**: No validation against reference test cases. Recommend adding a unit test with known inputs/outputs from the Glicko-2 paper.

---

## Final Verdict

**Overall Assessment**: The arena system is **functional but needs critical fixes** before scaling to serious self-play experiments.

**Strengths**:
- Clean separation of concerns (league, match, rating)
- Glicko-2 math appears solid
- Persistent league with env signature validation
- Dual-device support for arena opponents (line 461-462 in train_ppo.py)

**Critical Risks**:
- BUG-1: Lack of opponent resampling during training limits diversity
- BUG-2: Rating update logic may cause inconsistencies if opponents are updated
- ISSUE-1: Single opponent per rollout batch reduces effective diversity

**Recommendation**: Address BUG-1 and BUG-2 before running large-scale experiments. The current implementation will work for initial testing but may lead to overfitting or rating drift in prolonged self-play.

---

**End of Review**
