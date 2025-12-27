# Consolidated Improvement Plan - 2025-12-26

**Source:** 7 Specialist Review Reports (PyTorch + DRL experts)
**Status:** Ready for implementation

---

## Executive Summary

Seven specialist reviews identified **23 actionable improvements** across the Echelon codebase. The highest-impact item is batched LSTM processing in PPO (potential 2-5x speedup). Most items are low-to-medium effort with clear implementation paths.

---

## Priority Tiers

### Tier 1: Critical Performance (Do First)

These items have the highest impact-to-effort ratio.

| # | Item | File(s) | Effort | Impact | Source |
|---|------|---------|--------|--------|--------|
| **1** | **Batch LSTM chunk processing** | `ppo.py` | Medium | 2-5x PPO speedup | PyTorch PPO |
| **2** | **Use `torch.randperm` not `random.shuffle`** | `ppo.py:198` | Low | torch.compile compat | PyTorch PPO |
| **3** | **Add `@torch.no_grad()` to `compute_gae()`** | `rollout.py:71` | Low | Memory + safety | PyTorch Models |

**Details:**

1. **Batch LSTM chunk processing** (ppo.py:215-222)
   - Current: Sequential timestep-by-timestep `for i, t in enumerate(range(start_t, end_t))`
   - Target: Process entire chunk as sequence `[chunk_len, batch, obs_dim]`
   - Requires: New `model.get_chunk_values()` method
   - Impact: LSTM is much faster with batched sequences; most training time is in PPO update

2. **torch.randperm** (ppo.py:198)
   - Current: `random.shuffle(chunk_indices)` causes torch.compile graph breaks
   - Fix: `chunk_indices = torch.randperm(num_chunks, device='cpu').tolist()`

3. **@torch.no_grad() on GAE** (rollout.py:71)
   - GAE computes value targets, not gradients
   - Adding decorator prevents accidental gradient tracking

---

### Tier 2: High Priority Training Improvements

Items that improve training quality or prevent common failures.

| # | Item | File(s) | Effort | Source |
|---|------|---------|--------|--------|
| **4** | Add LR scheduling (linear/cosine decay) | `train_ppo.py` | Low | DRL PPO |
| **5** | Add entropy coefficient annealing | `train_ppo.py`, `ppo.py` | Low | DRL PPO |
| **6** | Skill-matched PFSP opponent sampling | `arena.py`, `league.py` | Medium | DRL Arena |
| **7** | Normalize `sensor_range_mult` to [0,1] | `suite.py` | Low | PyTorch Suite |

**Details:**

4. **LR scheduling**
   - Current: Fixed `lr=3e-4`
   - Add: `torch.optim.lr_scheduler.LinearLR` or `CosineAnnealingLR`
   - Benefit: Prevents late-training instability, improves final performance

5. **Entropy annealing**
   - Current: Fixed `ent_coef=0.01`
   - Target: Start at 0.05, decay to 0.01 over training
   - Add `--ent-coef-start` and `--ent-coef-end` CLI args

6. **PFSP opponent sampling** (arena.py)
   - Current: Uniform random from pool
   - Target: Weight by expected learning signal (skill-matched bell curve)
   ```python
   def sample_opponent(candidate_rating, pool):
       weights = [exp(-(o.rating - candidate_rating)**2 / (2*200**2)) for o in pool]
       return random.choices(pool, weights=weights)[0]
   ```

7. **sensor_range_mult normalization**
   - Current: Range [0.8, 1.5] in suite descriptor
   - Fix: `(sensor_range_mult - 0.8) / 0.7` to normalize to [0, 1]

---

### Tier 3: Medium Priority Improvements

Important but non-urgent refinements.

| # | Item | File(s) | Effort | Source |
|---|------|---------|--------|--------|
| **8** | Pre-allocate chunk buffers outside epoch loop | `ppo.py:206-208` | Low | PyTorch PPO |
| **9** | Per-role advantage normalization | `ppo.py:146` | Medium | DRL PPO |
| **10** | Team reward mixing (alpha blending) | `env.py` | Medium | DRL Env |
| **11** | PBRS-compliant approach shaping | `env.py:2045-2056` | Medium | DRL Env |
| **12** | Exploiter agents in arena | `arena/` | High | DRL Arena |
| **13** | Cold-start warmup for new policies | `arena.py` | Low | DRL Arena |
| **14** | Replace final Tanh with LayerNorm+ReLU | `suite_model.py` | Low | PyTorch Suite |
| **15** | Add NaN detection in training loop | `train_ppo.py` | Low | PyTorch PPO |

**Details:**

8. **Pre-allocate chunk buffers**
   ```python
   # Outside epoch loop (once per update):
   chunk_logprobs = torch.zeros(chunk_length, num_agents, device=device)
   chunk_values = torch.zeros(chunk_length, num_agents, device=device)
   chunk_entropies = torch.zeros(chunk_length, num_agents, device=device)
   # Inside: use slices, clear with .zero_()
   ```

9. **Per-role advantage normalization**
   - Heavy/Medium/Light/Scout may have different reward scales
   - Normalize advantages per-role to prevent role dominance
   - Requires role indices in buffer

10. **Team reward mixing**
    ```python
    r_individual = ...  # current per-agent reward
    r_team = team_rewards[agent.team].mean()
    r_final = alpha * r_individual + (1 - alpha) * r_team
    ```
    - Start alpha=1.0, gradually decrease to encourage cooperation

11. **PBRS-compliant approach shaping**
    - Current: `approach_scale = 0.5**n_teammates_in_zone` violates strict PBRS
    - Fix: Incorporate teammate count into potential function directly
    - Low practical impact, but theoretically cleaner

12. **Exploiter agents**
    - AlphaStar-style exploiters specifically target main agent weaknesses
    - High effort: requires separate training loop targeting specific policies
    - Benefits: Prevents policy collapse, creates curriculum

13. **Cold-start warmup**
    - New policies (high RD) face skill-matched opponents only
    - Prevents new policies from being crushed by established commanders
    ```python
    if candidate.games < 20:
        nearby = [o for o in pool if abs(o.rating - candidate.rating) < 200]
    ```

14. **Replace Tanh activation**
    - `SuiteStreamEncoder.fusion` ends with Tanh, which saturates gradients
    - Replace with LayerNorm + ReLU for better gradient flow

15. **NaN detection**
    ```python
    if torch.isnan(loss):
        raise ValueError(f"NaN loss at update {update}, gradients: {gradients}")
    ```

---

### Tier 4: Low Priority / Future Work

Nice-to-haves and long-term improvements.

| # | Item | File(s) | Effort | Source |
|---|------|---------|--------|--------|
| **16** | NamedTuple for LSTMState (torch.compile) | `model.py`, `suite_model.py` | Low | PyTorch Models |
| **17** | Pre-allocate advantages/returns in RolloutBuffer | `rollout.py` | Low | PyTorch Models |
| **18** | Lazy import wandb in spatial.py | `spatial.py` | Low | PyTorch Models |
| **19** | Consolidate LSTMState dataclass | `model.py`, `suite_model.py` | Low | PyTorch Suite |
| **20** | Commander retirement mechanism | `league.py` | Low | DRL Arena |
| **21** | Extend curriculum (map difficulty, enemy count) | `vec_env.py` | Medium | DRL Env |
| **22** | Add gradient flow tests | `tests/` | Medium | PyTorch Suite |
| **23** | Verify flat_obs_to_suite_obs dimension alignment | `suite_model.py` | Low | DRL Suite |

---

## Implementation Order Recommendation

**Phase 1: Quick Wins (1-2 hours)**
- [x] Items 2, 3, 7, 15 - Simple fixes, immediate benefit

**Phase 2: Performance (2-4 hours)**
- [ ] Item 1 - Batched LSTM processing (biggest impact)
- [ ] Item 8 - Pre-allocate buffers

**Phase 3: Training Dynamics (2-4 hours)**
- [ ] Items 4, 5 - LR and entropy scheduling
- [ ] Item 9 - Per-role normalization (if role imbalance observed)

**Phase 4: Arena Improvements (4-8 hours)**
- [ ] Item 6 - PFSP sampling
- [ ] Item 13 - Cold-start warmup
- [ ] Item 12 - Exploiter agents (stretch goal)

**Phase 5: Cleanup (ongoing)**
- [ ] Items 16-23 as time permits

---

## Cross-Cutting Observations

### What's Working Well
1. **GAE implementation** - Mathematically correct, good done masking
2. **TBPTT chunking** - Proper state caching and detachment
3. **Reward design** - Terminal rewards removed (good), zone shaping is informative
4. **DeepSet architecture** - Correct permutation invariance, proper masking
5. **Glicko-2 rating** - Mathematically sound, two-phase commit correct
6. **Suite-based POMDP** - Elegant role differentiation through observation

### Recurring Themes
1. **torch.compile compatibility** - Several graph break risks identified
2. **Multi-agent credit assignment** - Needs team-level reward mixing
3. **Memory efficiency** - Minor optimization opportunities
4. **Curriculum learning** - Good foundation, could be more sophisticated
5. **Test coverage** - Solid unit tests, some edge cases missing

---

## Specialist Report Cross-References

| Report | Critical | High | Medium | Low |
|--------|----------|------|--------|-----|
| PyTorch Models | 0 | 1 | 1 | 3 |
| DRL Environment | 0 | 1 | 3 | 1 |
| PyTorch PPO | 1 | 1 | 3 | 1 |
| DRL PPO | 0 | 2 | 3 | 2 |
| DRL Arena | 0 | 2 | 3 | 1 |
| PyTorch Suite | 0 | 1 | 2 | 2 |
| DRL Suite | 0 | 0 | 2 | 1 |
| **Total** | **1** | **8** | **17** | **11** |

---

## Appendix: File Impact Summary

Files requiring changes, sorted by number of items:

| File | Items | Priority Range |
|------|-------|----------------|
| `ppo.py` | 1, 2, 8, 9 | Critical-Medium |
| `train_ppo.py` | 4, 5, 15 | High-Medium |
| `arena.py` / `league.py` | 6, 12, 13, 20 | High-Low |
| `rollout.py` | 3, 17 | Critical-Low |
| `suite.py` | 7 | High |
| `suite_model.py` | 14, 19 | Medium-Low |
| `env.py` | 10, 11 | Medium |
| `model.py` | 16, 19 | Low |
| `spatial.py` | 18 | Low |
| `vec_env.py` | 21 | Low |

---

*Generated from specialist reports in `docs/temp_reports/`*
