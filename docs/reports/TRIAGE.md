# Bug Triage: Consolidated Review Findings

**Generated:** 2025-12-23
**Sources:** batch1-4 DRL and PyTorch reviews

---

## Summary

| Severity | Count | Description |
|----------|-------|-------------|
| **CRITICAL** | 6 | Crashes, security vulnerabilities, training corruption |
| **HIGH** | 15 | Correctness bugs, major performance issues, learning blockers |
| **MEDIUM** | 18 | Suboptimal behavior, moderate performance, code quality |
| **LOW** | 12 | Nice-to-have improvements, minor optimizations |

---

## CRITICAL (Fix Immediately)

### CRIT-1: Triple Tensor Conversion in Rollout Loop
- **File:** `scripts/train_ppo.py:873-882`
- **Source:** Batch 1 DRL + PyTorch
- **Description:** Same tensor conversions performed 3 times consecutively
```python
rewards_buf[step] = torch.from_numpy(rewards_flat).to(device)  # Line 873
next_done = torch.from_numpy(done_flat).to(device)             # Line 874
next_obs = torch.from_numpy(stack_obs_many(...)).to(device)    # Line 876

rewards_buf[step] = torch.from_numpy(rewards_flat).to(device)  # Line 878 DUPLICATE
next_done = torch.from_numpy(done_flat).to(device)             # Line 879 DUPLICATE
next_obs = torch.from_numpy(stack_obs_many(...)).to(device)    # Line 880 DUPLICATE

next_obs = torch.from_numpy(stack_obs_many(...)).to(device)    # Line 882 TRIPLICATE
```
- **Impact:** 3x memory allocations, 2x wasted `stack_obs_many` calls per rollout step. With 512 rollout steps, significant waste.
- **Fix:** Delete lines 878-882 entirely.

---

### CRIT-2: Missing LSTM State Detachment in Value Bootstrap
- **File:** `scripts/train_ppo.py:885`
- **Source:** Batch 1 DRL
- **Description:** LSTM state passed to value bootstrapping may retain computation graph
```python
with torch.no_grad():
    next_value, _ = model.get_value(next_obs, lstm_state, next_done)
    # lstm_state was created with gradients enabled (line 744)
```
- **Impact:** Potential memory leak; computation graph retains rollout history. Can cause OOM in long training runs.
- **Fix:**
```python
with torch.no_grad():
    next_value, _ = model.get_value(
        next_obs,
        LSTMState(h=lstm_state.h.detach(), c=lstm_state.c.detach()),
        next_done
    )
```

---

### CRIT-3: Unsafe Checkpoint Loading (Security Vulnerability)
- **Files:**
  - `echelon/arena/match.py:39`
  - `scripts/arena.py:52,102,125`
  - `scripts/train_ppo.py:474,567`
- **Source:** Batch 3 PyTorch
- **Description:** All `torch.load()` calls lack `weights_only=True`, enabling arbitrary code execution via malicious pickle files.
```python
ckpt = torch.load(ckpt_path, map_location=device)  # VULNERABLE
```
- **Impact:** Remote code execution if loading untrusted checkpoints (especially dangerous in arena mode with external opponents).
- **Fix:** Add `weights_only=True` to ALL `torch.load()` calls:
```python
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
```

---

### CRIT-4: Missing EWAR_DIM Class Attribute
- **File:** `echelon/env/env.py:1005`
- **Source:** Batch 2 DRL + PyTorch
- **Description:** Error message references undefined `self.EWAR_DIM`
```python
raise ValueError(
    f"action[{aid!r}] has size {a.size}, expected {self.ACTION_DIM} "
    f"(base={self.BASE_ACTION_DIM}, target={self.TARGET_DIM}, ewar={self.EWAR_DIM}, "
    #                                                                    ^^^^^^^^^^
    f"obs_ctrl={self.OBS_CTRL_DIM}, comm_dim={self.comm_dim})"
)
```
- **Impact:** When action dimension is wrong, crashes with `AttributeError` instead of helpful message.
- **Fix:** Either add `EWAR_DIM = 0` as class constant, or remove from error message.

---

### CRIT-5: Opponent Never Resampled During Training Rollouts
- **File:** `scripts/train_ppo.py:766-778,869-871`
- **Source:** Batch 3 DRL
- **Description:** In arena mode, opponent is sampled once at initialization and kept for entire rollout.
```python
# Line 869-871 comment:
# Currently we sample one opponent for the WHOLE vector.
# If an episode ends, we'll just keep the current opponent until the next update
```
- **Impact:** Training data comes from single frozen opponent per rollout. Limits diversity, causes overfitting to specific opponent strategies.
- **Fix:** Resample opponent after each episode ends:
```python
if ep_over and args.train_mode == "arena":
    _arena_sample_opponent(reset_hidden=False)
```

---

### CRIT-6: Double Rating Update Vulnerability
- **File:** `scripts/arena.py:170-175,194-195` and `scripts/train_ppo.py:1196-1201`
- **Source:** Batch 3 DRL
- **Description:** Both players in a match have ratings added to results dict, then `apply_rating_period` updates all. If opponents play each other in overlapping sessions, circular dependency occurs.
- **Impact:** Glicko-2 rating corruption when opponents are also being updated. Ratings become inconsistent.
- **Fix:** Implement two-phase rating updates (snapshot all ratings first, then commit atomically).

---

## HIGH PRIORITY

### HIGH-1: No Value Function Normalization
- **File:** `scripts/train_ppo.py:887-896,920`
- **Source:** Batch 1 DRL
- **Description:** Advantages are normalized but returns are not. Returns can grow unbounded with gamma=0.99.
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
# returns NOT normalized!
v_loss = 0.5 * (returns - new_values).pow(2).mean()
```
- **Impact:** Value function struggles to converge; large gradients can dominate policy gradients.
- **Fix:** Implement running normalization for returns (CleanRL-style `RunningMeanStd`).

---

### HIGH-2: Suboptimal Action Distribution (Fixed Log-Std)
- **File:** `echelon/rl/model.py:42,87-88`
- **Source:** Batch 1 DRL
- **Description:** Log-std is shared across all action dimensions (movement, weapons, target selection).
```python
self.actor_logstd = nn.Parameter(torch.zeros(self.action_dim))
logstd = self.actor_logstd.expand_as(mean)
```
- **Impact:** Same exploration noise for all actions. Overexploration in some dimensions, underexploration in others.
- **Fix:** Use state-dependent std: `self.actor_logstd = nn.Linear(lstm_hidden_dim, action_dim)`

---

### HIGH-3: Tanh Squashing Epsilon Inconsistency
- **File:** `echelon/rl/model.py:9-11,99`
- **Source:** Batch 1 DRL + PyTorch
- **Description:** `_atanh` clamps to 0.999, but log correction uses 1e-6 epsilon. When actions saturate to ±1.0:
```python
x = torch.clamp(x, -0.999, 0.999)  # _atanh
logprob = ... - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)  # Different epsilon
```
- **Impact:** +13.8 bias in log-prob when actions saturate. PPO ratio becomes unstable.
- **Fix:** Use consistent epsilon (1e-6) in both places.

---

### HIGH-4: Entropy Bonus Applied to Pre-Squashing Distribution
- **File:** `echelon/rl/model.py:100`
- **Source:** Batch 1 DRL
- **Description:** Entropy computed on Gaussian, not the squashed distribution.
```python
entropy = dist.entropy().sum(-1)  # dist is Normal, not TanhNormal
```
- **Impact:** Entropy bonus is biased. When actions saturate, true entropy is lower than reported.
- **Fix:** Apply squashing correction: `entropy = base_entropy + log(1 - tanh(u)^2).sum(-1)`

---

### HIGH-5: No Gym Space Definitions
- **File:** `echelon/env/env.py:112-210`
- **Source:** Batch 2 DRL
- **Description:** `EchelonEnv` lacks `observation_space` and `action_space` attributes (Gym API requirement).
- **Impact:** Incompatible with RL libraries (SB3, RLlib, CleanRL). Can't use standard wrappers.
- **Fix:** Add Box/Dict spaces in `__init__`.

---

### HIGH-6: Extremely Sparse Reward Signal
- **File:** `echelon/env/env.py:1244-1314`
- **Source:** Batch 4 DRL
- **Description:** Only two reward components: approach shaping (0.25) and zone control (0.10). No combat rewards.
- **Impact:** Agents don't learn weapon usage, target selection, or tactics. Combat skills must emerge from terminal feedback only.
- **Fix:** Add combat shaping: damage (+0.005/point), kills (+1.0), assists (+0.5), death penalty (-2.0).

---

### HIGH-7: Breadcrumb Shaping Stops Prematurely
- **File:** `echelon/env/env.py:1302-1308`
- **Source:** Batch 2 DRL + Batch 4 DRL
- **Description:** Approach reward stops when ANY teammate reaches zone.
```python
if not team_reached_zone[m.team]:
    # Apply approach shaping
```
- **Impact:** Scouts reach first, then other mechs lose navigation signal. Encourages death-charging.
- **Fix:** Use per-agent zone status or exponential decay based on teammates in zone.

---

### HIGH-8: Heuristic Baseline is Exploitable
- **File:** `echelon/agents/heuristic.py:29-313`
- **Source:** Batch 4 DRL
- **Description:** Multiple weaknesses:
  - Always targets nearest enemy (no prioritization)
  - Fixed 5.5 voxel engagement range (suboptimal for all classes)
  - No smoke usage
  - Naive handicap (random weapon suppression)
- **Impact:** RL agents easily exploit by staying at range. No pressure to learn advanced tactics.
- **Fix:** Implement skill levels (trivial→expert) with progressive capabilities.

---

### HIGH-9: Model Caching Creates Expensive Env Instances
- **File:** `echelon/arena/match.py:42-46`
- **Source:** Batch 3 PyTorch
- **Description:** `load_policy()` creates full `EchelonEnv` just to infer obs dimensions.
```python
env = EchelonEnv(env_cfg)
obs, _ = env.reset(seed=int(env_cfg.seed or 0))  # Full terrain generation!
```
- **Impact:** 0.5-2s overhead per opponent load. 20 opponents = 10-40s wasted.
- **Fix:** Store `obs_dim`/`action_dim` in checkpoint metadata.

---

### HIGH-10: Unbounded Opponent Cache Memory
- **File:** `scripts/arena.py:119`, `scripts/train_ppo.py:577`
- **Source:** Batch 3 PyTorch
- **Description:** `opponent_models` cache never releases models.
- **Impact:** 50 opponents = 25-100MB+ persistent allocation. CUDA OOM risk.
- **Fix:** Implement LRU cache with max size.

---

### HIGH-11: O(n²) Mech Collision Detection
- **File:** `echelon/sim/sim.py:189-208`
- **Source:** Batch 4 PyTorch
- **Description:** `_collides_mechs` checks every other mech for AABB overlap.
- **Impact:** At 40 mechs: ~1600 checks per physics substep. Bottleneck at scale.
- **Fix:** Spatial hashing or grid bucketing for broad-phase.

---

### HIGH-12: O(n²) Target Selection in Weapon Fire
- **File:** `echelon/sim/sim.py:518-543,598-658,793-810`
- **Source:** Batch 4 PyTorch
- **Description:** Each weapon fire iterates all mechs to find targets.
- **Impact:** 400 distance calculations per weapon type per tick.
- **Fix:** Pre-filter by team, vectorize distance calculation.

---

### HIGH-13: Missing detach() in Rollout Collection
- **File:** `scripts/train_ppo.py:753,295`
- **Source:** Batch 1 PyTorch
- **Description:** Action tensors not detached before numpy conversion.
```python
action_np = action.cpu().numpy()  # Missing detach()
```
- **Impact:** Potential memory overhead if gradient tracking accidentally enabled.
- **Fix:** `action_np = action.detach().cpu().numpy()`

---

### HIGH-14: Checkpoint Loading Without strict=True
- **File:** `scripts/train_ppo.py:592,569`
- **Source:** Batch 1 PyTorch
- **Description:** `load_state_dict` called without strict mode.
- **Impact:** Silent failures on checkpoint/model mismatch.
- **Fix:** `model.load_state_dict(ckpt["model_state"], strict=True)`

---

### HIGH-15: Missing Inference Mode in Match Playback
- **File:** `echelon/arena/match.py:61-113`
- **Source:** Batch 3 PyTorch
- **Description:** Uses `@torch.no_grad()` instead of `@torch.inference_mode()`.
- **Impact:** Misses 5-10% speedup from inference mode optimizations.
- **Fix:** `@torch.inference_mode()`

---

## MEDIUM PRIORITY

### MED-1: Observation Normalization Issues
- **File:** `echelon/env/env.py:470,870,722`
- **Source:** Batch 2 DRL
- **Description:** Inconsistent normalization:
  - Velocity divided by hardcoded 10.0 (max is ~7 m/s)
  - Acoustic intensities use magic constant 5.0
  - Heat clipped to 2.0 (inconsistent with [0,1] convention)
- **Fix:** Document ranges, use config-driven normalization.

---

### MED-2: Termination vs Truncation Semantics Wrong
- **File:** `echelon/env/env.py:1316-1360`
- **Source:** Batch 2 DRL
- **Description:** Zone control win sets BOTH `terminations` AND `truncations` to True.
- **Impact:** Gym semantics say terminated + truncated shouldn't both be true. Affects value bootstrapping.
- **Fix:** Zone win = termination only. Time limit = truncation only.

---

### MED-3: Dead Agent Reward Handling
- **File:** `echelon/env/env.py:1295-1297`
- **Source:** Batch 4 DRL
- **Description:** Dead agents get 0.0 reward for all remaining steps.
- **Impact:** No credit assignment for pre-death contributions. Terminal outcome not backpropagated.
- **Fix:** Distribute terminal reward to dead agents at episode end.

---

### MED-4: Non-Deterministic Debris Placement
- **File:** `echelon/sim/sim.py:356`
- **Source:** Batch 4 DRL
- **Description:** Per-voxel debris type uses `self.rng.random()`.
- **Impact:** Breaks deterministic replay from seed.
- **Fix:** Use hash of mech_id + position for deterministic debris type.

---

### MED-5: Heuristic Stuck Detection Frame Dependency
- **File:** `echelon/agents/heuristic.py:118-133`
- **Source:** Batch 4 DRL
- **Description:** Stuck counter uses decision step count, not time.
- **Impact:** Behavior changes with `decision_repeat` or `dt_sim`.
- **Fix:** Track accumulated time in seconds.

---

### MED-6: Observable State Leaks Velocity During Shutdown
- **File:** `echelon/sim/sim.py:218-223`, `echelon/env/env.py:870`
- **Source:** Batch 4 DRL
- **Description:** Shutdown mechs can't control velocity but it's in observation.
- **Impact:** Agents may learn spurious correlations with uncontrollable state.
- **Fix:** Zero out velocity in obs during shutdown.

---

### MED-7: Arena Opponent Sampling is Uniform Random Only
- **File:** `scripts/train_ppo.py:575`, `scripts/arena.py:135`
- **Source:** Batch 3 DRL
- **Description:** No prioritized/curriculum opponent sampling.
- **Impact:** Inefficient learning (may fight too-easy or too-hard opponents).
- **Fix:** Implement PFSP or rating-based sampling.

---

### MED-8: Conservative Score Too Aggressive
- **File:** `echelon/arena/league.py:189-190`
- **Source:** Batch 3 DRL
- **Description:** Uses μ-2σ for promotion ranking (2.5th percentile).
- **Impact:** Strong but uncertain candidates may not promote.
- **Fix:** Make conservatism factor configurable.

---

### MED-9: LSTM State Not Reset When Opponent Pool Refreshes
- **File:** `scripts/train_ppo.py:867-871`
- **Source:** Batch 3 DRL
- **Description:** After `_arena_refresh_pool()`, no resample or LSTM reset.
- **Impact:** Opponent state may be stale after pool changes.
- **Fix:** Call `_arena_sample_opponent(reset_hidden=True)` after refresh.

---

### MED-10: Redundant astype Calls Throughout
- **Files:** `echelon/env/env.py:467,470,870,946,1012-1013`
- **Source:** Batch 2 PyTorch
- **Description:** Defensive `astype(np.float32, copy=False)` when data is already float32.
- **Impact:** Function call overhead in hot paths.
- **Fix:** Remove unnecessary casts, add dtype assertions at initialization.

---

### MED-11: Telemetry Downsampling Not Cached
- **File:** `echelon/env/env.py:647-671`
- **Source:** Batch 2 PyTorch
- **Description:** Static terrain downsampling computed every `_obs()` call.
- **Impact:** 16×16 loop with np.any() per step (terrain is static).
- **Fix:** Cache `telemetry_flat` at reset.

---

### MED-12: Occupancy Map Not Shared Across Agents
- **File:** `echelon/env/env.py:558-601`
- **Source:** Batch 2 PyTorch
- **Description:** `_local_map` computes `occupancy_2d` per agent.
- **Impact:** 20 agents × ~100KB allocation per step.
- **Fix:** Compute once in `_obs()` and pass to `_local_map`.

---

### MED-13: Device Mismatch Risk in LSTM State
- **File:** `echelon/arena/match.py:77-81,97-98`
- **Source:** Batch 3 PyTorch
- **Description:** LSTM state created on `device` param, but policies may be on different devices.
- **Impact:** Device mismatch error if policies are on different GPUs.
- **Fix:** Use `next(policy.parameters()).device` for each policy.

---

### MED-14: Numerical Epsilon Values Inconsistent
- **Files:** `sim.py:132,93,669,1000`, `los.py:37`, `heuristic.py:188`
- **Source:** Batch 4 PyTorch
- **Description:** Epsilon values range from 1e-6 to 1e-18 across codebase.
- **Impact:** Inconsistent behavior near singularities.
- **Fix:** Standardize: `EPS_NORM=1e-7` for norms, `EPS_DIV=1e-9` for division.

---

### MED-15: _wrap_pi Duplicated
- **Files:** `sim.py:53`, `heuristic.py:14`
- **Source:** Batch 4 PyTorch
- **Description:** Same function defined in two files.
- **Impact:** DRY violation, risk of divergence.
- **Fix:** Move to shared utility module.

---

### MED-16: No Gradient Clipping Diagnostics
- **File:** `scripts/train_ppo.py:926`
- **Source:** Batch 1 DRL
- **Description:** Gradient norm is clipped but actual norm never logged.
- **Impact:** Can't diagnose exploding/vanishing gradients.
- **Fix:** `grad_norm = nn.utils.clip_grad_norm_(...)` and log it.

---

### MED-17: Missing cuDNN Autotuner
- **File:** `scripts/train_ppo.py` (missing)
- **Source:** Batch 1 PyTorch
- **Description:** No `torch.backends.cudnn.benchmark = True`.
- **Impact:** 10-30% slower LSTM ops.
- **Fix:** Add after device setup.

---

### MED-18: Non-Atomic Checkpoint Saves
- **File:** `scripts/train_ppo.py:1070,1032`
- **Source:** Batch 1 PyTorch
- **Description:** `torch.save(ckpt, p)` is not atomic.
- **Impact:** Crash during save = corrupted checkpoint.
- **Fix:** Save to `.tmp` then atomic rename.

---

## LOW PRIORITY

### LOW-1: eval_policy.py Missing no_grad Context
- **File:** `scripts/eval_policy.py:82`
- **Source:** Batch 1 DRL
- **Fix:** Wrap in `torch.no_grad()`.

---

### LOW-2: Contact Tuple Stores Unused dist
- **File:** `echelon/env/env.py:763-776`
- **Source:** Batch 2 PyTorch
- **Fix:** Remove redundant `dist` from tuple.

---

### LOW-3: Action Clipping In-Place May Modify Caller
- **File:** `echelon/env/env.py:1014`
- **Source:** Batch 2 PyTorch
- **Fix:** Make defensive copy before clipping.

---

### LOW-4: String Parsing in Hot Loop for Pack Index
- **File:** `sim.py:59-67`, `heuristic.py:18-26`
- **Source:** Batch 4 PyTorch
- **Fix:** Add `pack_id` field to MechState or cache results.

---

### LOW-5: Smoke LOS Check is O(n_clouds)
- **File:** `echelon/sim/sim.py:127-148`
- **Source:** Batch 4 PyTorch
- **Fix:** Filter dead clouds periodically, spatial hash if scales.

---

### LOW-6: Lists for Living Mechs Rebuilt Each Call
- **File:** `echelon/sim/sim.py:118-119`
- **Source:** Batch 4 PyTorch
- **Fix:** Cache and invalidate on death.

---

### LOW-7: Redundant np.linalg.norm Calls
- **Files:** `sim.py:668-674,1075-1081`
- **Source:** Batch 4 PyTorch
- **Fix:** Cache norm results within function.

---

### LOW-8: Float Casting Overhead
- **File:** `sim.py` (pervasive)
- **Source:** Batch 4 PyTorch
- **Fix:** Remove unnecessary `float()` casts.

---

### LOW-9: Acoustic Intensity Calculation Not Vectorized
- **File:** `echelon/env/env.py:692-722`
- **Source:** Batch 2 PyTorch
- **Fix:** Vectorize distance/intensity calculation.

---

### LOW-10: GAE Accumulation Could Use float64
- **File:** `scripts/train_ppo.py:226-249`
- **Source:** Batch 1 PyTorch
- **Fix:** Use double precision for long rollouts (>1000 steps).

---

### LOW-11: Optimizer State Cleanup on Device Transfer
- **File:** `scripts/train_ppo.py:156-160`
- **Source:** Batch 1 PyTorch
- **Fix:** Clear old device memory after transfer.

---

### LOW-12: Missing LSTM Weight Initialization
- **File:** `echelon/rl/model.py:40`
- **Source:** Batch 1 PyTorch
- **Fix:** Add explicit orthogonal init for LSTM weights.

---

## Recommended Fix Order

### Phase 1: Critical Bugs (Day 1)
1. CRIT-1: Delete duplicate tensor conversions
2. CRIT-2: Detach LSTM state in bootstrap
3. CRIT-3: Add `weights_only=True` to all torch.load
4. CRIT-4: Fix EWAR_DIM in error message

### Phase 2: Training Stability (Day 2-3)
5. HIGH-1: Value normalization
6. HIGH-3: Tanh epsilon consistency
7. HIGH-4: Entropy squashing correction
8. HIGH-13: Detach actions in rollout
9. MED-16: Gradient norm logging

### Phase 3: Self-Play Fixes (Day 3-4)
10. CRIT-5: Resample opponent per episode
11. CRIT-6: Two-phase rating updates
12. HIGH-9: Cache model metadata
13. HIGH-10: LRU opponent cache

### Phase 4: Environment/Reward (Day 4-5)
14. HIGH-5: Add Gym space definitions
15. HIGH-6: Add combat shaping rewards
16. HIGH-7: Fix breadcrumb termination
17. MED-2: Fix termination/truncation semantics
18. MED-3: Dead agent terminal rewards

### Phase 5: Performance (Week 2)
19. HIGH-11: Spatial hash for collisions
20. HIGH-12: Vectorize target selection
21. MED-11: Cache telemetry
22. MED-12: Share occupancy map

### Phase 6: Curriculum/Heuristic (Week 2-3)
23. HIGH-8: Heuristic skill levels
24. MED-7: Prioritized opponent sampling