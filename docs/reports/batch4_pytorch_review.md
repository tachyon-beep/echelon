# PyTorch Engineering Review: Simulation Layer

**Reviewers**: Claude Opus 4.5
**Date**: 2025-12-23
**Files Reviewed**:
- `/home/john/echelon/echelon/sim/sim.py` (1280 lines)
- `/home/john/echelon/echelon/sim/mech.py` (72 lines)
- `/home/john/echelon/echelon/agents/heuristic.py` (314 lines)
- `/home/john/echelon/echelon/sim/los.py` (122 lines)
- `/home/john/echelon/echelon/sim/world.py` (250 lines)

---

## Summary

The simulation layer is **correctly implemented as pure NumPy** with no PyTorch dependencies. This is architecturally sound for a DRL environment where the simulation must be framework-agnostic. However, the code exhibits several numerical stability issues, missed vectorization opportunities, and inefficient data structure patterns that impact performance at scale.

**Critical Findings**: 0 bugs, 3 numerical stability issues, 8 optimization opportunities
**Architecture**: Clean separation between sim (NumPy) and RL layers (PyTorch)
**Performance Impact**: Moderate - O(n²) patterns in mech collision detection and target selection could bottleneck at scale

---

## Bugs Found

**None identified.** The code is functionally correct with proper numerical guards in place for most edge cases.

---

## Issues

### 1. Numerical Stability: Division by Zero Guards Inconsistent

**Severity**: Medium
**Files**: `sim.py`, `los.py`, `heuristic.py`

The codebase uses inconsistent epsilon values for division-by-zero protection:

- **sim.py:132**: `mag2 < 1e-9` (conservative)
- **sim.py:93**: `n <= 1e-6` (moderate)
- **sim.py:669**: `np.linalg.norm(delta) + 1e-6` (additive guard)
- **sim.py:1000**: `np.dot(d, d) <= 1e-18` (very tight)
- **los.py:37**: `length <= 1e-9` (conservative)
- **heuristic.py:188**: `n > 1e-6` (moderate)

**Impact**: Inconsistent behavior near singularities. The tight epsilon at sim.py:1000 (1e-18) may cause false positives on float32 operations, while additive guards like `1e-6` don't scale with magnitude.

**Recommendation**: Standardize on magnitude-relative epsilon:
```python
EPS_NORM = 1e-7  # For norm comparisons
EPS_DIV = 1e-9   # For safe division (additive)
```

**References**:
- sim.py:132, 93, 669, 1000, 1010, 1081, 1109
- los.py:37
- heuristic.py:188

---

### 2. Numerical Stability: Float32 to Float64 Conversions Without Justification

**Severity**: Low
**Files**: `sim.py`, `los.py`

Several functions convert float32 data to float64 for intermediate calculations but don't document why:

- **los.py:28-30**: `raycast_voxels` converts inputs to float64 for DDA traversal
  - **Comment says**: "Use float64 for DDA to avoid precision drift"
  - **Good**: Documented reasoning

- **sim.py:997**: `_impact_pos_before_voxel` converts to float64
  - **No comment** on why float64 is needed
  - Only used for 3-axis intersection calc, unclear if precision is critical

**Impact**: Minor performance cost from mixed-precision operations. Without documentation, future maintainers may "optimize away" the conversion.

**Recommendation**: Document precision requirements or use float32 consistently if 64-bit precision isn't needed.

**References**:
- los.py:28-30 (documented, good)
- sim.py:997-998 (undocumented)

---

### 3. Numerical Stability: Angle Wrapping Not Vectorized

**Severity**: Low
**Files**: `sim.py:53`, `heuristic.py:14`

The `_wrap_pi` function appears twice (duplicated) and operates on scalars:

```python
def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
```

**Issues**:
1. Not vectorized - can't wrap arrays of angles efficiently
2. Duplicated in two files (DRY violation)
3. Modulo on floats can accumulate errors over many wraps

**Impact**: Minor - only called once per mech per tick. But duplication risks divergence.

**Recommendation**:
- Move to shared utility module
- Vectorize: `np.arctan2(np.sin(angles), np.cos(angles))` (numerically robust)
- Or use `(angles + np.pi) % (2*np.pi) - np.pi` if vectorization is needed

**References**:
- sim.py:53-56
- heuristic.py:14-15

---

### 4. Performance: O(n²) Mech Collision Detection

**Severity**: High
**Files**: `sim.py:189-208`

The `_collides_mechs` function checks every other mech for AABB overlap:

```python
def _collides_mechs(self, mech: MechState, pos: np.ndarray) -> bool:
    # ... (lines 189-208)
    for other in self.mechs.values():
        if other.mech_id == mech.mech_id or not other.alive:
            continue
        # AABB intersection test
```

Called from `_collides_any` → `_integrate` → for each mech → for each axis (X, Y, Z) → up to 3 trials per axis due to step-up logic.

**Complexity**: O(n²) where n = number of living mechs (up to 20 per pack, 40+ in multi-pack scenarios)

**Impact**:
- At 20 mechs: ~400 AABB checks per physics substep
- At 40 mechs: ~1600 checks
- With 5 substeps per decision: 2000-8000 checks per agent action

**Recommendation**: Spatial hashing or broad-phase with grid bucketing:
```python
# Pseudo-code
grid_size = max(mech_sizes)
buckets = defaultdict(list)
for mech in living_mechs:
    bucket_key = (int(mech.pos[0] // grid_size), int(mech.pos[1] // grid_size))
    buckets[bucket_key].append(mech)

# Only check mechs in same/adjacent buckets
```

**References**: sim.py:189-208, called from sim.py:210-215, 285-300, 318

---

### 5. Performance: O(n²) Target Selection in Weapon Fire

**Severity**: Medium
**Files**: `sim.py:518-543, 598-658, 793-810`

Each weapon fire function (`_try_fire_laser`, `_try_fire_missile`, `_try_paint`) iterates all mechs to find valid targets:

```python
for target in self.mechs.values():
    if not target.alive or target.team == shooter.team:
        continue
    delta = target.pos - shooter.pos
    dist = float(np.linalg.norm(delta))
    # ... range/arc checks
```

**Complexity**: O(n*m) where n = shooters, m = potential targets

**Impact**:
- At 20 mechs per side: 20 * 20 = 400 distance calculations per weapon type
- 4 weapon types checked per tick → 1600 norm calls
- Each norm is a sqrt: relatively expensive

**Recommendation**:
1. **Pre-filter by team**: Maintain `self.enemies[team]` lists updated on death
2. **Vectorize distance calc**: Compute all distances at once
   ```python
   enemy_positions = np.array([e.pos for e in enemies])  # [n, 3]
   deltas = enemy_positions - shooter.pos  # broadcast
   dists = np.linalg.norm(deltas, axis=1)  # single vectorized call
   ```
3. **Cache sensor range**: Only check mechs within max weapon range

**References**:
- sim.py:527-543 (laser)
- sim.py:637-657 (missile)
- sim.py:793-810 (paint)
- Also: sim.py:870-880 (kinetic)

---

### 6. Performance: Redundant `np.linalg.norm` Calls

**Severity**: Low
**Files**: `sim.py` (multiple locations)

Distance vectors are normalized twice in several functions:

**Example** (sim.py:668-674):
```python
delta = target.pos - shooter.pos
delta_norm = delta / (np.linalg.norm(delta) + 1e-6)  # norm #1
# ... later ...
vel /= np.linalg.norm(vel)  # norm #2
```

**sim.py:1075-1081** (projectile homing):
```python
delta = target.pos - p.pos
dist = np.linalg.norm(delta)  # norm #1
if dist > 0.1:
    desired = (delta / dist) * p.speed  # division by dist
    # ...
    p.vel = (p.vel / (np.linalg.norm(p.vel) + 1e-6)) * p.speed  # norm #2
```

**Impact**: `np.linalg.norm` is sqrt-based. Redundant calls waste ~10-20% of vector ops.

**Recommendation**: Cache norm results:
```python
delta = target.pos - p.pos
dist = np.linalg.norm(delta)
if dist > 0.1:
    delta_norm = delta / dist  # reuse dist
    desired = delta_norm * p.speed
```

**References**: sim.py:668-674, 1075-1081, 669, 728, 886-893

---

### 7. Performance: String Parsing in Hot Loop

**Severity**: Low
**Files**: `sim.py:59-67, 70-77`, `heuristic.py:18-26`

The `_pack_index` function parses mech IDs via string ops:

```python
def _pack_index(mech: MechState) -> int | None:
    _, sep, suffix = mech.mech_id.rpartition("_")
    if not sep:
        return None
    try:
        idx = int(suffix)
    except ValueError:
        return None
    return int(idx) // int(PACK_SIZE)
```

Called from:
- `_same_pack` (sim.py:70) → called in `_get_paint_bonus` (sim.py:438)
- `_lock_type` for missile paint locks (sim.py:615)
- Heuristic policy on every action (heuristic.py:231-233)

**Impact**: String ops in hot paths. Minor but unnecessary overhead.

**Recommendation**:
1. Add `pack_id: int` field to `MechState` (computed once at init)
2. Or cache results in a dict: `self._pack_cache[mech_id]`

**References**:
- sim.py:59-67 (definition)
- sim.py:70-77 (usage in `_same_pack`)
- sim.py:438, 615 (calls)
- heuristic.py:18-26, 231-233

---

### 8. Performance: Smoke LOS Check is O(n_clouds) per Ray

**Severity**: Low
**Files**: `sim.py:127-148`

The `has_smoke_los` function iterates all smoke clouds for every LOS check:

```python
def has_smoke_los(self, start_xyz: np.ndarray, end_xyz: np.ndarray) -> bool:
    # ... (lines 127-148)
    for cloud in self.smoke_clouds:
        if not cloud.alive:
            continue
        # ray-sphere intersection
```

**Complexity**: O(n_rays * n_clouds)

**Impact**:
- Current: Max ~5-10 clouds typical → minor
- Future: If smoke becomes common (e.g., 50+ clouds), this becomes a bottleneck

**Recommendation**:
- Filter dead clouds periodically (e.g., every 10 ticks)
- Spatial hash if cloud count scales
- Vectorize intersection tests if needed

**References**: sim.py:127-148, called from sim.py:150

---

### 9. Data Structure: Lists for Living Mechs

**Severity**: Low
**Files**: `sim.py:118-119`

```python
def living_mechs(self) -> list[MechState]:
    return [m for m in self.mechs.values() if m.alive]
```

Called frequently (e.g., in `step` at sim.py:1262). List comprehension allocates new list every call.

**Recommendation**: Cache living mechs and invalidate on death:
```python
self._living_cache = None

def living_mechs(self) -> list[MechState]:
    if self._living_cache is None:
        self._living_cache = [m for m in self.mechs.values() if m.alive]
    return self._living_cache

def _handle_death(self, target, shooter_id):
    # ...
    self._living_cache = None  # invalidate
```

**References**: sim.py:118-119

---

## Improvement Opportunities

### 1. Vectorization: Cooldown Updates

**Files**: `sim.py:1217-1223`

Cooldown timers are updated one mech at a time:

```python
for mech in self.mechs.values():
    if not mech.alive:
        continue
    mech.laser_cooldown = max(0.0, float(mech.laser_cooldown - self.dt))
    mech.missile_cooldown = max(0.0, float(mech.missile_cooldown - self.dt))
    # ... 4 more cooldowns
```

**Opportunity**: Batch as NumPy arrays:
```python
# Store cooldowns as arrays: self.cooldowns[mech_idx, cooldown_type]
alive_mask = self.alive_mask  # bool array
self.cooldowns[alive_mask] = np.maximum(0.0, self.cooldowns[alive_mask] - self.dt)
```

**Benefit**: ~5x faster for 20+ mechs (vectorized max vs. Python loop)

**Caveat**: Requires refactoring `MechState` to array-of-structs or struct-of-arrays pattern.

---

### 2. Vectorization: Projectile Integration

**Files**: `sim.py:1023-1199`

Projectiles are updated one at a time in a Python loop. Physics (velocity, guidance) could be batched:

```python
# Current: for p in self.projectiles: p.vel[2] -= g * dt
# Vectorized:
velocities = np.array([p.vel for p in self.projectiles])  # [n, 3]
velocities[:, 2] -= g * self.dt
for i, p in enumerate(self.projectiles):
    p.vel = velocities[i]
```

**Benefit**: 3-10x faster for 10+ projectiles (typical missile volleys)

**Caveat**: Collision detection still needs per-projectile raycasts (harder to vectorize).

---

### 3. Batched Computation: Gravity Application

**Files**: `sim.py:220-221, 252-254`

Gravity constant is recomputed per mech:

```python
g_vox = 9.81 / float(self.world.voxel_size_m)
mech.vel[2] = float(mech.vel[2] - g_vox * self.dt)
```

**Opportunity**: Precompute `self.g_vox` in `__init__` and apply vectorized:
```python
# Init:
self.g_vox_per_dt = (9.81 / self.world.voxel_size_m) * self.dt

# Apply:
for mech in mechs:
    mech.vel[2] -= self.g_vox_per_dt
```

**Benefit**: Eliminates repeated division (minor, ~2% speedup)

---

### 4. Memory: Projectile Filtering Creates Garbage

**Files**: `sim.py:1025, 1198`

```python
keep = []
for p in self.projectiles:
    # ... logic ...
    keep.append(p)
self.projectiles = keep
```

Creates new list every substep. With 10 projectiles and 5 substeps → 50 list allocations per decision tick.

**Opportunity**: In-place filtering:
```python
# Mark dead: p.alive = False
# Periodic cleanup:
if self.tick % 10 == 0:
    self.projectiles = [p for p in self.projectiles if p.alive]
```

Or use fixed-size array with active indices.

---

### 5. Heuristic: Redundant Distance Calculations

**Files**: `heuristic.py:136-148`

```python
for other in sim.mechs.values():
    # ...
    d = other.pos - mech.pos
    dist = float(np.linalg.norm(d))
```

Then later (line 172):
```python
move_delta = move_target - mech.pos
```

The target distance is computed again at line 199 for range checks.

**Opportunity**: Cache enemy distances in a dict:
```python
enemy_dists = {e.mech_id: np.linalg.norm(e.pos - mech.pos) for e in enemies}
best = min(enemy_dists.items(), key=lambda x: x[1])
```

---

### 6. Heuristic: Stuck Detection State Allocation

**Files**: `heuristic.py:76-83`

Per-mech state is stored in a dict:
```python
if mech_id not in self.states:
    self.states[mech_id] = {"last_pos": ..., "stuck_counter": 0, ...}
```

**Issue**: Grows unbounded if mech IDs change (e.g., between episodes).

**Opportunity**: Clear state on episode reset or use WeakValueDictionary.

---

### 7. Float Casting Overhead

**Files**: `sim.py` (pervasive)

Excessive `float()` casts on values already float32:

```python
mech.heat = float(mech.heat + LAVA_HEAT_PER_S * self.dt)  # line 459
mech.vel[0] = float(desired[0])  # line 248
```

**Issue**: NumPy float32 → Python float (float64) → back to float32 when assigned. Unnecessary precision promotion.

**Opportunity**: Remove casts or use `np.float32()` if type enforcement is needed.

**Note**: Some casts are for JSON serialization (events) - those are justified.

---

### 8. AABB Collision: Could Use NumPy Fancy Indexing

**Files**: `sim.py:199-207, world.py:226-243`

AABB overlap is checked with explicit conditionals:
```python
if (a_min[0] < b_max[0] and a_max[0] > b_min[0] and
    a_min[1] < b_max[1] and a_max[1] > b_min[1] and
    a_min[2] < b_max[2] and a_max[2] > b_min[2]):
```

**Opportunity**: Vectorize when checking many AABBs:
```python
overlaps = np.all((aabbs_min < target_max) & (aabbs_max > target_min), axis=1)
```

For single checks, branching is fine. But in `_collides_mechs`, vectorizing could help.

---

## Architecture Notes

### Positive: Clean Sim/RL Separation

The simulation layer uses **pure NumPy** with zero PyTorch dependencies. This is the correct design:
- Sim can be used standalone (replays, testing)
- RL model in `/echelon/rl/model.py` uses PyTorch separately
- No tensor conversion overhead in hot paths

### Positive: Float32 Discipline

All position/velocity arrays are `float32`, matching typical RL observation precision. This is good for:
- Memory efficiency (half the size of float64)
- GPU compatibility if sim is ever ported to torch/jax

### Note: No Torch Usage in Sim

Verified via grep: `import torch` only appears in:
- `echelon/rl/model.py` (expected)
- `scripts/train_ppo.py` (expected)
- Test files (expected)

**Recommendation**: Keep it this way. Don't introduce torch to sim layer.

---

## Recommendations Priority

### High Priority (Performance)
1. **Spatial hash for mech collisions** (Issue #4) - O(n²) → O(n) at scale
2. **Pre-filter enemies by team** (Issue #5) - Reduces target search space by 50%
3. **Cache pack indices** (Issue #7) - Eliminate string parsing in hot loop

### Medium Priority (Numerical)
4. **Standardize epsilon values** (Issue #1) - Prevent future NaN bugs
5. **Vectorize cooldown updates** (Opportunity #1) - Clean code + 5x speedup
6. **Document float64 conversions** (Issue #2) - Maintainability

### Low Priority (Code Quality)
7. **Deduplicate `_wrap_pi`** (Issue #3) - DRY principle
8. **Cache living mechs** (Issue #9) - Minor allocation savings
9. **Remove excessive float() casts** (Opportunity #7) - Cleaner code

---

## Testing Recommendations

1. **Numerical Stability Tests**: Add unit tests for edge cases:
   - Zero-length vectors in `_angle_between_yaw`
   - Overlapping mechs in `_collides_mechs`
   - Projectiles spawned at exact target position

2. **Performance Benchmarks**: Profile with 40+ mechs to measure O(n²) impact:
   ```bash
   PYTHONPATH=. uv run pytest tests/performance -k collision
   ```

3. **Vectorization Validation**: If implementing vectorized ops, ensure:
   - Results match scalar version (use `np.allclose` with `atol=1e-6`)
   - No NaN/Inf propagation on edge cases

---

## Conclusion

The simulation layer is **well-architected** with clean NumPy implementation, but exhibits **performance bottlenecks** at scale due to O(n²) collision detection and target selection. Numerical stability is generally sound but inconsistent epsilon values risk future issues.

**Immediate Actions**:
1. Profile with 40-mech scenarios to quantify O(n²) impact
2. Implement spatial hashing for collision broad-phase
3. Standardize division-by-zero guards

**Long-term**:
- Consider struct-of-arrays (SoA) for MechState to enable full vectorization
- Benchmark numpy vs. numba JIT for hot loops (raycast, collision)

The code is **production-ready** for current scale (10v10) but will need optimization for larger battles (20v20+).
