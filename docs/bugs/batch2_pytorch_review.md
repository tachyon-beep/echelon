# Batch 2: PyTorch/Numpy Review - Environment Layer

**Reviewed Files:**
- `/home/john/echelon/echelon/env/env.py`
- `/home/john/echelon/echelon/actions.py`
- `/home/john/echelon/echelon/config.py`

**Reviewer:** Claude Opus 4.5
**Date:** 2025-12-23

---

## Summary

The environment layer uses **pure numpy** (no PyTorch imports), which is appropriate for a gym-like environment. Overall code quality is good with consistent dtype usage (`np.float32`). However, there are several performance and correctness issues related to:

1. **Redundant astype calls** - Multiple unnecessary dtype conversions that create array copies
2. **Inefficient array operations** - Mixed use of copy=False that doesn't prevent copies in some cases
3. **Potential view/copy bugs** - Inconsistent handling of array references
4. **Memory allocation patterns** - Repeated allocations in hot loops during observation construction
5. **Action validation issues** - Shape validation bug and inefficient clipping

---

## Bugs Found

### Bug 1: EWAR_DIM Attribute Reference Missing
**File:** `/home/john/echelon/echelon/env/env.py:1005`
**Severity:** High - Code will crash on action dimension mismatch

```python
raise ValueError(
    f"action[{aid!r}] has size {a.size}, expected {self.ACTION_DIM} "
    f"(base={self.BASE_ACTION_DIM}, target={self.TARGET_DIM}, ewar={self.EWAR_DIM}, "
    #                                                                    ^^^^^^^^^^^^
    f"obs_ctrl={self.OBS_CTRL_DIM}, comm_dim={self.comm_dim})"
)
```

**Issue:** References `self.EWAR_DIM` which is not defined as a class attribute. The class defines:
- `TARGET_DIM = CONTACT_SLOTS` (line 146)
- `OBS_CTRL_DIM` (line 143)
- But no `EWAR_DIM`

**Impact:** Error message will fail with `AttributeError: 'EchelonEnv' object has no attribute 'EWAR_DIM'` when an action dimension mismatch occurs.

**Fix:** Either define `EWAR_DIM = 0` as a class constant or remove it from the error message.

---

### Bug 2: Redundant dtype Conversion After nan_to_num
**File:** `/home/john/echelon/echelon/env/env.py:948`
**Severity:** Low - Performance waste, not correctness

```python
vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
```

**Issue:** `np.nan_to_num()` already returns the same dtype as input. Since `vec` is already `np.float32` (from concatenate on line 946), the `.astype(np.float32, copy=False)` is redundant.

**Impact:** Potential unnecessary array copy (though `copy=False` may prevent it, the call itself is wasteful).

**Same pattern at:** `/home/john/echelon/echelon/env/env.py:1012` in action processing.

---

### Bug 3: Double astype with copy=False May Still Copy
**File:** `/home/john/echelon/echelon/env/env.py:1012-1013`
**Severity:** Medium - Performance issue

```python
a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
a = a.astype(np.float32, copy=False)
```

**Issue:** Two consecutive `astype` calls on the same dtype. The second is completely redundant.

**Impact:** Wastes CPU cycles on every agent action processing (20 agents * N steps).

---

## Issues

### Issue 1: Inefficient astype Pattern Throughout Observation Construction
**Files:** Multiple locations in `env.py`
**Severity:** Medium - Performance

**Pattern found at:**
- Line 467: `rel = (other.pos - viewer.pos).astype(np.float32, copy=False)`
- Line 470: `rel_vel = (other.vel - viewer.vel).astype(np.float32, copy=False)`
- Line 870: `self_vel = (viewer.vel / 10.0).astype(np.float32, copy=False)`
- Line 946: `vec = np.concatenate(parts).astype(np.float32, copy=False)`

**Analysis:**
1. **Line 467, 470:** If `other.pos` and `viewer.pos` are already `np.float32`, the subtraction produces `float32`, making `astype` redundant.
2. **Line 870:** Division by float already produces the correct dtype if input is `float32`.
3. **Line 946:** If all `parts` are `float32`, `concatenate` returns `float32`.

**Root Cause:** The code doesn't trust that operations preserve dtype. This is defensive but wasteful.

**Impact:**
- `copy=False` prevents copying if dtype matches, but the function call overhead remains
- In hot paths (_obs is called every step for 20 agents), this adds up
- Profiling would likely show these as hotspots

**Better Pattern:**
```python
# Instead of:
rel = (other.pos - viewer.pos).astype(np.float32, copy=False)

# Use (if confident inputs are float32):
rel = other.pos - viewer.pos  # Already float32 from MechState

# Or add a single assertion at initialization:
assert self.sim.mechs[aid].pos.dtype == np.float32
```

---

### Issue 2: Array Allocation in _local_map Loop
**File:** `/home/john/echelon/echelon/env/env.py:558-601`
**Severity:** Medium - Performance

**Current Code:**
```python
def _local_map(self, viewer: MechState) -> np.ndarray:
    # ...
    solid_slice = world.voxels[:clearance_z, :, :]  # Slice (view, good)
    occupancy_2d = np.any(                           # NEW allocation
        (solid_slice == VoxelWorld.SOLID) | (solid_slice == VoxelWorld.SOLID_DEBRIS),
        axis=0,
    )
    # ... vectorized sampling is good ...
    out = np.ones((size, size), dtype=np.float32)   # NEW allocation per agent
```

**Issue:** `_local_map` is called once per alive agent in `_obs()` (potentially 20 times per step). Each call:
1. Creates `occupancy_2d` (size_y × size_x boolean array)
2. Creates `out` array (11×11 float32)

**Impact:**
- For 20 agents: 20 × (size_y × size_x + 121 floats) allocations per step
- For 100x100 world: ~200KB of temporary allocations per step
- Over 1000 steps: ~200MB of churn

**Optimization Opportunity:**
Cache `occupancy_2d` at the `_obs()` level and pass it to `_local_map`:
```python
def _obs(self) -> dict[str, np.ndarray]:
    # Compute once for all agents
    solid_slice = world.voxels[:clearance_z, :, :]
    occupancy_2d = np.any(...)

    for aid in self.agents:
        # Pass cached map
        parts.append(self._local_map(viewer, occupancy_2d))
```

**Comment in code (line 568-571) already mentions this optimization**, but it's not implemented!

---

### Issue 3: Telemetry Downsampling Not Cached
**File:** `/home/john/echelon/echelon/env/env.py:647-671`
**Severity:** Medium - Performance

**Current Code:**
```python
def _obs(self) -> dict[str, np.ndarray]:
    # ...
    # Collapse Z once
    world_2d = np.any(
        (world.voxels == VoxelWorld.SOLID) | (world.voxels == VoxelWorld.SOLID_DEBRIS),
        axis=0,
    )

    sy, sx = world_2d.shape
    y_bins = np.linspace(0, sy, TELEMETRY_SIZE + 1).astype(int)
    x_bins = np.linspace(0, sx, TELEMETRY_SIZE + 1).astype(int)

    telemetry = np.zeros((TELEMETRY_SIZE, TELEMETRY_SIZE), dtype=np.float32)
    for iy in range(TELEMETRY_SIZE):
        for ix in range(TELEMETRY_SIZE):
            region = world_2d[y_bins[iy]:y_bins[iy+1], x_bins[ix]:x_bins[ix+1]]
            if region.size > 0 and np.any(region):
                telemetry[iy, ix] = 1.0
    telemetry_flat = telemetry.reshape(-1)
```

**Issue:** This downsampling is **identical for all agents** but computed on every `_obs()` call. For static terrain (which doesn't change during episodes in this codebase), this is pure waste.

**Impact:**
- 16×16 loop with `np.any()` calls per iteration
- Called every step
- Over 1000 steps: thousands of redundant computations

**Optimization:** Cache `telemetry_flat` at reset and reuse:
```python
def reset(self, ...):
    # ... after world generation ...
    self._telemetry_cache = self._compute_telemetry()

def _compute_telemetry(self) -> np.ndarray:
    # Move current telemetry code here
    # ...
    return telemetry_flat

def _obs(self):
    # ...
    telemetry_flat = self._telemetry_cache
```

---

### Issue 4: Acoustic Intensity Calculation Can Be Vectorized
**File:** `/home/john/echelon/echelon/env/env.py:692-722`
**Severity:** Low - Performance

**Current Code:**
```python
acoustic_intensities = np.zeros(4, dtype=np.float32)

for other_id in self.possible_agents:
    if other_id == aid: continue
    other = sim.mechs[other_id]
    if not other.alive or other.noise_level <= 0: continue

    delta = other.pos - viewer.pos
    dist_sq = float(np.dot(delta, delta))
    intensity = other.noise_level / (1.0 + dist_sq)

    if not _cached_los(aid, other_id, viewer.pos, other.pos):
        intensity *= 0.20

    # ... quadrant calculation ...
    acoustic_intensities[q] += float(intensity)
```

**Issue:** Loop over 20 agents per observer, computing one at a time.

**Optimization Opportunity:**
- Pre-compute positions/noise_levels for all alive mechs as arrays
- Vectorize delta/distance calculations
- Apply LOS mask
- Use `np.add.at()` for quadrant accumulation

**Impact:** Moderate - Would eliminate ~20 iterations × 20 observers = 400 scalar operations per step.

---

### Issue 5: Contact Sorting Uses Tuple Keys
**File:** `/home/john/echelon/echelon/env/env.py:763-776`
**Severity:** Low - Code cleanliness

**Current Code:**
```python
if sort_mode == 0:
    key = (dist,)
elif sort_mode == 1:
    cls_rank = {"scout": 1, "light": 1, "medium": 2, "heavy": 3}.get(other.spec.name, 0)
    key = (-cls_rank, dist)
elif sort_mode == 2:
    hp_norm = float(np.clip(other.hp / max(1.0, other.spec.hp), 0.0, 1.0))
    key = (hp_norm, dist)
else:
    key = (dist,)
contacts[relation].append((key, dist, other_id, painted_by_pack))
```

**Issue:** Stores `(key, dist, other_id, painted_by_pack)` tuple, but later unpacking (line 792, 802) uses:
```python
_, _, oid, painted = contacts[rel][i]
```

This discards the `dist` component that was explicitly stored. The `dist` is only needed for sorting (already in `key`).

**Better Pattern:**
```python
contacts[relation].append((key, other_id, painted_by_pack))
# Later:
_, oid, painted = contacts[rel][i]
```

**Impact:** Minor - Wastes a bit of memory and makes code slightly confusing.

---

### Issue 6: Action Clipping Happens In-Place After Validation
**File:** `/home/john/echelon/echelon/env/env.py:1014`
**Severity:** Low - Design oddity

**Current Code:**
```python
a = a.astype(np.float32, copy=False)
np.clip(a, -1.0, 1.0, out=a)  # In-place modification
act[aid] = a
```

**Issue:** The `out=a` parameter modifies the array in-place. While this is efficient, it modifies the caller's array reference if they passed in a non-copy.

**Impact:**
- If user code passes same action array to multiple agents, clipping one affects others
- Best practice: Make defensive copy early if modifying

**Better Pattern:**
```python
a = np.clip(a, -1.0, 1.0)  # Returns new array, cleaner semantics
```
or
```python
a = np.asarray(a, dtype=np.float32).copy()  # Explicit defensive copy
np.clip(a, -1.0, 1.0, out=a)  # Then modify
```

---

### Issue 7: Zone Relative Position Uses np.array Instead of np.asarray
**File:** `/home/john/echelon/echelon/env/env.py:884`
**Severity:** Low - Performance

```python
zone_rel = np.array([zone_cx, zone_cy, 0.0], dtype=np.float32) - viewer.pos
```

**Issue:** `np.array()` always creates a new array. Since we're creating a new array anyway (from a list), this is fine, but inconsistent with the `np.asarray()` pattern used elsewhere.

**Impact:** Negligible - This is called once per agent per step and the array is small.

**Recommendation:** Use consistent pattern (`np.asarray` or `np.array`) throughout codebase for clarity.

---

## Improvement Opportunities

### Opportunity 1: Pre-allocate Observation Buffers
**File:** `/home/john/echelon/echelon/env/env.py:603-951`
**Severity:** Medium - Performance

**Current Pattern:**
```python
def _obs(self) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for aid in self.agents:
        # ... build parts list ...
        parts: list[np.ndarray] = []
        parts.append(contact_feats.reshape(-1))
        parts.append(comm.reshape(-1))
        # ...
        vec = np.concatenate(parts).astype(np.float32, copy=False)
        obs[aid] = vec
```

**Issue:** Every step creates:
- New `parts` list (20× per step)
- Multiple temporary arrays
- Calls `np.concatenate()` which allocates a new array

**Optimization:**
```python
class EchelonEnv:
    def __init__(self, config: EnvConfig):
        # Pre-allocate observation buffers
        obs_dim = self._obs_dim()
        self._obs_buffers = {
            aid: np.zeros(obs_dim, dtype=np.float32)
            for aid in self.possible_agents
        }

    def _obs(self) -> dict[str, np.ndarray]:
        for aid in self.agents:
            buf = self._obs_buffers[aid]
            offset = 0

            # Write directly into buffer
            contact_size = self.CONTACT_SLOTS * self.CONTACT_DIM
            buf[offset:offset+contact_size] = contact_feats.reshape(-1)
            offset += contact_size

            # ... continue for each part ...

            obs[aid] = buf  # Return view of pre-allocated buffer
```

**Impact:** Eliminates ~20 allocations + concatenations per step. Significant for long training runs.

---

### Opportunity 2: Use np.einsum for Rotation in _local_map
**File:** `/home/john/echelon/echelon/env/env.py:586-592`
**Severity:** Low - Readability/Performance

**Current Code:**
```python
c, s = math.cos(yaw), math.sin(yaw)
fwd_grid, right_grid = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing='ij')

dx = c * fwd_grid - s * right_grid
dy = s * fwd_grid + c * right_grid
```

**Optimization:** Could use rotation matrix with `np.einsum` for clarity:
```python
rot = np.array([[c, -s], [s, c]], dtype=np.float32)
grid = np.stack([fwd_grid, right_grid], axis=-1)
rotated = np.einsum('ij,hwj->hwi', rot, grid)
dx, dy = rotated[..., 0], rotated[..., 1]
```

**Impact:** Negligible performance change, but more mathematically explicit.

---

### Opportunity 3: Batch MechState Attribute Access
**File:** Multiple locations in `env.py`
**Severity:** Low - Architecture

**Current Pattern:**
```python
for m in sim.mechs.values():
    if not m.alive: continue
    # Access m.pos, m.vel, m.team, etc. one at a time
```

**Observation:** Python attribute access is slow. For vectorized operations, it's faster to batch-extract into arrays.

**Better Pattern:**
```python
# In Sim class, maintain parallel arrays:
self.positions = np.zeros((num_mechs, 3), dtype=np.float32)
self.velocities = np.zeros((num_mechs, 3), dtype=np.float32)
self.alive_mask = np.zeros(num_mechs, dtype=bool)

# Update in step(), then env can access directly:
alive_positions = sim.positions[sim.alive_mask]
```

**Impact:** Would enable fully vectorized observation construction, but requires significant refactoring of `Sim` and `MechState`.

---

### Opportunity 4: Type Hints for NumPy Arrays
**File:** All reviewed files
**Severity:** Low - Maintainability

**Current Pattern:**
```python
def _contact_features(
    self,
    viewer: MechState,
    other: MechState,
    *,
    relation: str,
    painted_by_pack: bool,
) -> np.ndarray:
```

**Issue:** Return type is `np.ndarray` without shape/dtype hints.

**Better Pattern:** Use numpy typing (requires numpy 1.20+):
```python
from numpy.typing import NDArray

def _contact_features(
    self,
    viewer: MechState,
    other: MechState,
    *,
    relation: str,
    painted_by_pack: bool,
) -> NDArray[np.float32]:  # Or better: npt.NDArray[np.float32]
    """Returns contact features [CONTACT_DIM] array."""
```

**Impact:** Better IDE support, easier to catch dtype bugs at static analysis time.

---

### Opportunity 5: Action Validation Could Use NumPy Vectorization
**File:** `/home/john/echelon/echelon/env/env.py:994-1022`
**Severity:** Low - Performance

**Current Code:**
```python
act: dict[str, np.ndarray] = {}
for aid in self.agents:
    a = actions.get(aid)
    if a is None:
        a = np.zeros(self.ACTION_DIM, dtype=np.float32)
    else:
        a = np.asarray(a, dtype=np.float32)
        # ... validation ...
    bad = bool(not np.all(np.isfinite(a)))
    if bad:
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    np.clip(a, -1.0, 1.0, out=a)
    act[aid] = a
```

**Optimization:** Batch process all actions at once:
```python
# Stack into (num_agents, action_dim) array
action_batch = np.zeros((len(self.agents), self.ACTION_DIM), dtype=np.float32)
for i, aid in enumerate(self.agents):
    if aid in actions:
        action_batch[i] = np.asarray(actions[aid], dtype=np.float32)[:self.ACTION_DIM]

# Vectorized validation
bad_mask = ~np.isfinite(action_batch)
action_batch[bad_mask] = 0.0
np.clip(action_batch, -1.0, 1.0, out=action_batch)

# Unpack back to dict
act = {aid: action_batch[i] for i, aid in enumerate(self.agents)}
```

**Impact:** Eliminates 20 individual validation calls, uses vectorized numpy ops.

---

## Recommendations Summary

### Critical (Fix Immediately)
1. **Fix Bug 1:** Add `EWAR_DIM` attribute or remove from error message
2. **Remove redundant astype calls** (Bugs 2-3)

### High Priority (Performance Wins)
3. **Cache telemetry map** (Issue 3) - Easy win, significant speedup
4. **Cache occupancy_2d and pass to _local_map** (Issue 2) - Comment already suggests this
5. **Pre-allocate observation buffers** (Opportunity 1) - Biggest performance gain

### Medium Priority (Code Quality)
6. **Remove defensive astype(copy=False) where not needed** (Issue 1)
7. **Simplify contact tuple structure** (Issue 5)
8. **Make action clipping defensive copy explicit** (Issue 6)

### Low Priority (Nice to Have)
9. **Vectorize acoustic intensity calculation** (Issue 4)
10. **Add numpy type hints** (Opportunity 4)
11. **Batch action validation** (Opportunity 5)

### Future Work (Requires Architecture Changes)
12. **Refactor Sim to use parallel arrays instead of MechState objects** (Opportunity 3)
    - Would enable fully vectorized env operations
    - Large refactor, consider for v2

---

## Testing Recommendations

1. **Add unit tests for observation dimensions:**
   ```python
   def test_obs_dimension():
       env = EchelonEnv(config)
       obs, _ = env.reset()
       expected_dim = env._obs_dim()
       for aid, o in obs.items():
           assert o.shape == (expected_dim,)
           assert o.dtype == np.float32
   ```

2. **Add dtype consistency tests:**
   ```python
   def test_observation_dtype_consistency():
       env = EchelonEnv(config)
       obs, _ = env.reset()
       for aid, o in obs.items():
           assert o.dtype == np.float32
           assert np.all(np.isfinite(o))
   ```

3. **Profile observation construction:**
   ```python
   import cProfile
   env = EchelonEnv(config)
   env.reset()
   cProfile.run('env._obs()', sort='cumtime')
   # Identify actual bottlenecks
   ```

4. **Test action handling edge cases:**
   ```python
   def test_action_validation():
       env = EchelonEnv(config)
       obs, _ = env.reset()

       # Test NaN/Inf handling
       bad_actions = {aid: np.full(env.ACTION_DIM, np.nan) for aid in env.agents}
       obs, rewards, _, _, infos = env.step(bad_actions)
       # Should not crash, should track bad_actions stat
   ```

---

## Conclusion

The environment layer is well-structured with consistent dtype usage and good defensive programming against NaN/Inf values. The main issues are:

1. **Performance**: Unnecessary allocations and dtype conversions in hot paths
2. **Missed optimizations**: Code comments identify caching opportunities that aren't implemented
3. **Minor bugs**: Missing attribute in error message, redundant operations

**Estimated Performance Impact of Fixes:**
- Critical bugs: Fixes enable proper error reporting
- Caching optimizations (Issues 2-3): ~20-30% speedup in `_obs()`
- Pre-allocated buffers (Opp 1): ~15-25% speedup in `_obs()`
- **Total potential speedup: 35-50% faster environment step time**

For a training run of 1M steps, this could save hours of wall-clock time.
