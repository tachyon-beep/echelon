# Track Store System Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Per-pack track persistence and datalink sharing so enemies don't blink out when LOS breaks and scouts get credit for intel contributions.

**Architecture:** Each pack maintains a shared `PackTrackStore` that persists enemy detections with decay. Paint action auto-reports to packmates. Commanders get fused view of all pack tracks.

**Tech Stack:** Tensorized storage (`PackTrackTensors`) for vectorized environments, integrated with existing `CombatSuiteSpec` and `ObservationBuilder`.

**Status:** REVISED after DRL, PyTorch, and Simulation specialist reviews (2025-12-27).

---

## Review-Driven Changes

| Change | Rationale | Source |
|--------|-----------|--------|
| Remove behavior inference | Let LSTM learn patterns from raw telemetry | Simulation review |
| Per-episode detection dedup | Prevent scout from cycling targets for unlimited +0.3 | DRL review |
| Causal tightening for track usage | Only award if shooter lacked LOS AND track < 5s old | DRL review |
| Protected eviction for painted | Never evict active paint locks | Simulation review |
| Tensorized storage | Python dicts bottleneck vectorized env | PyTorch review |
| Extend sensor.py | Don't duplicate existing TrackStore | Simulation review |

---

## 1. Data Model

### 1.1 TrackReason Enum

```python
class TrackReason(Enum):
    """Why this track exists in the store."""
    VISUAL_CONTACT = "visual"      # Direct LOS detection
    SCOUT_REPORT = "scout"         # Scout painted/reported
    PACKLEADER_ORDER = "order"     # Pack leader designated target
    TARGET_PAINTED = "painted"     # Currently painted (live lock)
```

### 1.2 SensorTrack Dataclass

**Design choice:** No behavior inference. Raw sensor telemetry only - let the LSTM learn behavioral patterns from sequences.

```python
@dataclass
class SensorTrack:
    """A single track in the store.

    Contains only what automated sensors can detect:
    - IR signature (heat)
    - Visual/structural assessment (damage, posture)
    - RF emissions (firing, ECM, painting)
    - Kinematics (position, velocity, facing)
    """
    target_id: str              # Mech agent ID
    target_class: str           # "heavy", "medium", "light", "scout"

    # Position & motion (kinematics)
    position: tuple[float, float, float]  # (x, y, z)
    velocity: tuple[float, float, float]  # (vx, vy, vz)
    facing_yaw: float           # Radians

    # IR signature
    heat_level: float           # 0.0-1.0 normalized

    # Visual/structural assessment
    damage_level: float         # 0.0-1.0 (1.0 = full HP)
    is_fallen: bool
    is_legged: bool

    # RF emissions (detectable without LOS)
    is_firing: bool             # Weapon discharge detected
    ecm_active: bool            # ECM jamming active
    is_painting: bool           # Target designation radar active

    # Track metadata
    track_age: float            # Seconds since last update
    track_reason: TrackReason
    sender_id: str              # Who reported this
    sender_role: str            # "scout", "pack_leader", etc.
    paint_active: bool          # Currently painted BY US (missile lock valid)

    # Credit tracking (per-episode dedup)
    first_detector_id: str      # Who detected first (for rewards)
    detection_time: float       # Sim time of first detection
```

### 1.3 Track Freshness States

| State | Age Range | Paint Lock | Description |
|-------|-----------|------------|-------------|
| PAINTED | Active | Valid | Currently painted, live missile lock |
| FRESH | 0-5s | Invalid | Recent data, high confidence |
| STALE | 5-10s | Invalid | Degraded confidence |
| LOST | 10-15s | Invalid | About to be evicted |

After 15s without update, track is removed.

**Note:** These thresholds may need tuning per mech class (fast mechs move 100+ voxels in 15s). Consider adaptive decay or position uncertainty in future iteration.

---

## 2. PackTrackStore Class

### 2.1 Interface

```python
class PackTrackStore:
    """Per-pack shared track repository with decay and protected eviction."""

    def __init__(self, pack_id: str, max_tracks: int = 20):
        self.pack_id = pack_id
        self.max_tracks = max_tracks
        self.tracks: dict[str, SensorTrack] = {}  # target_id -> track
        self._access_order: list[str] = []  # LRU tracking

        # Per-episode deduplication for first-detection rewards
        # Prevents scouts from cycling through targets for unlimited +0.3
        self._ever_detected: set[str] = set()

    def update_track(
        self,
        target_id: str,
        target_state: MechState,
        reporter_id: str,
        reporter_role: str,
        reason: TrackReason,
        paint_active: bool,
        sim_time: float,
    ) -> tuple[bool, str | None]:
        """
        Update or create track.

        Returns:
            (is_novel_detection, first_detector_id)
            is_novel_detection: True if FIRST TIME EVER this episode seeing this target
            first_detector_id: Who gets credit for first detection (only if novel)
        """
        is_novel = target_id not in self._ever_detected
        if is_novel:
            self._ever_detected.add(target_id)
        # ... rest of update logic

    def tick(self, dt: float, sim_time: float) -> list[str]:
        """
        Age all tracks and remove expired ones.

        Returns:
            List of target_ids that were evicted
        """
        ...

    def get_tracks_for_role(self, role: str, max_slots: int) -> list[SensorTrack]:
        """
        Get tracks filtered/sorted for a specific role.

        Sorting priority:
        1. Paint-active tracks (missile locks)
        2. High heat targets (about to vent, vulnerable)
        3. Low damage targets (priority kills)
        4. Heavies (high-value)
        5. By freshness (newer first)
        """
        ...

    def clear_paint(self, target_id: str) -> None:
        """Mark paint lock as inactive (painter broke LOS or died)."""
        ...

    def reset(self) -> None:
        """Clear all tracks and detection history (call on episode reset)."""
        self.tracks.clear()
        self._access_order.clear()
        self._ever_detected.clear()
```

### 2.2 Protected Eviction

When `len(tracks) >= max_tracks` and a new track arrives:

```python
def _find_eviction_candidate(self) -> str | None:
    """Find track to evict. Protected classes cannot be evicted."""
    candidates = []
    for target_id, track in self.tracks.items():
        # NEVER evict active paint locks - scout worked hard for these
        if track.paint_active:
            continue
        candidates.append((target_id, self._eviction_priority(track)))

    if not candidates:
        return None  # All tracks protected, cannot evict

    # Evict lowest priority (highest age, lowest value)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]

def _eviction_priority(self, track: SensorTrack) -> float:
    """Lower = more likely to evict."""
    priority = -track.track_age  # Older = lower priority (more negative)

    # Boost for high-value targets
    class_boost = {"heavy": 10.0, "medium": 5.0, "light": 2.0, "scout": 1.0}
    priority += class_boost.get(track.target_class, 0)

    return priority
```

**Protected classes:**
- `paint_active=True` - Active paint locks are never evicted (scout's effort preserved)

---

## 3. Paint Action Integration

### 3.1 Current Paint Flow

```
Scout fires PRIMARY (paint) → Target gets paint lock → Packmates can use lock for missiles
```

### 3.2 New Flow with Track Store

```
Scout fires PRIMARY (paint) → Target gets paint lock
                           → Track added/updated in PackTrackStore
                           → reason=TARGET_PAINTED, paint_active=True
                           → Packmates see in alert slots (if sensor-limited)
```

### 3.3 Implementation Hook

In `EchelonEnv._process_combat_events()`:

```python
def _handle_paint_event(self, ev: dict) -> None:
    """Handle paint lock event - update track store."""
    painter_id = ev["shooter"]
    target_id = ev["target"]

    painter = self.sim.mechs[painter_id]
    target = self.sim.mechs[target_id]
    pack_id = painter.pack_id

    store = self._pack_track_stores[pack_id]
    is_new, first_detector = store.update_track(
        target_id=target_id,
        target_state=target,
        reporter_id=painter_id,
        reporter_role=painter.spec.name,
        reason=TrackReason.TARGET_PAINTED,
        paint_active=True,
        sim_time=self.sim.time,
    )

    if is_new:
        # Credit scout for first detection
        self._first_detection_credits[painter_id] = \
            self._first_detection_credits.get(painter_id, 0) + 1
```

---

## 4. Alert Slots in Observation Space

### 4.1 Slot Allocation by Role

| Suite | Intel Slots | Alert Slots | Notes |
|-------|-------------|-------------|-------|
| BASIC | 5 | 3 | Heavy/Medium - need team intel |
| SCOUT | 20 | 0 | Can see everything directly |
| PACK_COMMAND | 10 | 3 | Leader gets fused pack view |
| SQUAD_COMMAND | 12 | 3 | Commander gets all squad tracks |

### 4.2 Alert Slot Dimensions

Each alert slot encodes one track with 18 dimensions (raw sensor telemetry, no behavior inference):

```python
ALERT_SLOT_DIM = 18

def encode_alert_slot(track: SensorTrack | None, viewer_pos: np.ndarray) -> np.ndarray:
    """Encode track into fixed-size observation slot.

    Raw sensor telemetry only - no behavior inference.
    Let the LSTM learn patterns from sequences.
    """
    slot = np.zeros(ALERT_SLOT_DIM, dtype=np.float32)
    if track is None:
        return slot  # Empty slot

    slot[0] = 1.0  # Slot occupied flag

    # Position (relative to viewer, normalized)
    slot[1:4] = normalize_position(track.position - viewer_pos)

    # Velocity (normalized)
    slot[4:7] = normalize_velocity(track.velocity)

    # Facing (sin/cos encoding - avoids discontinuity at 0/2pi)
    slot[7] = np.sin(track.facing_yaw)
    slot[8] = np.cos(track.facing_yaw)

    # IR signature
    slot[9] = track.heat_level

    # Visual/structural assessment
    slot[10] = track.damage_level
    slot[11] = float(track.is_fallen)
    slot[12] = float(track.is_legged)

    # RF emissions
    slot[13] = float(track.is_firing)
    slot[14] = float(track.ecm_active)
    slot[15] = float(track.is_painting)

    # Track metadata
    slot[16] = min(track.track_age / 15.0, 1.0)  # Normalized age
    slot[17] = float(track.paint_active)  # OUR paint lock on them

    return slot
```

**Note:** No behavior inference - agent learns "high heat + not firing = about to vent" from experience.

### 4.3 Integration with ObservationBuilder

The alert slots are appended after existing contact slots:

```python
# In ObservationBuilder.build()
if self.config.use_track_store:
    alert_tracks = pack_store.get_tracks_for_role(
        role=mech.spec.name,
        max_slots=self.alert_slots,
    )
    for i, track in enumerate(alert_tracks[:self.alert_slots]):
        obs[alert_start + i*ALERT_SLOT_DIM : alert_start + (i+1)*ALERT_SLOT_DIM] = \
            encode_alert_slot(track)
```

---

## 5. Scout Credit Rewards

### 5.1 Reward Components

| Event | Reward | Condition |
|-------|--------|-----------|
| First Detection | +0.3 | Scout is first to detect target THIS EPISODE (per-episode dedup) |
| Track Usage | +0.2 | Packmate deals damage using track, AND shooter lacked LOS, AND track < 5s old |

### 5.2 Causal Tightening for Track Usage

The track usage reward has strict conditions to ensure causal connection:

```python
def should_award_track_usage(
    shooter: MechState,
    target: MechState,
    track: SensorTrack,
    sim: Sim,
) -> bool:
    """Determine if track usage credit should be awarded.

    Conditions (ALL must be true):
    1. Track must be FRESH (< 5s old) - stale tracks don't provide actionable intel
    2. Shooter must NOT have direct LOS - track was actually necessary
    3. Damage must be dealt - not just a shot fired

    This prevents scouts from painting everything and getting credit
    for shots that would have happened anyway.
    """
    # Track must be fresh
    if track.track_age > 5.0:
        return False

    # Shooter must not have direct LOS (track was necessary)
    if sim.has_los(shooter.pos, target.pos):
        return False

    return True
```

### 5.3 Implementation

In `rewards.py`:

```python
@dataclass
class RewardWeights:
    # ... existing weights ...

    # Scout intel rewards
    first_detection: float = 0.3  # First to detect enemy (per-episode dedup)
    track_usage: float = 0.2      # Packmate uses your track (causally tightened)
```

In `StepContext`:

```python
@dataclass
class StepContext:
    # ... existing fields ...

    # Track store credit events
    step_first_detections: dict[str, int] | None = None  # aid -> count
    step_track_usages: dict[str, int] | None = None      # aid -> count
```

In `RewardComputer._compute_agent_reward()`:

```python
# Scout intel rewards
if ctx.step_first_detections is not None:
    detections = ctx.step_first_detections.get(aid, 0)
    if detections > 0:
        comp.first_detection = w.first_detection * detections

if ctx.step_track_usages is not None:
    usages = ctx.step_track_usages.get(aid, 0)
    if usages > 0:
        comp.track_usage = w.track_usage * usages
```

---

## 6. Commander Stitched View

### 6.1 Pack Leader View

Pack leader sees **all tracks from their pack's store** (up to intel_slots=10):
- Their own detections
- Subordinate reports
- Paint locks from pack scouts

### 6.2 Squad Commander View

Squad commander (when implemented) sees **merged tracks from all packs**:
- Deduplication by target_id
- Freshest track wins on collision
- Up to intel_slots=12

### 6.3 Implementation

```python
def get_commander_tracks(
    pack_stores: dict[str, PackTrackStore],
    commander_pack_id: str,
    is_squad_commander: bool,
    max_slots: int,
) -> list[SensorTrack]:
    """Get stitched view for commander."""
    if is_squad_commander:
        # Merge all packs - freshest observation of each target wins
        all_tracks: dict[str, SensorTrack] = {}
        for store in pack_stores.values():
            for target_id, track in store.tracks.items():
                if target_id not in all_tracks or track.track_age < all_tracks[target_id].track_age:
                    all_tracks[target_id] = track
        tracks = list(all_tracks.values())
    else:
        # Pack leader just sees own pack
        tracks = list(pack_stores[commander_pack_id].tracks.values())

    # Sort by threat priority (raw telemetry signals, no behavior inference)
    tracks.sort(key=lambda t: (
        -float(t.paint_active),           # Paint locks first (missile ready)
        -float(t.is_firing),              # Currently firing = active threat
        -t.heat_level,                    # High heat = about to vent (vulnerable)
        -(1.0 - t.damage_level),          # Low HP = priority kill
        -{"heavy": 4, "medium": 3, "light": 2, "scout": 1}.get(t.target_class, 0),
        t.track_age,                      # Fresher tracks last tiebreaker
    ))

    return tracks[:max_slots]
```

---

## 7. Implementation Tasks

**Note:** Simulation review found existing `sensor.py` has `TrackStore`. Consider extending it rather than creating parallel system. Tasks below assume new implementation but can be adapted.

### Task 1: Create Track Data Model

**Files:**
- Create: `echelon/env/tracks.py`
- Test: `tests/unit/test_tracks.py`

**Steps:**
1. Create `TrackReason` enum (VISUAL_CONTACT, SCOUT_REPORT, PACKLEADER_ORDER, TARGET_PAINTED)
2. Create `SensorTrack` dataclass with raw telemetry fields (NO behavior inference)
3. Create `TrackFreshness` enum (PAINTED, FRESH, STALE, LOST)
4. Add `freshness` property to SensorTrack based on track_age
5. Write unit tests for freshness transitions

### Task 2: Implement PackTrackStore

**Files:**
- Modify: `echelon/env/tracks.py`
- Test: `tests/unit/test_tracks.py`

**Steps:**
1. Implement `PackTrackStore.__init__` with `_ever_detected: set[str]` for per-episode dedup
2. Implement `update_track` with novel detection tracking (returns is_novel only if first time THIS EPISODE)
3. Implement `tick` with age decay and protected eviction
4. Implement `_find_eviction_candidate` with paint lock protection
5. Implement `get_tracks_for_role` with priority sorting (paint > firing > heat > damage > class > age)
6. Implement `clear_paint` and `reset`
7. Write unit tests for protected eviction (painted tracks never evicted)
8. Write unit tests for per-episode dedup (rediscovery doesn't give credit)

### Task 3: Wire Track Store to Environment

**Files:**
- Modify: `echelon/env/env.py`
- Test: `tests/unit/test_env_tracks.py`

**Steps:**
1. Add `_pack_track_stores: dict[str, PackTrackStore]` to EchelonEnv
2. Initialize stores in `reset()`
3. Call `store.tick(dt)` in `step()`
4. Add `_handle_paint_event` to update track store on paint
5. Track first-detection credits per step
6. Write integration test verifying paint populates store

### Task 4: Add Alert Slots to Observations

**Files:**
- Modify: `echelon/env/observations.py`
- Modify: `echelon/config.py`
- Test: `tests/unit/test_observations_tracks.py`

**Steps:**
1. Add `ALERT_SLOT_DIM = 18` constant (raw telemetry, no behavior)
2. Add `alert_slots` to `EnvConfig` (default 3)
3. Implement `encode_alert_slot` function with raw sensor fields
4. Modify `ObservationBuilder.obs_dim()` to include alert slots (54 dims for 3 slots)
5. Modify `ObservationBuilder.build()` to populate alert slots
6. Write test verifying observation dimension increases correctly
7. Write test verifying alert slot content matches track telemetry

### Task 5: Add Scout Credit Rewards (with Causal Tightening)

**Files:**
- Modify: `echelon/env/rewards.py`
- Modify: `echelon/env/env.py`
- Test: `tests/unit/test_rewards_tracks.py`

**Steps:**
1. Add `first_detection: float = 0.3` and `track_usage: float = 0.2` to `RewardWeights`
2. Add `step_first_detections` and `step_track_usages` to `StepContext`
3. Add `first_detection` and `track_usage` to `RewardComponents`
4. Implement `should_award_track_usage()` with causal tightening (LOS check + age check)
5. Implement credit computation in `_compute_agent_reward`
6. Wire credit tracking in `EchelonEnv.step()`
7. Write test: scout gets credit on FIRST detection only (not rediscovery)
8. Write test: track usage credit ONLY if shooter lacked LOS AND track < 5s old
9. Write test: no track usage credit if shooter had direct LOS

### Task 6: Implement Commander View

**Files:**
- Modify: `echelon/env/tracks.py`
- Modify: `echelon/env/observations.py`
- Test: `tests/unit/test_commander_view.py`

**Steps:**
1. Add `get_commander_tracks` function
2. Modify observation building for PACK_COMMAND suite
3. Use existing `intel_slots` from suite spec
4. Write test verifying pack leader sees all pack tracks
5. Write test verifying threat priority sorting

### Task 7: W&B Metrics Integration

**Files:**
- Modify: `scripts/train_ppo.py`
- Modify: `echelon/env/env.py`

**Steps:**
1. Add `first_detections_blue`, `track_usages_blue` to episode_stats
2. Add metrics to W&B perception stats
3. Add `track_store_size` metric (average tracks per pack)
4. Test metrics appear in W&B dashboard

---

## 8. Testing Strategy

### 8.1 Unit Tests

- Track freshness transitions (age -> FRESH -> STALE -> LOST)
- Protected eviction (painted tracks never evicted)
- Per-episode deduplication (rediscovery returns is_novel=False)
- Alert slot encoding (18 dims, raw telemetry, no behavior)
- Commander track merging (freshest wins per target)
- Priority sorting (paint > firing > heat > damage > class > age)

### 8.2 Integration Tests

- Paint action populates track store
- Tracks decay over simulation time
- Alert slots appear in observations
- Scout credit: first detection gives +0.3
- Scout credit: rediscovery gives nothing
- Track usage: +0.2 ONLY when shooter lacked LOS AND track < 5s old
- Track usage: no credit when shooter had direct LOS

### 8.3 Manual Verification

- Run training with W&B
- Verify perception stats show track activity
- Monitor `first_detections_blue`, `track_usages_blue` metrics
- Check that scout reward from intel doesn't dominate damage reward

---

## 9. Curriculum Rollout (DRL Review Recommendation)

Adding 54 dims mid-training will destabilize learning. Recommended phased rollout:

| Phase | Config Flags | What's Active | Duration |
|-------|--------------|---------------|----------|
| 1 | `track_store_enabled=True` | Track store infrastructure only, alert slots always zero | Until stable |
| 2 | `track_store_obs_enabled=True` | Alert slots populated with track data | Until value loss stabilizes |
| 3 | `first_detection_reward_enabled=True` | +0.3 first detection reward | Until scouts show detection-seeking |
| 4 | `track_usage_reward_enabled=True` | +0.2 track usage reward (causally tightened) | Monitor coordination |

```python
@dataclass
class EnvConfig:
    # Track store curriculum flags
    track_store_enabled: bool = False
    track_store_obs_enabled: bool = False
    first_detection_reward_enabled: bool = False
    track_usage_reward_enabled: bool = False
```

---

## 10. Future Extensions (Not MVP)

- **Order slots**: Pack leader issues orders via separate slot system
- **Track uncertainty**: Position covariance that grows with age (adaptive decay)
- **Acoustic detections**: Sound-based tracks without LOS
- **Squad-level fusion**: Cross-pack track sharing for large battles
- **Field-level fusion**: Independent timestamps per field (position vs damage)
- **Tensorized storage**: Convert to `PackTrackTensors` for vectorization (PyTorch review)
- **Replay visualization**: Show tracks in viewer.html
