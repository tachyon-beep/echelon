# Track Store System Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Per-pack track persistence and datalink sharing so enemies don't blink out when LOS breaks and scouts get credit for intel contributions.

**Architecture:** Each pack maintains a shared `PackTrackStore` that persists enemy detections with decay. Paint action auto-reports to packmates. Commanders get fused view of all pack tracks.

**Tech Stack:** Pure Python dataclasses, integrated with existing `CombatSuiteSpec` and `ObservationBuilder`.

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

### 1.2 TargetBehavior Enum

```python
class TargetBehavior(Enum):
    """Inferred behavioral state from telemetry."""
    STATIONARY = "stationary"  # Not moving
    WALKING = "walking"        # Moving, not engaging
    ENGAGING = "engaging"      # Firing weapons
    FLEEING = "fleeing"        # Moving away from pack centroid
```

### 1.3 SensorTrack Dataclass

```python
@dataclass
class SensorTrack:
    """A single track in the store."""
    target_id: str              # Mech agent ID
    target_class: str           # "heavy", "medium", "light", "scout"

    # Position & motion
    position: tuple[float, float, float]  # (x, y, z)
    velocity: tuple[float, float, float]  # (vx, vy, vz)
    facing_yaw: float           # Radians

    # Telemetry
    heat_level: float           # 0.0-1.0 normalized
    damage_level: float         # 0.0-1.0 (1.0 = full HP)
    is_fallen: bool
    is_legged: bool
    is_firing: bool

    # Track metadata
    track_age: float            # Seconds since last update
    track_reason: TrackReason
    sender_id: str              # Who reported this
    sender_role: str            # "scout", "pack_leader", etc.
    paint_active: bool          # Currently painted (missile lock valid)
    behavior: TargetBehavior

    # Credit tracking
    first_detector_id: str      # Who detected first (for rewards)
    detection_time: float       # Sim time of first detection
```

### 1.4 Track Freshness States

| State | Age Range | Paint Lock | Description |
|-------|-----------|------------|-------------|
| PAINTED | Active | Valid | Currently painted, live missile lock |
| FRESH | 0-5s | Invalid | Recent data, high confidence |
| STALE | 5-10s | Invalid | Degraded confidence |
| LOST | 10-15s | Invalid | About to be evicted |

After 15s without update, track is removed.

---

## 2. PackTrackStore Class

### 2.1 Interface

```python
class PackTrackStore:
    """Per-pack shared track repository with decay and LRU eviction."""

    def __init__(self, pack_id: str, max_tracks: int = 20):
        self.pack_id = pack_id
        self.max_tracks = max_tracks
        self.tracks: dict[str, SensorTrack] = {}  # target_id -> track
        self._access_order: list[str] = []  # LRU tracking

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
            (is_new_detection, first_detector_id)
            is_new_detection: True if this is first time seeing this target
            first_detector_id: Who gets credit for first detection
        """
        ...

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
        2. ENGAGING targets (threats)
        3. Heavies (high-value)
        4. By freshness (newer first)
        """
        ...

    def clear_paint(self, target_id: str) -> None:
        """Mark paint lock as inactive (painter broke LOS or died)."""
        ...
```

### 2.2 LRU Eviction

When `len(tracks) >= max_tracks` and a new track arrives:
1. Find oldest track by `track_age` (regardless of freshness state)
2. Evict it
3. Insert new track

**No protection for any freshness state** - new intel always wins.

### 2.3 Behavior Inference

```python
def _infer_behavior(
    target: MechState,
    pack_centroid: tuple[float, float, float],
) -> TargetBehavior:
    """Infer behavioral state from telemetry."""
    if target.is_firing:
        return TargetBehavior.ENGAGING

    speed = np.linalg.norm(target.velocity[:2])  # Horizontal speed
    if speed < 0.5:
        return TargetBehavior.STATIONARY

    # Check if moving away from pack centroid
    to_pack = np.array(pack_centroid) - np.array(target.position)
    movement_dot = np.dot(target.velocity, to_pack)
    if movement_dot < -0.5:  # Moving away
        return TargetBehavior.FLEEING

    return TargetBehavior.WALKING
```

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

Each alert slot encodes one track with ~20 dimensions:

```python
ALERT_SLOT_DIM = 20

def encode_alert_slot(track: SensorTrack | None) -> np.ndarray:
    """Encode track into fixed-size observation slot."""
    slot = np.zeros(ALERT_SLOT_DIM, dtype=np.float32)
    if track is None:
        return slot  # Empty slot

    slot[0] = 1.0  # Slot occupied flag

    # Position (relative to viewer, normalized)
    slot[1:4] = normalize_position(track.position)

    # Velocity
    slot[4:7] = normalize_velocity(track.velocity)

    # Facing (sin/cos encoding)
    slot[7] = np.sin(track.facing_yaw)
    slot[8] = np.cos(track.facing_yaw)

    # Telemetry
    slot[9] = track.heat_level
    slot[10] = track.damage_level
    slot[11] = float(track.is_fallen)
    slot[12] = float(track.is_legged)
    slot[13] = float(track.is_firing)

    # Track metadata
    slot[14] = min(track.track_age / 15.0, 1.0)  # Normalized age
    slot[15] = float(track.paint_active)

    # Behavior one-hot (4 values)
    behavior_idx = list(TargetBehavior).index(track.behavior)
    slot[16 + behavior_idx] = 1.0

    return slot
```

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
| First Detection | +0.3 | Scout is first to add target to pack store |
| Track Usage | +0.5 | Packmate fires at track scout reported |

### 5.2 Implementation

In `rewards.py`:

```python
@dataclass
class RewardWeights:
    # ... existing weights ...

    # Scout intel rewards
    first_detection: float = 0.3  # First to detect enemy
    track_usage: float = 0.5      # Packmate uses your track to shoot
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
        # Merge all packs
        all_tracks: dict[str, SensorTrack] = {}
        for store in pack_stores.values():
            for target_id, track in store.tracks.items():
                if target_id not in all_tracks or track.track_age < all_tracks[target_id].track_age:
                    all_tracks[target_id] = track
        tracks = list(all_tracks.values())
    else:
        # Pack leader just sees own pack
        tracks = list(pack_stores[commander_pack_id].tracks.values())

    # Sort by threat priority
    tracks.sort(key=lambda t: (
        -float(t.paint_active),
        -float(t.behavior == TargetBehavior.ENGAGING),
        -{"heavy": 4, "medium": 3, "light": 2, "scout": 1}.get(t.target_class, 0),
        t.track_age,
    ))

    return tracks[:max_slots]
```

---

## 7. Implementation Tasks

### Task 1: Create Track Data Model

**Files:**
- Create: `echelon/env/tracks.py`
- Test: `tests/unit/test_tracks.py`

**Steps:**
1. Create `TrackReason` enum
2. Create `TargetBehavior` enum
3. Create `SensorTrack` dataclass
4. Create `TrackFreshness` enum (PAINTED, FRESH, STALE, LOST)
5. Add `freshness` property to SensorTrack based on track_age
6. Write unit tests for freshness transitions

### Task 2: Implement PackTrackStore

**Files:**
- Modify: `echelon/env/tracks.py`
- Test: `tests/unit/test_tracks.py`

**Steps:**
1. Implement `PackTrackStore.__init__`
2. Implement `update_track` with first-detection tracking
3. Implement `tick` with age decay and eviction
4. Implement `get_tracks_for_role` with priority sorting
5. Implement `clear_paint`
6. Implement `_infer_behavior`
7. Write unit tests for LRU eviction
8. Write unit tests for freshness decay

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
1. Add `ALERT_SLOT_DIM = 20` constant
2. Add `alert_slots` to `EnvConfig` (default 3)
3. Implement `encode_alert_slot` function
4. Modify `ObservationBuilder.obs_dim()` to include alert slots
5. Modify `ObservationBuilder.build()` to populate alert slots
6. Write test verifying observation dimension increases correctly
7. Write test verifying alert slot content

### Task 5: Add Scout Credit Rewards

**Files:**
- Modify: `echelon/env/rewards.py`
- Modify: `echelon/env/env.py`
- Test: `tests/unit/test_rewards_tracks.py`

**Steps:**
1. Add `first_detection` and `track_usage` to `RewardWeights`
2. Add `step_first_detections` and `step_track_usages` to `StepContext`
3. Add `first_detection` and `track_usage` to `RewardComponents`
4. Implement credit computation in `_compute_agent_reward`
5. Wire credit tracking in `EchelonEnv.step()`
6. Write test verifying scout gets credit on first detection
7. Write test verifying scout gets credit when packmate shoots tracked target

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

- Track freshness transitions
- LRU eviction behavior
- Behavior inference logic
- Alert slot encoding
- Commander track merging

### 8.2 Integration Tests

- Paint action populates track store
- Tracks decay over simulation time
- Alert slots appear in observations
- Scout credit rewards fire correctly

### 8.3 Manual Verification

- Run training with W&B
- Verify perception stats show track activity
- Check replay viewer shows track data (future)

---

## 9. Future Extensions (Not MVP)

- **Order slots**: Pack leader issues orders via separate slot system
- **Track uncertainty**: Position covariance that grows with age
- **Acoustic detections**: Sound-based tracks without LOS
- **Squad-level fusion**: Cross-pack track sharing for large battles
- **Replay visualization**: Show tracks in viewer.html
