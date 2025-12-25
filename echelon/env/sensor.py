"""Sensor and tracking system for combat suites.

This module provides the intermediate layer between raw world state and
what gets displayed in the cockpit. It models:
- Track persistence and ageing
- Classification uncertainty
- Source attribution (visual, radar, datalink)
- Threat assessment

The key insight: "The mech can know more than it shows."
Sensors produce tracks; the DisplayManager curates what fills the display.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SensorTrack:
    """A detected entity with uncertainty.

    This is what sensors produce - not perfect entity states, but
    probabilistic estimates that degrade over time.
    """

    track_id: str  # Persistent ID for this track
    entity_id: str  # Actual entity being tracked (ground truth reference)

    # === POSITION ESTIMATE ===
    position: np.ndarray  # (3,) estimated position
    position_uncertainty: float  # Confidence radius in voxels
    velocity: np.ndarray  # (3,) estimated velocity

    # === CLASSIFICATION ===
    # Probabilities for each mech class (scout/light/medium/heavy)
    class_probabilities: np.ndarray  # (4,) sums to 1.0
    class_confidence: float  # How sure we are (0-1)

    # === TRACK QUALITY ===
    confidence: float  # Overall track quality (0-1)
    time_since_update: float  # Seconds since last sensor hit
    is_stale: bool = False  # Exceeded persistence threshold

    # === SOURCE ATTRIBUTION ===
    source_visual: bool = False  # Direct LOS observation
    source_datalink: bool = False  # Received from squadmate
    source_inferred: bool = False  # Predicted from motion model

    # === THREAT ASSESSMENT ===
    is_hostile: bool = True  # False for friendlies
    is_targeting_me: bool = False  # Pointing at observer
    threat_priority: float = 0.0  # Computed threat score (higher = more dangerous)

    # === TIMESTAMPS ===
    first_detected: float = 0.0  # When first acquired
    last_updated: float = 0.0  # When last refreshed

    @property
    def most_likely_class(self) -> str:
        """Return the most likely mech class."""
        classes = ["scout", "light", "medium", "heavy"]
        return classes[int(np.argmax(self.class_probabilities))]

    @property
    def is_fresh(self) -> bool:
        """Track was updated recently (< 1 second)."""
        return self.time_since_update < 1.0

    def age(self, dt: float) -> None:
        """Age the track by dt seconds."""
        self.time_since_update += dt

        # Confidence decays over time
        decay_rate = 0.1  # Lose 10% confidence per second without updates
        self.confidence *= 1.0 - (decay_rate * dt)
        self.confidence = max(0.0, self.confidence)

        # Classification becomes uncertain
        self.class_confidence *= 1.0 - (decay_rate * 0.5 * dt)
        self.class_confidence = max(0.0, self.class_confidence)

        # Mark as stale after threshold
        if self.time_since_update > 5.0:
            self.is_stale = True


@dataclass
class TrackStore:
    """Storage for all sensor tracks.

    This is the "full picture" that the mech knows about.
    The DisplayManager filters this down to what fits in the cockpit.
    """

    tracks: dict[str, SensorTrack] = field(default_factory=dict)
    next_track_id: int = 0

    # Track ageing thresholds
    STALE_THRESHOLD: float = 5.0  # Seconds until track goes stale
    DROP_THRESHOLD: float = 15.0  # Seconds until track is dropped

    def add_or_update_track(
        self,
        entity_id: str,
        position: np.ndarray,
        velocity: np.ndarray,
        mech_class: str,
        is_hostile: bool,
        sim_time: float,
        source: str = "visual",
        confidence: float = 1.0,
    ) -> SensorTrack:
        """Add a new track or update an existing one.

        Args:
            entity_id: The actual entity ID being tracked
            position: Current position estimate
            velocity: Current velocity estimate
            mech_class: Known or estimated class
            is_hostile: Whether this is an enemy
            sim_time: Current simulation time
            source: "visual", "datalink", or "inferred"
            confidence: Observation confidence (0-1)

        Returns:
            The created or updated track
        """
        # Check if we already have a track for this entity
        for track in self.tracks.values():
            if track.entity_id == entity_id:
                # Update existing track
                self._update_track(track, position, velocity, mech_class, sim_time, source, confidence)
                return track

        # Create new track
        track_id = f"trk_{self.next_track_id}"
        self.next_track_id += 1

        # Create class probability distribution
        class_probs = self._class_to_probs(mech_class, confidence)

        track = SensorTrack(
            track_id=track_id,
            entity_id=entity_id,
            position=position.copy(),
            position_uncertainty=1.0 - confidence,
            velocity=velocity.copy(),
            class_probabilities=class_probs,
            class_confidence=confidence,
            confidence=confidence,
            time_since_update=0.0,
            source_visual=(source == "visual"),
            source_datalink=(source == "datalink"),
            source_inferred=(source == "inferred"),
            is_hostile=is_hostile,
            first_detected=sim_time,
            last_updated=sim_time,
        )

        self.tracks[track_id] = track
        return track

    def _update_track(
        self,
        track: SensorTrack,
        position: np.ndarray,
        velocity: np.ndarray,
        mech_class: str,
        sim_time: float,
        source: str,
        confidence: float,
    ) -> None:
        """Update an existing track with new observation."""
        # Blend position estimate (simple Kalman-like update)
        alpha = min(1.0, confidence * 0.8)
        track.position = (1 - alpha) * track.position + alpha * position
        track.velocity = (1 - alpha) * track.velocity + alpha * velocity

        # Reduce uncertainty
        track.position_uncertainty = max(0.1, track.position_uncertainty * (1 - confidence * 0.5))

        # Update classification
        new_probs = self._class_to_probs(mech_class, confidence)
        track.class_probabilities = (track.class_probabilities + new_probs) / 2
        track.class_probabilities /= track.class_probabilities.sum()  # Renormalize
        track.class_confidence = min(1.0, track.class_confidence + confidence * 0.3)

        # Boost confidence
        track.confidence = min(1.0, track.confidence + confidence * 0.5)
        track.time_since_update = 0.0
        track.is_stale = False
        track.last_updated = sim_time

        # Update source flags
        if source == "visual":
            track.source_visual = True
        elif source == "datalink":
            track.source_datalink = True
        elif source == "inferred":
            track.source_inferred = True

    def _class_to_probs(self, mech_class: str, confidence: float) -> np.ndarray:
        """Convert a class name to probability distribution."""
        classes = ["scout", "light", "medium", "heavy"]
        probs = np.ones(4) * (1 - confidence) / 4  # Uniform baseline

        if mech_class in classes:
            idx = classes.index(mech_class)
            probs[idx] = confidence + probs[idx]

        probs /= probs.sum()
        return probs

    def age_all(self, dt: float) -> None:
        """Age all tracks and remove dead ones."""
        to_remove = []
        for track_id, track in self.tracks.items():
            track.age(dt)
            if track.time_since_update > self.DROP_THRESHOLD:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

    def get_tracks_sorted_by_threat(self) -> list[SensorTrack]:
        """Return tracks sorted by threat priority (highest first)."""
        return sorted(self.tracks.values(), key=lambda t: -t.threat_priority)

    def get_hostile_tracks(self) -> list[SensorTrack]:
        """Return only hostile tracks."""
        return [t for t in self.tracks.values() if t.is_hostile]

    def get_friendly_tracks(self) -> list[SensorTrack]:
        """Return only friendly tracks."""
        return [t for t in self.tracks.values() if not t.is_hostile]

    def clear(self) -> None:
        """Clear all tracks."""
        self.tracks.clear()


class DisplayManager:
    """Curates which tracks fill the display slots.

    The cockpit can only show K contacts. The DisplayManager decides
    which K of all known tracks get displayed based on:
    - Priority (threat, objective relevance)
    - Recency (stale tracks deprioritized)
    - Agent preferences (sort mode from action)

    This is where the "observation control" action takes effect.
    """

    # Sort modes that agents can select
    SORT_CLOSEST = 0
    SORT_BIGGEST = 1
    SORT_MOST_DAMAGED = 2

    def __init__(self, max_contacts: int):
        """Initialize display manager.

        Args:
            max_contacts: Maximum number of contacts to display
        """
        self.max_contacts = max_contacts
        self.current_sort_mode = self.SORT_CLOSEST
        self.filter_hostile_only = False

    def set_sort_mode(self, mode: int) -> None:
        """Set the current sort mode (from agent action)."""
        self.current_sort_mode = mode

    def set_filter_hostile_only(self, hostile_only: bool) -> None:
        """Set whether to filter to hostile contacts only."""
        self.filter_hostile_only = hostile_only

    def curate(
        self,
        track_store: TrackStore,
        viewer_pos: np.ndarray,
        viewer_hp_frac: float = 1.0,
    ) -> list[SensorTrack]:
        """Select and order tracks for display.

        Args:
            track_store: All known tracks
            viewer_pos: Observer's position
            viewer_hp_frac: Observer's HP fraction (affects threat priority)

        Returns:
            Sorted list of up to max_contacts tracks to display
        """
        # Start with all tracks
        candidates = list(track_store.tracks.values())

        # Apply hostile filter if set
        if self.filter_hostile_only:
            candidates = [t for t in candidates if t.is_hostile]

        # Update threat priorities
        for track in candidates:
            track.threat_priority = compute_threat_priority(track, viewer_pos, viewer_hp_frac)

        # Sort by the selected mode
        if self.current_sort_mode == self.SORT_CLOSEST:
            # Sort by distance (closest first)
            candidates.sort(key=lambda t: float(np.linalg.norm(t.position - viewer_pos)))
        elif self.current_sort_mode == self.SORT_BIGGEST:
            # Sort by threat class (heavy > medium > light > scout)
            class_order = {"heavy": 0, "medium": 1, "light": 2, "scout": 3}
            candidates.sort(key=lambda t: class_order.get(t.most_likely_class, 4))
        elif self.current_sort_mode == self.SORT_MOST_DAMAGED:
            # Sort by confidence that they're damaged (higher confidence first)
            # This is a proxy for "easy kills" - low HP, high confidence
            candidates.sort(key=lambda t: -t.confidence * (1.0 - t.threat_priority))
        else:
            # Fallback: sort by threat priority
            candidates.sort(key=lambda t: -t.threat_priority)

        # Secondary sort: within same priority, prefer fresh tracks
        # (This is a stable sort, so primary sort order is preserved for ties)
        candidates.sort(key=lambda t: t.time_since_update)

        # Take top K
        return candidates[: self.max_contacts]

    def curate_to_vectors(
        self,
        track_store: TrackStore,
        viewer_pos: np.ndarray,
        viewer_hp_frac: float = 1.0,
        entity_dim: int = 25,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Curate tracks and convert to observation vectors.

        Returns:
            entities: (max_contacts, entity_dim) padded entity features
            mask: (max_contacts,) 1.0 for padding, 0.0 for real
        """
        tracks = self.curate(track_store, viewer_pos, viewer_hp_frac)

        entities = np.zeros((self.max_contacts, entity_dim), dtype=np.float32)
        mask = np.ones(self.max_contacts, dtype=np.float32)

        for i, track in enumerate(tracks):
            entities[i] = _track_to_vector(track, viewer_pos, entity_dim)
            mask[i] = 0.0

        return entities, mask


def _track_to_vector(
    track: SensorTrack,
    viewer_pos: np.ndarray,
    entity_dim: int = 25,
) -> np.ndarray:
    """Convert a sensor track to an observation vector.

    Layout (25 dims):
    - rel_pos (3): relative position
    - rel_vel (3): relative velocity
    - yaw (2): sin/cos of facing
    - hp_heat (2): normalized HP and heat (if known)
    - stab_fallen_legged (3): stability info
    - relation_onehot (3): hostile/friendly/unknown
    - class_onehot (4): scout/light/medium/heavy
    - flags (5): painted, visible, threat_high, stale, fresh
    """
    vec = np.zeros(entity_dim, dtype=np.float32)

    # Relative position (normalized to ~20 voxel range)
    rel_pos = track.position - viewer_pos
    vec[0:3] = rel_pos / 20.0

    # Relative velocity (normalized)
    vec[3:6] = track.velocity / 10.0

    # Facing direction (we don't track this, so use 0)
    vec[6:8] = 0.0  # Unknown facing

    # HP and heat (unknown from sensors, could use class averages)
    vec[8:10] = 0.5  # Default to mid-range

    # Stability/fallen/legged (unknown)
    vec[10:13] = [0.0, 0.0, 0.0]

    # Relation one-hot
    if track.is_hostile:
        vec[13:16] = [1.0, 0.0, 0.0]  # Hostile
    else:
        vec[13:16] = [0.0, 1.0, 0.0]  # Friendly

    # Class probabilities (use directly instead of one-hot)
    vec[16:20] = track.class_probabilities

    # Flags
    vec[20] = 0.0  # painted (unknown at sensor level)
    vec[21] = 1.0 if track.source_visual else 0.5  # visible
    vec[22] = 1.0 if track.threat_priority > 0.7 else 0.0  # threat_high
    vec[23] = 1.0 if track.is_stale else 0.0  # stale
    vec[24] = 1.0 if track.is_fresh else 0.0  # fresh

    return vec


def compute_threat_priority(
    track: SensorTrack,
    viewer_pos: np.ndarray,
    viewer_hp_frac: float,
) -> float:
    """Compute threat priority for a track.

    Factors:
    - Distance (closer = more threatening)
    - Mech class (heavier = more threatening)
    - Is targeting me (much more threatening)
    - Track confidence (uncertain threats are less prioritized)

    Returns:
        Threat score (higher = more dangerous)
    """
    # Distance factor (inverse, capped)
    dist = float(np.linalg.norm(track.position - viewer_pos))
    dist_factor = 1.0 / max(1.0, dist / 10.0)  # Normalize to ~10 voxels

    # Class factor (heavier = more dangerous)
    class_weights = {"scout": 0.5, "light": 0.7, "medium": 1.0, "heavy": 1.5}
    class_factor = class_weights.get(track.most_likely_class, 1.0)

    # Targeting factor (huge boost if aimed at me)
    targeting_factor = 3.0 if track.is_targeting_me else 1.0

    # Confidence weighting (uncertain tracks are lower priority)
    conf_factor = 0.3 + 0.7 * track.confidence

    # Low HP makes everything scarier
    hp_factor = 1.0 + (1.0 - viewer_hp_frac) * 0.5

    return dist_factor * class_factor * targeting_factor * conf_factor * hp_factor
