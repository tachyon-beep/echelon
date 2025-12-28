"""Match statistics collection during training.

This module provides lightweight stats collection for the training hot path.
It accumulates events from info dicts and produces MatchRecords at episode end.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from echelon.arena.stats import MatchRecord, TeamStats


def _empty_stats() -> dict[str, Any]:
    """Return a zeroed stats dict matching TeamStats fields."""
    return {
        "kills": 0,
        "deaths": 0,
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "zone_ticks": 0,
        "primary_uses": 0,
        "secondary_uses": 0,
        "tertiary_uses": 0,
        "vents": 0,
        "smokes": 0,
        "ecm_toggles": 0,
        "paints": 0,
        "overheats": 0,
        "knockdowns": 0,
        "orders_issued": {},
        "orders_acknowledged": 0,
        "orders_overridden": 0,
    }


@dataclass
class EpisodeStats:
    """Accumulator for a single episode's stats."""

    blue: dict[str, Any] = field(default_factory=_empty_stats)
    red: dict[str, Any] = field(default_factory=_empty_stats)
    step_count: int = 0


class MatchStatsCollector:
    """Collects match statistics from training environments.

    This is designed for the training hot path - keep it lightweight.
    Events are processed from info dicts on each step, and a MatchRecord
    is produced when an episode ends.
    """

    def __init__(self, num_envs: int) -> None:
        """Initialize collector for a vectorized environment.

        Args:
            num_envs: Number of parallel environments.
        """
        self.num_envs = num_envs
        self.episodes: dict[int, EpisodeStats] = {i: EpisodeStats() for i in range(num_envs)}

    def on_step(self, env_idx: int, info: dict[str, Any]) -> None:
        """Process events from a single step.

        Args:
            env_idx: Index of the environment.
            info: Info dict containing 'events' list.
        """
        ep = self.episodes[env_idx]
        ep.step_count += 1

        events = info["events"]
        for event in events:
            self._process_event(ep, event)

    def _team_from_id(self, mech_id: str) -> str:
        """Extract team from mech ID (e.g., 'blue_0' -> 'blue')."""
        return "blue" if mech_id.startswith("blue") else "red"

    def _process_event(self, ep: EpisodeStats, event: dict[str, Any]) -> None:
        """Process a single event and update episode stats.

        The simulation emits specific event types:
        - kill: shooter killed target
        - laser_hit: laser damage dealt (primary weapon for most mechs)
        - projectile_hit: missile/kinetic damage dealt
        - missile_launch: missile fired (secondary for heavy)
        - smoke_launch: smoke grenade used (special slot)
        - paint: target painted (primary for scout, tertiary for light)
        - kinetic_fire: gauss/autocannon fired (tertiary for heavy/medium)
        """
        # NOTE: Use direct access (not .get()) to fail loudly on schema mismatch.
        # If an event is missing expected fields, we want to know immediately.
        etype = event["type"]

        if etype == "kill":
            # Kill event: shooter killed target
            shooter_id = event["shooter"]
            target_id = event["target"]
            shooter_team = self._team_from_id(shooter_id)
            target_team = self._team_from_id(target_id)

            shooter_stats = ep.blue if shooter_team == "blue" else ep.red
            target_stats = ep.blue if target_team == "blue" else ep.red
            shooter_stats["kills"] += 1
            target_stats["deaths"] += 1

        elif etype == "laser_hit":
            # Laser damage: counts as primary weapon use + damage
            shooter_id = event["shooter"]
            target_id = event["target"]
            damage = event["damage"]

            shooter_team = self._team_from_id(shooter_id)
            target_team = self._team_from_id(target_id)

            shooter_stats = ep.blue if shooter_team == "blue" else ep.red
            target_stats = ep.blue if target_team == "blue" else ep.red

            shooter_stats["damage_dealt"] += damage
            target_stats["damage_taken"] += damage
            shooter_stats["primary_uses"] += 1

        elif etype == "projectile_hit":
            # Projectile damage: missile or kinetic hit
            shooter_id = event["shooter"]
            target_id = event["target"]
            damage = event["damage"]

            shooter_team = self._team_from_id(shooter_id)
            target_team = self._team_from_id(target_id)

            shooter_stats = ep.blue if shooter_team == "blue" else ep.red
            target_stats = ep.blue if target_team == "blue" else ep.red

            shooter_stats["damage_dealt"] += damage
            target_stats["damage_taken"] += damage
            # Note: weapon firing counted at launch, not hit

        elif etype == "missile_launch":
            # Missile fired: secondary weapon for heavy
            shooter_id = event["shooter"]
            shooter_team = self._team_from_id(shooter_id)
            shooter_stats = ep.blue if shooter_team == "blue" else ep.red
            shooter_stats["secondary_uses"] += 1

        elif etype == "smoke_launch":
            # Smoke grenade: special slot utility
            shooter_id = event["shooter"]
            shooter_team = self._team_from_id(shooter_id)
            shooter_stats = ep.blue if shooter_team == "blue" else ep.red
            shooter_stats["smokes"] += 1

        elif etype == "paint":
            # Paint target: scout primary, light tertiary
            shooter_id = event["shooter"]
            shooter_team = self._team_from_id(shooter_id)
            shooter_stats = ep.blue if shooter_team == "blue" else ep.red
            shooter_stats["paints"] += 1

        elif etype == "kinetic_fire":
            # Kinetic weapon: gauss (heavy) or autocannon (medium) - tertiary slot
            shooter_id = event["shooter"]
            shooter_team = self._team_from_id(shooter_id)
            shooter_stats = ep.blue if shooter_team == "blue" else ep.red
            shooter_stats["tertiary_uses"] += 1

        elif etype == "order_issued":
            team = event["team"]
            order_type = event["order_type"]
            stats = ep.blue if team == "blue" else ep.red
            # NOTE: .get() is legitimate here - accumulating into a dict that may not have the key
            stats["orders_issued"][order_type] = stats["orders_issued"].get(order_type, 0) + 1

        elif etype == "order_response":
            team = event["team"]
            acknowledged = event["acknowledged"]
            stats = ep.blue if team == "blue" else ep.red
            if acknowledged:
                stats["orders_acknowledged"] += 1
            else:
                stats["orders_overridden"] += 1

    def get_current_stats(self, env_idx: int) -> dict[str, dict[str, Any]]:
        """Get current accumulated stats for an environment.

        Args:
            env_idx: Index of the environment.

        Returns:
            Dict with 'blue' and 'red' keys containing stats copies.
        """
        ep = self.episodes[env_idx]
        return {"blue": ep.blue.copy(), "red": ep.red.copy()}

    def on_episode_end(
        self,
        env_idx: int,
        winner: str,
        termination: str,
        duration_steps: int,
        blue_entry_id: str,
        red_entry_id: str,
        zone_ticks: dict[str, int] | None = None,
    ) -> MatchRecord:
        """Finalize episode stats and return MatchRecord.

        Args:
            env_idx: Index of the environment.
            winner: "blue", "red", or "draw".
            termination: "zone", "elimination", or "timeout".
            duration_steps: Number of steps in the episode.
            blue_entry_id: League entry ID for blue team.
            red_entry_id: League entry ID for red team.
            zone_ticks: Optional dict with 'blue' and 'red' zone tick counts.
                If provided, overrides any accumulated zone_ticks from events.

        Returns:
            MatchRecord containing the episode's statistics.
        """
        ep = self.episodes[env_idx]

        # Apply zone_ticks from environment if provided
        if zone_ticks is not None:
            ep.blue["zone_ticks"] = zone_ticks["blue"]
            ep.red["zone_ticks"] = zone_ticks["red"]

        # Build TeamStats from accumulated dicts
        # Copy orders_issued to avoid mutation issues
        blue_orders = dict(ep.blue["orders_issued"])
        red_orders = dict(ep.red["orders_issued"])

        record = MatchRecord(
            match_id=str(uuid.uuid4()),
            timestamp=time.time(),
            blue_entry_id=blue_entry_id,
            red_entry_id=red_entry_id,
            winner=winner,
            duration_steps=duration_steps,
            termination=termination,
            blue_stats=TeamStats(
                kills=ep.blue["kills"],
                deaths=ep.blue["deaths"],
                damage_dealt=ep.blue["damage_dealt"],
                damage_taken=ep.blue["damage_taken"],
                zone_ticks=ep.blue["zone_ticks"],
                primary_uses=ep.blue["primary_uses"],
                secondary_uses=ep.blue["secondary_uses"],
                tertiary_uses=ep.blue["tertiary_uses"],
                vents=ep.blue["vents"],
                smokes=ep.blue["smokes"],
                ecm_toggles=ep.blue["ecm_toggles"],
                paints=ep.blue["paints"],
                overheats=ep.blue["overheats"],
                knockdowns=ep.blue["knockdowns"],
                orders_issued=blue_orders,
                orders_acknowledged=ep.blue["orders_acknowledged"],
                orders_overridden=ep.blue["orders_overridden"],
            ),
            red_stats=TeamStats(
                kills=ep.red["kills"],
                deaths=ep.red["deaths"],
                damage_dealt=ep.red["damage_dealt"],
                damage_taken=ep.red["damage_taken"],
                zone_ticks=ep.red["zone_ticks"],
                primary_uses=ep.red["primary_uses"],
                secondary_uses=ep.red["secondary_uses"],
                tertiary_uses=ep.red["tertiary_uses"],
                vents=ep.red["vents"],
                smokes=ep.red["smokes"],
                ecm_toggles=ep.red["ecm_toggles"],
                paints=ep.red["paints"],
                overheats=ep.red["overheats"],
                knockdowns=ep.red["knockdowns"],
                orders_issued=red_orders,
                orders_acknowledged=ep.red["orders_acknowledged"],
                orders_overridden=ep.red["orders_overridden"],
            ),
        )

        # Reset for next episode
        self.episodes[env_idx] = EpisodeStats()

        return record
