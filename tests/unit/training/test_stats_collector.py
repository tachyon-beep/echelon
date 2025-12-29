"""Tests for match stats collection during training.

These tests use the actual event types emitted by the simulation:
- kill: {type: "kill", shooter: "blue_0", target: "red_0"}
- laser_hit: {type: "laser_hit", shooter: "...", target: "...", damage: float}
- projectile_hit: {type: "projectile_hit", shooter: "...", target: "...", damage: float}
- missile_launch: {type: "missile_launch", shooter: "..."}
- smoke_launch: {type: "smoke_launch", shooter: "..."}
- paint: {type: "paint", shooter: "...", target: "..."}
- kinetic_fire: {type: "kinetic_fire", shooter: "..."}
"""

from echelon.training.stats_collector import MatchStatsCollector


def test_collector_tracks_kills():
    """Collector accumulates kill events using actual sim event format."""
    collector = MatchStatsCollector(num_envs=1)

    # Actual format from sim.py: shooter/target, not killer/victim
    info = {"events": [{"type": "kill", "shooter": "blue_0", "target": "red_0"}]}
    collector.on_step(env_idx=0, info=info)

    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["kills"] == 1
    assert stats["red"]["deaths"] == 1


def test_collector_tracks_damage():
    """Collector accumulates damage from laser_hit and projectile_hit events."""
    collector = MatchStatsCollector(num_envs=1)

    # Actual format: laser_hit with shooter/target/damage
    info = {
        "events": [
            {
                "type": "laser_hit",
                "weapon": "laser",
                "shooter": "blue_1",
                "target": "red_2",
                "damage": 25.0,
                "pos": [10.0, 10.0, 5.0],
            }
        ]
    }
    collector.on_step(env_idx=0, info=info)

    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["damage_dealt"] == 25.0
    assert stats["red"]["damage_taken"] == 25.0
    assert stats["blue"]["primary_uses"] == 1  # laser_hit counts as primary


def test_collector_tracks_projectile_damage():
    """Collector accumulates damage from projectile hits (missiles, kinetics)."""
    collector = MatchStatsCollector(num_envs=1)

    info = {
        "events": [
            {
                "type": "projectile_hit",
                "weapon": "missile",
                "shooter": "red_0",
                "target": "blue_1",
                "damage": 40.0,
                "pos": [15.0, 15.0, 5.0],
            }
        ]
    }
    collector.on_step(env_idx=0, info=info)

    stats = collector.get_current_stats(env_idx=0)
    assert stats["red"]["damage_dealt"] == 40.0
    assert stats["blue"]["damage_taken"] == 40.0


def test_collector_tracks_weapon_usage():
    """Collector counts weapon activations using actual event types."""
    collector = MatchStatsCollector(num_envs=1)

    info = {
        "events": [
            # Two laser hits = 2 primary uses
            {"type": "laser_hit", "shooter": "blue_0", "target": "red_0", "damage": 14.0},
            {"type": "laser_hit", "shooter": "blue_0", "target": "red_1", "damage": 14.0},
            # One missile launch = 1 secondary use
            {"type": "missile_launch", "shooter": "blue_1", "target": "red_0", "weapon": "missile"},
            # One kinetic fire = 1 tertiary use
            {"type": "kinetic_fire", "shooter": "blue_2", "weapon": "gauss"},
        ]
    }
    collector.on_step(env_idx=0, info=info)

    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["primary_uses"] == 2
    assert stats["blue"]["secondary_uses"] == 1
    assert stats["blue"]["tertiary_uses"] == 1


def test_collector_tracks_utilities():
    """Collector counts utility usage (smoke, paint)."""
    collector = MatchStatsCollector(num_envs=1)

    info = {
        "events": [
            {"type": "smoke_launch", "shooter": "blue_0", "weapon": "smoke"},
            {"type": "paint", "shooter": "red_0", "target": "blue_1", "weapon": "painter"},
            {"type": "paint", "shooter": "red_0", "target": "blue_2", "weapon": "painter"},
        ]
    }
    collector.on_step(env_idx=0, info=info)

    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["smokes"] == 1
    assert stats["red"]["paints"] == 2


def test_collector_resets_on_episode_end():
    """Collector resets stats when episode ends."""
    collector = MatchStatsCollector(num_envs=1)

    info = {"events": [{"type": "kill", "shooter": "blue_0", "target": "red_0"}]}
    collector.on_step(env_idx=0, info=info)

    # End episode
    record = collector.on_episode_end(
        env_idx=0,
        winner="blue",
        termination="zone",
        duration_steps=100,
        blue_entry_id="contender",
        red_entry_id="heuristic",
    )

    assert record.blue_stats.kills == 1

    # Stats should be reset
    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["kills"] == 0


def test_collector_handles_multiple_envs():
    """Collector tracks stats independently per environment."""
    collector = MatchStatsCollector(num_envs=3)

    # Events in different environments
    collector.on_step(0, {"events": [{"type": "kill", "shooter": "blue_0", "target": "red_0"}]})
    collector.on_step(1, {"events": [{"type": "kill", "shooter": "red_0", "target": "blue_0"}]})
    collector.on_step(2, {"events": []})

    stats0 = collector.get_current_stats(0)
    stats1 = collector.get_current_stats(1)
    stats2 = collector.get_current_stats(2)

    assert stats0["blue"]["kills"] == 1
    assert stats0["red"]["kills"] == 0
    assert stats1["red"]["kills"] == 1
    assert stats1["blue"]["kills"] == 0
    assert stats2["blue"]["kills"] == 0
    assert stats2["red"]["kills"] == 0


def test_collector_zone_ticks_from_env():
    """Collector populates zone_ticks from environment stats at episode end."""
    collector = MatchStatsCollector(num_envs=1)

    # Simulate some steps (zone ticks come from env, not events)
    collector.on_step(0, {"events": []})
    collector.on_step(0, {"events": []})

    # End episode with zone ticks from environment
    zone_ticks = {"blue": 45, "red": 23}
    record = collector.on_episode_end(
        env_idx=0,
        winner="blue",
        termination="zone",
        duration_steps=100,
        blue_entry_id="contender",
        red_entry_id="heuristic",
        zone_ticks=zone_ticks,
    )

    # Zone ticks should be populated in the match record
    assert record.blue_stats.zone_ticks == 45
    assert record.red_stats.zone_ticks == 23


def test_collector_zone_ticks_defaults_to_zero():
    """Without zone_ticks parameter, zone_ticks defaults to 0."""
    collector = MatchStatsCollector(num_envs=1)

    collector.on_step(0, {"events": []})

    # End episode without zone_ticks parameter
    record = collector.on_episode_end(
        env_idx=0,
        winner="draw",
        termination="timeout",
        duration_steps=50,
        blue_entry_id="a",
        red_entry_id="b",
    )

    # Zone ticks should be 0 (default)
    assert record.blue_stats.zone_ticks == 0
    assert record.red_stats.zone_ticks == 0
