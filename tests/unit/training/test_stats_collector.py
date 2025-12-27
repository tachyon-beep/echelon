"""Tests for match stats collection during training."""

from echelon.training.stats_collector import MatchStatsCollector


def test_collector_tracks_kills():
    """Collector accumulates kill events."""
    collector = MatchStatsCollector(num_envs=1)

    info = {"events": [{"type": "kill", "killer": "blue_0", "victim": "red_0", "team": "blue"}]}
    collector.on_step(env_idx=0, info=info)

    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["kills"] == 1
    assert stats["red"]["deaths"] == 1


def test_collector_tracks_damage():
    """Collector accumulates damage events."""
    collector = MatchStatsCollector(num_envs=1)

    info = {
        "events": [
            {
                "type": "damage",
                "attacker": "blue_1",
                "target": "red_2",
                "amount": 25.0,
                "attacker_team": "blue",
            }
        ]
    }
    collector.on_step(env_idx=0, info=info)

    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["damage_dealt"] == 25.0
    assert stats["red"]["damage_taken"] == 25.0


def test_collector_tracks_weapon_usage():
    """Collector counts weapon activations."""
    collector = MatchStatsCollector(num_envs=1)

    info = {
        "events": [
            {"type": "weapon_fired", "agent": "blue_0", "weapon": "primary", "team": "blue"},
            {"type": "weapon_fired", "agent": "blue_0", "weapon": "primary", "team": "blue"},
            {"type": "weapon_fired", "agent": "blue_1", "weapon": "secondary", "team": "blue"},
        ]
    }
    collector.on_step(env_idx=0, info=info)

    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["primary_uses"] == 2
    assert stats["blue"]["secondary_uses"] == 1


def test_collector_resets_on_episode_end():
    """Collector resets stats when episode ends."""
    collector = MatchStatsCollector(num_envs=1)

    info = {"events": [{"type": "kill", "killer": "blue_0", "victim": "red_0", "team": "blue"}]}
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
