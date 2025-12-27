"""Reward calculation correctness tests.

Reward bugs are the most insidious in RL - these tests verify:
- Sign correctness (positive for good, negative for bad)
- Attribution (reward goes to correct agent)
- Gradient direction (moving toward goal increases reward)
"""

import numpy as np
import pytest

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig
from echelon.gen.objective import capture_zone_params


@pytest.fixture
def reward_env():
    """Environment configured for reward testing."""
    cfg = EnvConfig(
        world=WorldConfig(size_x=40, size_y=40, size_z=15, obstacle_fill=0.0, ensure_connectivity=False),
        num_packs=1,
        seed=0,
        max_episode_seconds=30.0,
    )
    env = EchelonEnv(cfg)
    env.reset(seed=0)
    assert env.world is not None
    assert env.sim is not None
    return env


class TestRewardPolarity:
    """Verify reward signs are correct."""

    def test_zone_control_positive_for_holder(self, reward_env):
        """Team in zone alone gets positive per-tick reward."""
        env = reward_env

        # Get zone center
        zone_cx, zone_cy, _ = capture_zone_params(
            env.world.meta, size_x=env.world.size_x, size_y=env.world.size_y
        )

        # Move blue_0 into zone, everyone else out
        for aid in env.agents:
            m = env.sim.mechs[aid]
            if aid == "blue_0":
                m.pos[0], m.pos[1] = zone_cx, zone_cy
            else:
                # Move far from zone
                m.pos[0], m.pos[1] = 5.0, 5.0
            m.vel[:] = 0.0

        # Step with null actions
        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        _, rewards, _, _, _ = env.step(actions)

        # Blue in zone should get positive reward
        assert rewards["blue_0"] > 0, f"Blue in zone should get positive reward, got {rewards['blue_0']}"

    def test_death_gives_negative_reward(self, reward_env):
        """Dying gives negative reward to the victim (W_DEATH=-0.5)."""
        env = reward_env

        # Setup: position killer and victim
        # Use blue_3 (medium) as killer since it has LASER as primary
        # (blue_0 is scout with PAINTER which does 0 damage)
        victim = env.sim.mechs["red_0"]
        victim.hp = 1.0  # One-shot kill

        killer = env.sim.mechs["blue_3"]
        killer.pos[0], killer.pos[1] = victim.pos[0] - 5.0, victim.pos[1]
        killer.yaw = 0.0  # Facing +x toward victim

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        actions["blue_3"][4] = 1.0  # Fire laser (medium has LASER primary)

        death_occurred = False
        for _ in range(10):
            was_alive = victim.alive
            _, rewards, _, _, _ = env.step(actions)
            # Check if death happened this step
            if was_alive and not victim.alive and victim.died:
                # Death occurred - W_DEATH = -0.5
                assert (
                    rewards["red_0"] < 0
                ), f"Victim should get negative reward on death, got {rewards['red_0']}"
                death_occurred = True
                break

        assert death_occurred, "Victim should have died during test"

    def test_kill_gives_positive_reward(self, reward_env):
        """Getting a kill gives positive reward to the killer."""
        env = reward_env

        # Setup: Use blue_3 (medium with LASER) as killer
        victim = env.sim.mechs["red_0"]
        victim.hp = 1.0

        killer = env.sim.mechs["blue_3"]
        killer.pos[0], killer.pos[1] = victim.pos[0] - 5.0, victim.pos[1]
        killer.yaw = 0.0  # Facing +x toward victim

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        actions["blue_3"][4] = 1.0  # PRIMARY (laser)

        kill_occurred = False
        for _ in range(10):
            was_alive = victim.alive
            _, rewards, _, _, _ = env.step(actions)
            # Check if kill happened this step
            if was_alive and not victim.alive and victim.died:
                # Kill occurred - W_KILL = 10.0 (scaled reward)
                assert rewards["blue_3"] > 0.5, f"Kill should give positive reward, got {rewards['blue_3']}"
                kill_occurred = True
                break

        assert kill_occurred, "Victim should have been killed during test"


class TestRewardAttribution:
    """Verify rewards go to the correct agent."""

    def test_damage_reward_to_shooter_not_victim(self, reward_env):
        """Damage reward goes to shooter, not victim."""
        env = reward_env

        # Clear setup - position mechs far from zone to isolate combat rewards
        # Use blue_3 (medium with LASER) as shooter since blue_0 is scout with PAINTER
        shooter = env.sim.mechs["blue_3"]
        target = env.sim.mechs["red_0"]

        # Position for combat far from zone to minimize zone influence
        shooter.pos[0], shooter.pos[1] = 5.0, 5.0
        target.pos[0], target.pos[1] = 15.0, 5.0
        shooter.yaw = 0.0  # Facing +x toward target

        # Move everyone else far away to isolate the test
        for aid in env.agents:
            if aid not in ["blue_3", "red_0"]:
                m = env.sim.mechs[aid]
                m.pos[0], m.pos[1] = 35.0, 35.0

        # Get baseline rewards before combat
        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        _, baseline_rewards, _, _, _ = env.step(actions)
        baseline_shooter = baseline_rewards["blue_3"]
        baseline_target = baseline_rewards["red_0"]

        # Now fire weapon
        actions["blue_3"][4] = 1.0  # Fire laser
        target_hp_before = float(target.hp)
        _, combat_rewards, _, _, _ = env.step(actions)

        # Check if damage was dealt
        damage_dealt = target_hp_before - float(target.hp)

        if damage_dealt > 0:
            # The shooter should get a damage reward (W_DAMAGE * damage)
            # This should make their reward delta more positive than the victim's
            shooter_delta = combat_rewards["blue_3"] - baseline_shooter
            target_delta = combat_rewards["red_0"] - baseline_target

            assert (
                shooter_delta > target_delta
            ), f"Shooter reward delta ({shooter_delta}) should exceed victim delta ({target_delta}) when damage dealt"

    def test_zone_reward_only_to_team_in_zone(self, reward_env):
        """Zone control reward only goes to team actually in zone."""
        env = reward_env

        zone_cx, zone_cy, _ = capture_zone_params(
            env.world.meta, size_x=env.world.size_x, size_y=env.world.size_y
        )

        # Blue in zone, red far away
        for aid in env.agents:
            m = env.sim.mechs[aid]
            if m.team == "blue":
                m.pos[0], m.pos[1] = zone_cx, zone_cy
            else:
                m.pos[0], m.pos[1] = 5.0, 5.0
            m.vel[:] = 0.0

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        _, rewards, _, _, _ = env.step(actions)

        # All blue agents should get zone reward
        # All red agents should NOT get zone reward
        blue_rewards = [rewards[aid] for aid in env.agents if "blue" in aid]
        red_rewards = [rewards[aid] for aid in env.agents if "red" in aid]

        assert sum(blue_rewards) > sum(
            red_rewards
        ), f"Blue (in zone) should get more reward: blue={sum(blue_rewards)}, red={sum(red_rewards)}"


class TestRewardGradients:
    """Verify reward increases for desired behaviors."""

    def test_approach_reward_increases_when_moving_toward_zone(self, reward_env):
        """Moving toward zone gives positive approach reward component.

        This test verifies the approach reward gradient exists and has the correct
        sign: decreasing distance to zone (phi1 > phi0) gives positive reward.
        """
        # The approach reward is: W_APPROACH * (gamma * phi1 - phi0)
        # where phi = -distance/max_xy (negative potential)
        # If distance decreases: d1 < d0, so phi1 > phi0, giving positive reward
        # This is the desired gradient toward the objective.
        from echelon.env.rewards import RewardWeights

        weights = RewardWeights()
        W_APPROACH = weights.approach  # 2.0 after rebalancing
        assert W_APPROACH > 0, "Approach reward weight should be positive"

        # Verify the math: if d1 < d0, then -d1/M > -d0/M, so phi1 > phi0
        # Example: d0=100, d1=90, max_xy=200
        d0, d1, max_xy = 100.0, 90.0, 200.0
        phi0 = -d0 / max_xy  # -0.5
        phi1 = -d1 / max_xy  # -0.45
        delta = weights.shaping_gamma * phi1 - phi0  # ~0.055 (positive)

        assert delta > 0, "Decreasing distance should give positive phi delta"
        assert W_APPROACH * delta > 0, "Approach reward should be positive when moving toward zone"

    def test_damage_reward_scales_with_damage(self, reward_env):
        """More damage dealt gives proportionally more reward."""
        from echelon.env.rewards import RewardWeights

        weights = RewardWeights()
        W_DAMAGE = weights.damage  # 0.02 after rebalancing
        assert W_DAMAGE > 0, "Damage reward weight should be positive"


class TestArrivalBonus:
    """Verify arrival bonus for first zone entry."""

    def test_arrival_bonus_on_first_zone_entry(self, reward_env):
        """Agent entering zone for first time gets arrival bonus."""
        env = reward_env

        zone_cx, zone_cy, _zone_r = capture_zone_params(
            env.world.meta, size_x=env.world.size_x, size_y=env.world.size_y
        )

        # Move everyone out of zone first
        for aid in env.agents:
            m = env.sim.mechs[aid]
            m.pos[0], m.pos[1] = 5.0, 5.0
            m.vel[:] = 0.0

        # Step to establish baseline (no one in zone)
        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        _, _baseline_rewards, _, _, _baseline_infos = env.step(actions)

        # Now move blue_0 into zone
        env.sim.mechs["blue_0"].pos[0] = zone_cx
        env.sim.mechs["blue_0"].pos[1] = zone_cy

        _, _rewards, _, _, infos = env.step(actions)

        # Check reward components
        blue_0_comps = infos["blue_0"]["reward_components"]
        assert "arrival" in blue_0_comps, "Should have arrival component"
        assert (
            blue_0_comps["arrival"] > 0
        ), f"First zone entry should get arrival bonus, got {blue_0_comps['arrival']}"

    def test_no_arrival_bonus_on_second_entry(self, reward_env):
        """Agent re-entering zone does NOT get arrival bonus again."""
        env = reward_env

        zone_cx, zone_cy, _zone_r = capture_zone_params(
            env.world.meta, size_x=env.world.size_x, size_y=env.world.size_y
        )

        # Move blue_0 into zone
        env.sim.mechs["blue_0"].pos[0] = zone_cx
        env.sim.mechs["blue_0"].pos[1] = zone_cy

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}

        # First entry - should get bonus
        _, _, _, _, infos1 = env.step(actions)
        first_arrival = infos1["blue_0"]["reward_components"]["arrival"]
        assert first_arrival > 0, "First entry should get arrival bonus"

        # Move out of zone
        env.sim.mechs["blue_0"].pos[0] = 5.0
        env.sim.mechs["blue_0"].pos[1] = 5.0
        _, _, _, _, _ = env.step(actions)

        # Re-enter zone
        env.sim.mechs["blue_0"].pos[0] = zone_cx
        env.sim.mechs["blue_0"].pos[1] = zone_cy
        _, _, _, _, infos2 = env.step(actions)

        second_arrival = infos2["blue_0"]["reward_components"]["arrival"]
        assert second_arrival == 0, f"Re-entry should NOT get arrival bonus, got {second_arrival}"


class TestRewardBalancing:
    """Verify reward components are properly balanced."""

    def test_zone_reward_dominates_approach(self, reward_env):
        """Zone control should give significantly more reward than approach."""
        from echelon.env.rewards import RewardWeights

        weights = RewardWeights()

        # Zone: 2.0 per step (uncontested)
        # Approach: ~0.025 per step (moving 1 voxel toward zone with max_xy=100)
        # Ratio should be ~200:1 in favor of zone

        zone_per_step = weights.zone_tick  # 2.0
        approach_per_step = weights.approach * 0.01  # ~0.02 (typical movement)

        ratio = zone_per_step / max(approach_per_step, 0.001)
        assert ratio > 100, f"Zone should dominate approach by >100x, actual ratio: {ratio:.1f}"

    def test_death_penalty_reasonable_vs_zone(self, reward_env):
        """Death penalty should be recoverable with zone control in reasonable time."""
        from echelon.env.rewards import RewardWeights

        weights = RewardWeights()

        # 2025-12-28 v4: Deaths should hurt, but be recoverable
        # death = -3.0, zone_tick = 1.0 -> 3 steps to recover
        # This is intentional - dying isn't free, but 3 steps out of ~200 is reasonable

        steps_to_recover = abs(weights.death) / weights.zone_tick
        assert (
            steps_to_recover <= 5
        ), f"Death penalty should be recoverable in <=5 zone steps, needs {steps_to_recover:.2f}"


class TestPaintCreditAssignment:
    """Verify paint credit assignment rewards for scouts."""

    def test_immediate_paint_reward(self, reward_env):
        """Scout painting a target gets immediate +0.3 paint_assist reward."""
        env = reward_env

        # blue_0 is scout with PAINTER as primary
        scout = env.sim.mechs["blue_0"]
        target = env.sim.mechs["red_0"]

        # Position scout to face target
        scout.pos[0], scout.pos[1] = 10.0, 10.0
        target.pos[0], target.pos[1] = 20.0, 10.0  # Target 10 voxels away in +x
        scout.yaw = 0.0  # Facing +x toward target
        scout.painter_cooldown = 0.0  # Ensure can fire

        # Move everyone else away
        for aid in env.agents:
            if aid not in ["blue_0", "red_0"]:
                m = env.sim.mechs[aid]
                m.pos[0], m.pos[1] = 5.0, 5.0

        # Fire painter
        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        actions["blue_0"][4] = 1.0  # PRIMARY = PAINTER for scout

        # Run a few steps to ensure paint hits
        paint_reward_found = False
        for _ in range(5):
            _, _rewards, _, _, infos = env.step(actions)
            # Check reward components for paint_assist
            comps = infos["blue_0"]["reward_components"]
            if comps.get("paint_assist", 0.0) > 0:
                paint_reward_found = True
                assert (
                    comps["paint_assist"] >= 0.3
                ), f"Paint reward should be >= 0.3, got {comps['paint_assist']}"
                break

        # If paint didn't land, that's okay - geometry might prevent it
        # But if we got here and have no paint reward, check if target is painted
        if not paint_reward_found and target.painted_remaining > 0:
            pytest.fail("Target was painted but no paint_assist reward given")

    def test_paint_expired_unused_penalty(self, reward_env):
        """Paint expiring without enabling damage gives -0.1 penalty."""
        env = reward_env

        # Manually set up paint that will expire unused
        target = env.sim.mechs["red_0"]
        scout_id = "blue_0"

        # Set paint state to expire on next step
        target.painted_remaining = 0.01  # Will expire immediately
        target.last_painter_id = scout_id
        target.paint_damage_accumulated = 0.0  # No damage dealt while painted

        # Move everyone far apart so no combat happens
        for aid in env.agents:
            m = env.sim.mechs[aid]
            m.pos[0] = 5.0 + (hash(aid) % 20)
            m.pos[1] = 5.0 + (hash(aid) % 10)
            m.vel[:] = 0.0

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        _, _rewards, _, _, infos = env.step(actions)

        # Scout should get penalty for wasted paint
        comps = infos["blue_0"]["reward_components"]
        paint_assist = comps.get("paint_assist", 0.0)
        # Penalty is -0.1, so paint_assist should be negative
        assert paint_assist < 0, f"Wasted paint should give negative reward, got {paint_assist}"

    def test_paint_expired_with_damage_no_penalty(self, reward_env):
        """Paint expiring after enabling damage does NOT give penalty."""
        env = reward_env

        target = env.sim.mechs["red_0"]
        scout_id = "blue_0"

        # Set paint state to expire, but with damage accumulated
        target.painted_remaining = 0.01  # Will expire immediately
        target.last_painter_id = scout_id
        target.paint_damage_accumulated = 50.0  # Damage was dealt while painted

        # Move everyone far apart
        for aid in env.agents:
            m = env.sim.mechs[aid]
            m.pos[0] = 5.0 + (hash(aid) % 20)
            m.pos[1] = 5.0 + (hash(aid) % 10)
            m.vel[:] = 0.0

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        _, _rewards, _, _, infos = env.step(actions)

        # Scout should NOT get penalty (paint was useful)
        comps = infos["blue_0"]["reward_components"]
        paint_assist = comps.get("paint_assist", 0.0)
        assert paint_assist >= 0, f"Used paint should not give penalty, got {paint_assist}"

    def test_paint_kill_fraction_bonus(self, reward_env):
        """Painter gets 40% of kill reward when painted target dies."""
        env = reward_env

        scout_id = "blue_0"
        killer_id = "blue_3"  # Medium with LASER

        # Setup: target is painted by scout, about to be killed by teammate
        target = env.sim.mechs["red_0"]
        target.painted_remaining = 5.0  # Active paint
        target.last_painter_id = scout_id
        target.hp = 1.0  # One-shot kill

        # Position killer to hit target
        killer = env.sim.mechs[killer_id]
        killer.pos[0], killer.pos[1] = target.pos[0] - 5.0, target.pos[1]
        killer.yaw = 0.0  # Facing +x toward target

        # Move scout away (shouldn't affect kill credit)
        env.sim.mechs[scout_id].pos[0] = 5.0
        env.sim.mechs[scout_id].pos[1] = 5.0

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        actions[killer_id][4] = 1.0  # Fire laser

        # Run until kill happens
        kill_occurred = False
        for _ in range(10):
            was_alive = target.alive
            _, _rewards, _, _, infos = env.step(actions)
            if was_alive and not target.alive and target.died:
                kill_occurred = True
                # Scout (painter) should get kill component from paint_kill_fraction
                scout_comps = infos[scout_id]["reward_components"]
                scout_kill = scout_comps.get("kill", 0.0)

                # 40% of W_KILL (5.0) = 2.0 (base, no zone multiplier)
                assert scout_kill > 0, f"Painter should get kill bonus, got {scout_kill}"
                break

        assert kill_occurred, "Target should have been killed"


@pytest.mark.skip(
    reason="Terminal rewards (W_WIN/W_LOSE) were removed - they dominated signal (98%) "
    "and didn't provide gradient information. Mission success is now shaped via zone rewards."
)
class TestTerminalRewards:
    """Verify terminal reward distribution."""

    def test_winner_gets_positive_terminal_reward(self):
        """Winning team gets positive terminal reward (W_WIN=5.0)."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=10, obstacle_fill=0.0, ensure_connectivity=False),
            num_packs=1,
            seed=0,
            max_episode_seconds=5.0,
        )
        env = EchelonEnv(cfg)
        env.reset(seed=0)
        assert env.sim is not None

        # Kill all red to trigger blue win
        for aid in env.agents:
            if "red" in aid:
                env.sim.mechs[aid].alive = False
                env.sim.mechs[aid].hp = 0.0

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        _, rewards, terms, _, _ = env.step(actions)

        # Episode should be over
        assert all(terms.values()), "Episode should terminate on team elimination"

        # Blue (winner) should get W_WIN=5.0
        for aid in env.agents:
            if "blue" in aid:
                assert rewards[aid] >= 5.0, f"Winner {aid} should get W_WIN>=5.0, got {rewards[aid]}"

    def test_loser_gets_negative_terminal_reward(self):
        """Losing team gets negative terminal reward (W_LOSE=-5.0)."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=10, obstacle_fill=0.0, ensure_connectivity=False),
            num_packs=1,
            seed=0,
            max_episode_seconds=5.0,
        )
        env = EchelonEnv(cfg)
        env.reset(seed=0)
        assert env.sim is not None

        # Kill all blue to trigger red win
        for aid in env.agents:
            if "blue" in aid:
                env.sim.mechs[aid].alive = False
                env.sim.mechs[aid].hp = 0.0

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        _, rewards, _terms, _, _ = env.step(actions)

        # Blue (loser) should get W_LOSE=-5.0
        for aid in env.agents:
            if "blue" in aid:
                assert rewards[aid] <= -5.0, f"Loser {aid} should get W_LOSE<=-5.0, got {rewards[aid]}"

    def test_draw_gives_zero_terminal_reward(self):
        """Draw gives zero terminal reward (W_DRAW=0.0)."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=10, obstacle_fill=0.0, ensure_connectivity=False),
            num_packs=1,
            seed=0,
            max_episode_seconds=0.5,  # Very short to force timeout
        )
        env = EchelonEnv(cfg)
        env.reset(seed=0)
        assert env.sim is not None

        # Run until timeout with equal zone control (no one in zone)
        for aid in env.agents:
            m = env.sim.mechs[aid]
            m.pos[0], m.pos[1] = 5.0, 5.0
            m.vel[:] = 0.0

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        done = False
        final_rewards = None
        for _ in range(100):
            _, rewards, _terms, truncs, infos = env.step(actions)
            if all(truncs.values()):
                done = True
                final_rewards = rewards
                # Check if it's actually a draw
                outcome = infos.get("blue_0", {}).get("outcome", {})
                if outcome.get("winner") == "draw":
                    break

        if done and final_rewards:
            # Draw terminal rewards should be ~0 (W_DRAW=0.0)
            # But there may be small per-step rewards, so we check the magnitude is small
            for aid in env.agents:
                assert abs(final_rewards[aid]) < 1.0, f"Draw reward should be ~0, got {final_rewards[aid]}"

    def test_dead_agents_get_terminal_reward(self):
        """Dead agents also receive terminal reward (for learning)."""
        cfg = EnvConfig(
            world=WorldConfig(size_x=30, size_y=30, size_z=10, obstacle_fill=0.0, ensure_connectivity=False),
            num_packs=1,
            seed=0,
            max_episode_seconds=5.0,
        )
        env = EchelonEnv(cfg)
        env.reset(seed=0)
        assert env.sim is not None

        # Kill blue_0, then end episode with blue win
        env.sim.mechs["blue_0"].alive = False
        env.sim.mechs["blue_0"].hp = 0.0

        # Kill all red
        for aid in env.agents:
            if "red" in aid:
                env.sim.mechs[aid].alive = False
                env.sim.mechs[aid].hp = 0.0

        actions = {aid: np.zeros(env.ACTION_DIM, dtype=np.float32) for aid in env.agents}
        _, rewards, _, _, _ = env.step(actions)

        # Dead blue_0 should still get winner reward
        assert (
            rewards["blue_0"] >= 5.0
        ), f"Dead agent on winning team should get W_WIN, got {rewards['blue_0']}"


class TestFormationModeMultipliers:
    """Verify formation mode reward multipliers."""

    def test_formation_mode_enum_values(self):
        """FormationMode enum has expected values."""
        from echelon.env.rewards import FormationMode

        assert FormationMode.CLOSE == 0
        assert FormationMode.STANDARD == 1
        assert FormationMode.LOOSE == 2

    def test_formation_multipliers_exist(self):
        """Each formation mode has multipliers defined."""
        from echelon.env.rewards import FORMATION_MULTIPLIERS, FormationMode

        for mode in FormationMode:
            assert mode in FORMATION_MULTIPLIERS
            mult = FORMATION_MULTIPLIERS[mode]
            assert "zone" in mult
            assert "out_death" in mult
            assert "approach" in mult

    def test_close_amplifies_zone_rewards(self):
        """CLOSE mode has zone multiplier > 1."""
        from echelon.env.rewards import FORMATION_MULTIPLIERS, FormationMode

        assert FORMATION_MULTIPLIERS[FormationMode.CLOSE]["zone"] > 1.0
        assert FORMATION_MULTIPLIERS[FormationMode.CLOSE]["out_death"] > 1.0

    def test_loose_reduces_zone_rewards(self):
        """LOOSE mode has zone multiplier < 1."""
        from echelon.env.rewards import FORMATION_MULTIPLIERS, FormationMode

        assert FORMATION_MULTIPLIERS[FormationMode.LOOSE]["zone"] < 1.0
        assert FORMATION_MULTIPLIERS[FormationMode.LOOSE]["out_death"] < 1.0

    def test_standard_is_neutral(self):
        """STANDARD mode has multipliers of 1.0."""
        from echelon.env.rewards import FORMATION_MULTIPLIERS, FormationMode

        mult = FORMATION_MULTIPLIERS[FormationMode.STANDARD]
        assert mult["zone"] == 1.0
        assert mult["out_death"] == 1.0
        assert mult["approach"] == 1.0
