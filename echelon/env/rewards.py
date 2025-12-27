"""Reward computation for EchelonEnv.

This module encapsulates all reward logic, making it easier to:
1. Understand reward shaping decisions
2. Integrate curriculum-based reward schedules
3. Add new reward components without bloating env.py

Design:
- RewardWeights: Configurable weights for each reward component
- StepContext: All per-step state needed for reward computation
- RewardComputer: Stateless computation of rewards from context
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Configurable weights for reward components.

    These weights are designed to be modified by a curriculum system.
    Default values are the current production settings (2025-12-27 rebalance v3).

    Behavioral objective: "Go to zone. Stay in zone. Fight in zone."

    Key design (2025-12-27 v3 - zone-centric combat):
    - approach: 0.1 (TINY gravity well - guides to zone, penalizes leaving)
    - arrival_bonus: 1.0 (single ping = 10x max approach budget)
    - in_zone_damage_mult: 10x (fight hard in zone!)
    - in_zone_death_mult: 0.05x (dying in zone is FREE)
    - out_zone_death_mult: 2x (dying outside is expensive)

    Reward math:
    - Max approach: ~0.1 per episode (just beats "do nothing")
    - Arrival ping: 1.0 (10x approach, signals "you made it")
    - 1 HP damage in zone: 1.0 pts (10x approach)
    - Kill in zone: 50 pts (500x approach)
    - Die in zone: -0.1 (negligible)
    - Die outside: -4.0 (40x approach penalty)
    - Leaving zone: negative approach reward (gravity well)
    """

    # Zone-based rewards (PRIMARY OBJECTIVE)
    # Reduced from 5.0 - still highest priority but agents can briefly leave for easy kills
    zone_tick: float = 2.0  # Being IN zone and controlling it
    arrival_bonus: float = 1.0  # Single "ping" for reaching zone (dwarfs approach)

    # Approach shaping (TINY - just beats "do nothing")
    # Max approach ≈ 0.1 over entire episode
    # Arrival ping (1.0) = 10x max approach - getting there is just the start
    approach: float = 0.1  # Moving toward zone (PBRS shaping, stops at zone)

    # Combat rewards - HIGH IN ZONE
    # Base damage = 0.1/HP, in zone = 1.0/HP (10x mult)
    # Kill bonus high enough to incentivize focus fire over damage spreading
    damage: float = 0.1  # Per point of damage dealt
    kill: float = 20.0  # Per kill (in zone = 200 pts) - FINISH YOUR TARGETS
    assist: float = 5.0  # Per assist (in zone = 50 pts)
    death: float = -2.0  # Death penalty (in zone = -0.1, outside = -2.0)

    # Zone mechanics
    # Increased from 0.25: contested zone should still give meaningful reward
    contested_floor: float = 0.5  # Min reward when zone is contested

    # Shaping parameters
    shaping_gamma: float = 0.99  # Discount for PBRS compliance

    # Team reward mixing (for coordination)
    # Reduced from 1.0: 30% team reward helps credit assignment
    team_reward_alpha: float = 0.7  # 1.0 = individual, 0.0 = team average

    # Zone-centric combat multipliers (2025-12-27 redesign)
    # Philosophy: Fight for the zone, don't die outside it
    in_zone_damage_mult: float = 10.0  # 10x damage/kill/assist IN zone - fight hard!
    in_zone_death_mult: float = 0.05  # 0.05x death penalty IN zone - deaths are FREE
    out_zone_death_mult: float = 1.0  # 1x death penalty OUTSIDE zone - base penalty

    # Paint/support bonuses - scouts who paint targets help the team
    paint_assist_bonus: float = 2.0  # Bonus when teammate uses your paint lock


@dataclass
class StepContext:
    """All per-step state needed for reward computation.

    This bundles the various dicts and values computed during step()
    into a single object for cleaner function signatures.
    """

    # Agent lists
    agents: list[str]
    blue_ids: list[str]
    red_ids: list[str]

    # Mech state accessor (team lookup)
    mech_teams: dict[str, str]  # aid -> "blue" | "red"
    mech_alive: dict[str, bool]  # aid -> alive
    mech_died: dict[str, bool]  # aid -> died this step

    # Zone geometry
    zone_center: tuple[float, float]
    zone_radius: float
    max_xy: float  # max(world.size_x, world.size_y)

    # Zone state
    in_zone_by_agent: dict[str, bool]
    in_zone_tonnage: dict[str, float]  # "blue" -> tonnage, "red" -> tonnage
    blue_tick: float  # Zone control tick for blue
    red_tick: float  # Zone control tick for red

    # Distance tracking (for approach shaping)
    dist_to_zone_before: dict[str, float]
    dist_to_zone_after: dict[str, float]

    # Combat events this step
    step_damage_dealt: dict[str, float]
    step_kills: dict[str, int]
    step_assists: dict[str, int]
    step_deaths: dict[str, bool]

    # Paint lock usage tracking: painter_id -> count of times paint lock was used this step
    # When a teammate fires a guided weapon using a paint lock, the painter gets credit
    step_paint_assists: dict[str, int] | None = None

    # First zone entry tracking: aid -> True if this is the first time entering zone this episode
    # Used for arrival_bonus reward (one-time bonus for reaching objective)
    first_zone_entry_this_step: dict[str, bool] | None = None


@dataclass
class RewardComponents:
    """Breakdown of reward into components for debugging/analysis."""

    approach: float = 0.0
    zone: float = 0.0
    arrival: float = 0.0  # One-time bonus for first zone entry
    damage: float = 0.0
    kill: float = 0.0
    assist: float = 0.0
    death: float = 0.0
    paint_assist: float = 0.0  # Reward for providing paint locks that teammates use
    terminal: float = 0.0  # Reserved for future use

    def total(self) -> float:
        return (
            self.approach
            + self.zone
            + self.arrival
            + self.damage
            + self.kill
            + self.assist
            + self.death
            + self.paint_assist
            + self.terminal
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "approach": self.approach,
            "zone": self.zone,
            "arrival": self.arrival,
            "damage": self.damage,
            "kill": self.kill,
            "assist": self.assist,
            "death": self.death,
            "paint_assist": self.paint_assist,
            "terminal": self.terminal,
        }


class RewardComputer:
    """Stateless reward computation from step context.

    This class encapsulates all reward logic. It can be subclassed
    or replaced by a curriculum-aware version.
    """

    def __init__(self, weights: RewardWeights | None = None):
        self.weights = weights or RewardWeights()

    def compute(self, ctx: StepContext) -> tuple[dict[str, float], dict[str, RewardComponents]]:
        """Compute rewards for all agents.

        Args:
            ctx: Step context with all required state

        Returns:
            (rewards, components) where:
            - rewards: dict mapping agent_id -> total reward
            - components: dict mapping agent_id -> RewardComponents breakdown
        """
        w = self.weights
        rewards: dict[str, float] = {}
        components: dict[str, RewardComponents] = {}

        # Count teammates in zone for scaled breadcrumb decay
        teammates_in_zone: dict[str, int] = {"blue": 0, "red": 0}
        for aid, in_zone in ctx.in_zone_by_agent.items():
            if in_zone:
                team = ctx.mech_teams.get(aid)
                if team:
                    teammates_in_zone[team] += 1

        for aid in ctx.agents:
            team = ctx.mech_teams[aid]
            alive = ctx.mech_alive.get(aid, False)
            died = ctx.mech_died.get(aid, False)

            # Dead agents get 0 after the death step
            if not (alive or died):
                rewards[aid] = 0.0
                components[aid] = RewardComponents()
                continue

            comp = self._compute_agent_reward(aid, team, ctx, teammates_in_zone)
            rewards[aid] = comp.total()
            components[aid] = comp

        # Team reward mixing
        if w.team_reward_alpha < 1.0:
            rewards = self._apply_team_mixing(rewards, ctx)

        return rewards, components

    def _compute_agent_reward(
        self,
        aid: str,
        team: str,
        ctx: StepContext,
        teammates_in_zone: dict[str, int],
    ) -> RewardComponents:
        """Compute reward components for a single agent."""
        w = self.weights
        comp = RewardComponents()

        # (1) Approach shaping: PBRS-compliant distance-based reward
        # r = gamma * phi(s') - phi(s) where phi(s) = -distance / max_xy
        # ALWAYS applies - creates gravity well effect:
        # - Moving toward zone = positive reward
        # - Leaving zone = negative reward (penalty for backing out)
        d0 = ctx.dist_to_zone_before.get(aid)
        d1 = ctx.dist_to_zone_after.get(aid)
        if d0 is not None and d1 is not None and ctx.max_xy > 0.0:
            phi0 = -d0 / ctx.max_xy
            phi1 = -d1 / ctx.max_xy
            comp.approach = w.approach * (w.shaping_gamma * phi1 - phi0)

        # (1.5) Arrival bonus: one-time reward for first zone entry this episode
        # This breaks the "approach forever" trap by giving a discrete signal for
        # achieving the intermediate goal of reaching the zone
        if ctx.first_zone_entry_this_step is not None and ctx.first_zone_entry_this_step.get(aid, False):
            comp.arrival = w.arrival_bonus

        # (2) Zone control reward: base divided by sqrt(teammates in zone)
        # sqrt(n) flattens the distribution - first agent doesn't get massive jackpot
        # With n=1: full reward, n=4: 50% each, n=9: 33% each
        # This reduces "race to zone" incentive while still rewarding presence
        team_tick = ctx.blue_tick if team == "blue" else ctx.red_tick
        if ctx.in_zone_by_agent.get(aid, False):
            n_in_zone = max(1, teammates_in_zone.get(team, 1))
            # sqrt(n) instead of n for fairer distribution
            comp.zone = w.zone_tick * team_tick / math.sqrt(n_in_zone)

        # (3) Combat rewards: ZONE-CENTRIC
        # In zone: high damage rewards, low death penalty (fight for the zone!)
        # Out of zone: low damage rewards, high death penalty (don't waste lives)
        agent_in_zone = ctx.in_zone_by_agent.get(aid, False)

        if agent_in_zone:
            # Fight for the zone - damage is valuable, death is cheap
            damage_mult = w.in_zone_damage_mult  # 5x
            death_mult = w.in_zone_death_mult  # 0.1x (practically free)
        else:
            # Don't die pointlessly outside the zone
            damage_mult = 1.0  # base damage reward
            death_mult = w.out_zone_death_mult  # 3x (expensive)

        comp.damage = w.damage * damage_mult * ctx.step_damage_dealt.get(aid, 0.0)
        comp.kill = w.kill * damage_mult * ctx.step_kills.get(aid, 0)
        comp.assist = w.assist * damage_mult * ctx.step_assists.get(aid, 0)

        # (4) Death penalty: zone-dependent
        if ctx.step_deaths.get(aid, False):
            comp.death = w.death * death_mult

        # (5) Paint assist bonus: scouts who paint targets get reward when teammates use the lock
        # This incentivizes scouts to perform their support role rather than rushing to zone
        if ctx.step_paint_assists is not None:
            paint_count = ctx.step_paint_assists.get(aid, 0)
            if paint_count > 0:
                comp.paint_assist = w.paint_assist_bonus * paint_count

        return comp

    def _apply_team_mixing(self, rewards: dict[str, float], ctx: StepContext) -> dict[str, float]:
        """Apply team reward mixing (alpha blending).

        alpha=1.0: fully individual
        alpha=0.0: fully team average
        """
        alpha = self.weights.team_reward_alpha

        # Compute team averages
        team_sums: dict[str, float] = {"blue": 0.0, "red": 0.0}
        team_counts: dict[str, int] = {"blue": 0, "red": 0}

        for aid in ctx.agents:
            team = ctx.mech_teams[aid]
            alive = ctx.mech_alive.get(aid, False)
            died = ctx.mech_died.get(aid, False)
            if alive or died:
                team_sums[team] += rewards[aid]
                team_counts[team] += 1

        team_avg: dict[str, float] = {t: (team_sums[t] / max(1, team_counts[t])) for t in ["blue", "red"]}

        # Blend
        blended = {}
        for aid in ctx.agents:
            team = ctx.mech_teams[aid]
            alive = ctx.mech_alive.get(aid, False)
            died = ctx.mech_died.get(aid, False)
            if alive or died:
                blended[aid] = alpha * rewards[aid] + (1.0 - alpha) * team_avg[team]
            else:
                blended[aid] = rewards[aid]

        return blended


def compute_zone_ticks(
    in_zone_tonnage: dict[str, float],
    contested_floor: float = 0.25,
) -> tuple[float, float]:
    """Compute zone control tick values for each team.

    Args:
        in_zone_tonnage: {"blue": tonnage, "red": tonnage}
        contested_floor: Minimum reward when zone is contested

    Returns:
        (blue_tick, red_tick) where each is in [0, 1]
    """
    total_tonnage = in_zone_tonnage["blue"] + in_zone_tonnage["red"]

    if total_tonnage <= 0:
        return 0.0, 0.0

    blue_margin = (in_zone_tonnage["blue"] - in_zone_tonnage["red"]) / total_tonnage

    if in_zone_tonnage["blue"] > 0 and in_zone_tonnage["red"] > 0:
        # Contested: floor + bonus for winning
        blue_tick = contested_floor + max(0.0, blue_margin) * (1.0 - contested_floor)
        red_tick = contested_floor + max(0.0, -blue_margin) * (1.0 - contested_floor)
    else:
        # Uncontested: winner gets full reward
        blue_tick = 1.0 if in_zone_tonnage["blue"] > 0 else 0.0
        red_tick = 1.0 if in_zone_tonnage["red"] > 0 else 0.0

    return blue_tick, red_tick
