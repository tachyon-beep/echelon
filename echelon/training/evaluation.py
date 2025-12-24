"""Policy evaluation against baseline opponents."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from echelon import EchelonEnv, EnvConfig
from echelon.agents.heuristic import HeuristicPolicy

if TYPE_CHECKING:
    from echelon.rl.model import ActorCriticLSTM


@dataclass
class EvalStats:
    """Evaluation results."""

    win_rate: float
    mean_hp_margin: float
    mean_episode_length: float
    episodes: int


def _stack_obs(obs: dict[str, np.ndarray], ids: list[str]) -> np.ndarray:
    """Stack observations for multiple agents."""
    return np.stack([obs[aid] for aid in ids], axis=0)


@torch.no_grad()
def evaluate_vs_heuristic(
    model: ActorCriticLSTM,
    env_cfg: EnvConfig,
    episodes: int,
    seeds: list[int],
    device: torch.device,
    save_replay: bool = False,
) -> tuple[EvalStats, dict | None]:
    """Evaluate policy against HeuristicPolicy baseline.

    Args:
        model: The policy to evaluate (plays as blue team)
        env_cfg: Environment configuration
        episodes: Number of evaluation episodes
        seeds: List of seeds for episode initialization (must match episodes count)
        device: Device for model inference
        save_replay: If True, save replay from first episode

    Returns:
        stats: Win rate, mean HP margin, episode lengths
        replay: Optional replay dict from first episode (if save_replay=True)
    """
    if len(seeds) != episodes:
        raise ValueError(f"seeds length ({len(seeds)}) must match episodes ({episodes})")

    # Set record_replay if requested
    env_cfg = replace(env_cfg, record_replay=save_replay)
    env = EchelonEnv(env_cfg)
    heuristic = HeuristicPolicy()

    blue_ids = env.blue_ids
    red_ids = env.red_ids

    wins = {"blue": 0, "red": 0, "draw": 0}
    hp_margins: list[float] = []
    episode_lengths: list[int] = []
    last_replay = None

    for ep, ep_seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(ep_seed))
        lstm_state = model.initial_state(batch_size=len(blue_ids), device=device)
        done = torch.zeros(len(blue_ids), device=device)
        steps = 0

        while True:
            obs_b = torch.from_numpy(_stack_obs(obs, blue_ids)).to(device)
            action_b, _, _, _, lstm_state = model.get_action_and_value(obs_b, lstm_state, done)
            action_np = action_b.cpu().numpy()

            actions = {bid: action_np[i] for i, bid in enumerate(blue_ids)}
            for rid in red_ids:
                actions[rid] = heuristic.act(env, rid)

            obs, _rewards, terminations, truncations, _infos = env.step(actions)
            steps += 1

            # env.sim is guaranteed to be non-None after step
            assert env.sim is not None
            blue_alive = env.sim.team_alive("blue")
            red_alive = env.sim.team_alive("red")
            done = torch.tensor(
                [terminations[bid] or truncations[bid] for bid in blue_ids],
                dtype=torch.float32,
                device=device,
            )

            if any(truncations.values()) or (not blue_alive) or (not red_alive):
                outcome = env.last_outcome or {"winner": "draw", "hp": env.team_hp()}
                winner = outcome.get("winner", "draw")
                wins[winner] += 1
                hp = outcome.get("hp") or env.team_hp()
                hp_margins.append(float(hp.get("blue", 0.0) - hp.get("red", 0.0)))
                episode_lengths.append(steps)

                if save_replay and ep == 0:
                    last_replay = env.get_replay()
                break

    stats = EvalStats(
        win_rate=float(wins["blue"] / max(1, episodes)),
        mean_hp_margin=float(np.mean(hp_margins)) if hp_margins else 0.0,
        mean_episode_length=float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        episodes=episodes,
    )

    return stats, last_replay
