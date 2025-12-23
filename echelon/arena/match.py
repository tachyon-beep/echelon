from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..config import EnvConfig, WorldConfig
from ..env.env import EchelonEnv
from ..rl.model import ActorCriticLSTM

if TYPE_CHECKING:
    from pathlib import Path


def _stack_obs(obs: dict[str, np.ndarray], ids: list[str]) -> np.ndarray:
    return np.stack([obs[aid] for aid in ids], axis=0)


@dataclass(frozen=True)
class LoadedPolicy:
    ckpt_path: Path
    env_cfg: EnvConfig
    model: ActorCriticLSTM


def _env_cfg_from_checkpoint(ckpt: dict[str, Any]) -> EnvConfig:
    env_cfg_dict = dict(ckpt["env_cfg"])
    world = WorldConfig(**env_cfg_dict["world"])
    return EnvConfig(
        **{
            **{k: v for k, v in env_cfg_dict.items() if k != "world"},
            "world": world,
        }
    )


def load_policy(ckpt_path: Path, *, device: torch.device) -> LoadedPolicy:
    ckpt_path = ckpt_path.resolve()
    ckpt = torch.load(
        ckpt_path, map_location=device, weights_only=False
    )  # TODO: migrate to weights_only=True after adding safe globals
    env_cfg = _env_cfg_from_checkpoint(ckpt)

    # Use cached model dimensions if available (HIGH-9), otherwise fall back to env creation
    if "obs_dim" in ckpt and "action_dim" in ckpt:
        obs_dim = int(ckpt["obs_dim"])
        action_dim = int(ckpt["action_dim"])
    else:
        # Legacy checkpoint without cached dimensions - build env to infer
        env = EchelonEnv(env_cfg)
        obs, _ = env.reset(seed=int(env_cfg.seed or 0))
        obs_dim = int(next(iter(obs.values())).shape[0])
        action_dim = int(env.ACTION_DIM)

    model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return LoadedPolicy(ckpt_path=ckpt_path, env_cfg=env_cfg, model=model)


@dataclass(frozen=True)
class MatchOutcome:
    winner: str  # "blue" | "red" | "draw"
    hp: dict[str, float]
    seed: int


@torch.no_grad()
def play_match(
    *,
    env_cfg: EnvConfig,
    blue_policy: ActorCriticLSTM,
    red_policy: ActorCriticLSTM,
    seed: int,
    device: torch.device,
    max_steps: int | None = None,
) -> MatchOutcome:
    env = EchelonEnv(env_cfg)
    obs, _ = env.reset(seed=seed)

    blue_ids = env.blue_ids
    red_ids = env.red_ids

    blue_state = blue_policy.initial_state(batch_size=len(blue_ids), device=device)
    red_state = red_policy.initial_state(batch_size=len(red_ids), device=device)
    blue_done = torch.ones(len(blue_ids), device=device)
    red_done = torch.ones(len(red_ids), device=device)

    steps = 0
    while True:
        obs_b = torch.from_numpy(_stack_obs(obs, blue_ids)).to(device)
        act_b, _, _, _, blue_state = blue_policy.get_action_and_value(obs_b, blue_state, blue_done)
        act_b_np = act_b.cpu().numpy()

        obs_r = torch.from_numpy(_stack_obs(obs, red_ids)).to(device)
        act_r, _, _, _, red_state = red_policy.get_action_and_value(obs_r, red_state, red_done)
        act_r_np = act_r.cpu().numpy()

        actions = {bid: act_b_np[i] for i, bid in enumerate(blue_ids)}
        actions.update({rid: act_r_np[i] for i, rid in enumerate(red_ids)})

        obs, _rewards, terminations, truncations, _infos = env.step(actions)

        blue_done = torch.tensor(
            [terminations[bid] or truncations[bid] for bid in blue_ids], device=device, dtype=torch.float32
        )
        red_done = torch.tensor(
            [terminations[rid] or truncations[rid] for rid in red_ids], device=device, dtype=torch.float32
        )

        steps += 1
        if max_steps is not None and steps >= int(max_steps):
            break

        if env.sim is None:
            break
        blue_alive = env.sim.team_alive("blue")
        red_alive = env.sim.team_alive("red")
        if any(truncations.values()) or (not blue_alive) or (not red_alive):
            break

    outcome = env.last_outcome or {"winner": "draw", "hp": env.team_hp()}
    winner = str(outcome.get("winner", "draw"))
    hp = outcome.get("hp") or env.team_hp()
    return MatchOutcome(winner=winner, hp={k: float(v) for k, v in hp.items()}, seed=int(seed))
