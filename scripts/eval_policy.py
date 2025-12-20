# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig
from echelon.agents.heuristic import HeuristicPolicy
from echelon.rl.model import ActorCriticLSTM


def stack_obs(obs: dict[str, np.ndarray], ids: list[str]) -> np.ndarray:
    return np.stack([obs[aid] for aid in ids], axis=0)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--out", type=str, default="runs/eval")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device={device} cuda_available={torch.cuda.is_available()} torch={torch.__version__}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    env_cfg_dict = dict(ckpt["env_cfg"])
    world = WorldConfig(**env_cfg_dict["world"])
    env_cfg = EnvConfig(
        **{
            **{k: v for k, v in env_cfg_dict.items() if k != "world"},
            "world": world,
            "record_replay": bool(args.record),
            "seed": int(args.seed),
        }
    )

    env = EchelonEnv(env_cfg)
    heuristic = HeuristicPolicy()

    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(next(iter(obs.values())).shape[0])
    action_dim = env.ACTION_DIM
    model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = Path(args.out)
    if args.record:
        out_dir.mkdir(parents=True, exist_ok=True)

    blue_ids = env.blue_ids
    red_ids = env.red_ids

    wins = {"blue": 0, "red": 0, "draw": 0}
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        lstm_state = model.initial_state(batch_size=len(blue_ids), device=device)
        done = torch.zeros(len(blue_ids), device=device)

        while True:
            obs_b = torch.from_numpy(stack_obs(obs, blue_ids)).to(device)
            action_b, _, _, _, lstm_state = model.get_action_and_value(obs_b, lstm_state, done)
            action_np = action_b.detach().cpu().numpy()

            actions = {bid: action_np[i] for i, bid in enumerate(blue_ids)}
            for rid in red_ids:
                actions[rid] = heuristic.act(env, rid)

            obs, rewards, terminations, truncations, infos = env.step(actions)
            done = torch.tensor(
                [terminations[bid] or truncations[bid] for bid in blue_ids], dtype=torch.float32, device=device
            )

            blue_alive = env.sim.team_alive("blue")
            red_alive = env.sim.team_alive("red")
            ep_over = any(truncations.values()) or (not blue_alive) or (not red_alive)
            if ep_over:
                outcome = env.last_outcome or {"winner": "draw"}
                winner = outcome.get("winner", "draw")
                wins[winner] += 1
                break

        if args.record:
            replay = env.get_replay()
            if replay is not None:
                p = out_dir / f"replay_ep_{ep:04d}.json"
                p.write_text(json.dumps(replay), encoding="utf-8")

        print(f"episode {ep}: wins={wins}")

    print("final:", wins)


if __name__ == "__main__":
    main()
