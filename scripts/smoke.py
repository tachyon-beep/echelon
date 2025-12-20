# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echelon import EchelonEnv, EnvConfig, WorldConfig
from echelon.agents.heuristic import HeuristicPolicy


def run_episode(env: EchelonEnv, policy: HeuristicPolicy, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)

    steps = 0
    while True:
        actions = {aid: policy.act(env, aid) for aid in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        steps += 1

        if any(truncations.values()) or (not env.sim.team_alive("blue")) or (not env.sim.team_alive("red")):
            break

    outcome = env.last_outcome or {"winner": "draw"}
    winner = outcome.get("winner", "draw")

    return {"seed": seed, "steps": steps, "winner": winner}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-per-team", type=int, default=2)
    parser.add_argument("--size", type=int, default=20, help="World size (x=y=z)")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "partial"])
    parser.add_argument("--record", action="store_true", help="Record replay frames to JSON")
    parser.add_argument("--out", type=str, default="runs/smoke", help="Output directory for replays")
    args = parser.parse_args()

    cfg = EnvConfig(
        world=WorldConfig(size_x=args.size, size_y=args.size, size_z=args.size),
        num_per_team=args.num_per_team,
        observation_mode=args.mode,
        record_replay=args.record,
        seed=args.seed,
        max_episode_seconds=30.0,
    )
    env = EchelonEnv(cfg)
    policy = HeuristicPolicy()

    out_dir = Path(args.out)
    if args.record:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        result = run_episode(env, policy, seed=seed)
        results.append(result)
        print(f"episode {ep}: {result}")

        replay = env.get_replay()
        if args.record and replay is not None:
            p = out_dir / f"replay_seed_{seed}.json"
            p.write_text(json.dumps(replay), encoding="utf-8")

    # Tiny summary
    wins = {"blue": 0, "red": 0, "draw": 0}
    for r in results:
        wins[r["winner"]] += 1
    print("summary:", wins)


if __name__ == "__main__":
    main()
