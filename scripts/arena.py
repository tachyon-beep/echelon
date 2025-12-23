# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import TYPE_CHECKING

from echelon.arena.glicko2 import GameResult
from echelon.arena.league import League
from echelon.arena.match import LoadedPolicy, load_policy, play_match

if TYPE_CHECKING:
    from echelon.config import EnvConfig


class LRUPolicyCache:
    """LRU cache for opponent policies to prevent OOM (HIGH-10)."""

    def __init__(self, max_size: int = 10):
        self.max_size = max(1, max_size)
        self._cache: dict[str, LoadedPolicy] = {}
        self._order: list[str] = []

    def get(self, key: str) -> LoadedPolicy | None:
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, policy: LoadedPolicy) -> None:
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict least recently used with proper GPU memory cleanup
            evict_key = self._order.pop(0)
            evicted = self._cache.pop(evict_key)
            evicted.model.cpu()  # Move to CPU to free GPU memory
            del evicted
        self._cache[key] = policy
        self._order.append(key)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def env_signature(cfg: EnvConfig) -> dict:
    d = asdict(cfg)
    # League signature should be invariant to run-time flags and seeds.
    d.pop("seed", None)
    d.pop("record_replay", None)
    return d


def cmd_init(args: argparse.Namespace) -> None:
    league_path = Path(args.league)
    if league_path.exists() and not args.force:
        raise SystemExit(f"League already exists: {league_path} (use --force to overwrite)")
    league = League()
    league.save(league_path)
    print(f"ok: wrote {league_path}")


def cmd_add(args: argparse.Namespace) -> None:
    league_path = Path(args.league)
    league = League.load(league_path) if league_path.exists() else League()

    device = resolve_device(args.device)
    policy = load_policy(Path(args.ckpt), device=device)
    sig = env_signature(policy.env_cfg)
    if league.env_signature is None:
        league.env_signature = sig
    elif league.env_signature != sig:
        raise SystemExit("checkpoint env_cfg does not match league env_signature")

    entry = league.upsert_checkpoint(Path(args.ckpt), kind=str(args.kind))
    if args.kind == "commander" and entry.commander_name is None and args.name:
        entry.commander_name = str(args.name)

    league.save(league_path)
    print(
        f"ok: added {entry.entry_id} kind={entry.kind} rating={entry.rating.rating:.1f}±{entry.rating.rd:.1f}"
    )


def cmd_list(args: argparse.Namespace) -> None:
    league_path = Path(args.league)
    if not league_path.exists():
        raise SystemExit(f"missing league: {league_path}")
    league = League.load(league_path)
    rows = list(league.entries.values())
    rows.sort(key=lambda e: float(e.rating.rating), reverse=True)

    out = []
    for e in rows:
        out.append(
            {
                "id": e.entry_id,
                "kind": e.kind,
                "name": e.commander_name,
                "rating": round(float(e.rating.rating), 1),
                "rd": round(float(e.rating.rd), 1),
                "vol": round(float(e.rating.vol), 4),
                "games": int(e.games),
                "ckpt": e.ckpt_path,
            }
        )
    print(json.dumps(out, indent=2))


def cmd_eval_candidate(args: argparse.Namespace) -> None:
    league_path = Path(args.league)
    if not league_path.exists():
        raise SystemExit(f"missing league: {league_path}")
    league = League.load(league_path)

    device = resolve_device(args.device)
    rng = random.Random(int(args.seed))

    candidate_path = Path(args.ckpt)
    candidate_policy = load_policy(candidate_path, device=device)
    sig = env_signature(candidate_policy.env_cfg)
    if league.env_signature is None:
        league.env_signature = sig
    elif league.env_signature != sig:
        raise SystemExit("candidate env_cfg does not match league env_signature")

    candidate_entry = league.upsert_checkpoint(candidate_path, kind="candidate")

    top_k = int(args.top_k)
    cand_k = int(args.candidate_k)
    pool = league.top_commanders(top_k) + league.recent_candidates(
        cand_k, exclude_id=candidate_entry.entry_id
    )
    pool = [e for e in pool if e.entry_id != candidate_entry.entry_id]
    if not pool:
        raise SystemExit("no opponents available (need commanders/candidates in league)")

    # LRU cache for opponent policies to prevent OOM (HIGH-10)
    opponent_cache = LRUPolicyCache(max_size=20)

    def get_opponent(entry_id: str) -> LoadedPolicy:
        cached = opponent_cache.get(entry_id)
        if cached is not None:
            return cached
        entry = league.entries[entry_id]
        policy = load_policy(Path(entry.ckpt_path), device=device)
        opponent_cache.put(entry_id, policy)
        return policy

    matches = int(args.matches)
    results: dict[str, list[GameResult]] = {}
    game_log: list[dict] = []

    # Use a deterministic seed stream so reruns are reproducible.
    seed0 = int(args.seed) * 1_000_000 + 12345
    for i in range(matches):
        opp_entry = rng.choice(pool)
        opp_policy = get_opponent(opp_entry.entry_id)

        if env_signature(opp_policy.env_cfg) != sig:
            raise SystemExit(f"opponent env_cfg mismatch: {opp_entry.entry_id}")

        candidate_as_blue = (i % 2) == 0
        env_cfg = replace(candidate_policy.env_cfg, record_replay=False)
        match_seed = seed0 + i * 9973

        if candidate_as_blue:
            out = play_match(
                env_cfg=env_cfg,
                blue_policy=candidate_policy.model,
                red_policy=opp_policy.model,
                seed=match_seed,
                device=device,
            )
            cand_team = "blue"
        else:
            out = play_match(
                env_cfg=env_cfg,
                blue_policy=opp_policy.model,
                red_policy=candidate_policy.model,
                seed=match_seed,
                device=device,
            )
            cand_team = "red"

        if out.winner == "draw":
            cand_score = 0.5
        else:
            cand_score = 1.0 if out.winner == cand_team else 0.0
        opp_score = 1.0 - cand_score if cand_score != 0.5 else 0.5

        results.setdefault(candidate_entry.entry_id, []).append(
            GameResult(opponent=opp_entry.rating, score=cand_score)
        )
        results.setdefault(opp_entry.entry_id, []).append(
            GameResult(opponent=candidate_entry.rating, score=opp_score)
        )

        game_log.append(
            {
                "i": i,
                "seed": int(match_seed),
                "candidate_as": cand_team,
                "opponent": opp_entry.entry_id,
                "opponent_name": opp_entry.commander_name,
                "winner": out.winner,
                "candidate_score": cand_score,
                "hp": out.hp,
            }
        )
        print(
            f"match {i + 1}/{matches}: opp={opp_entry.entry_id} cand_as={cand_team} winner={out.winner} "
            f"score={cand_score}"
        )

    league.apply_rating_period(results)
    promoted = league.promote_if_topk(candidate_entry.entry_id, top_k=top_k)
    league.save(league_path)

    cand_post = league.entries[candidate_entry.entry_id]
    print(
        f"candidate rating: {cand_post.rating.rating:.1f}±{cand_post.rating.rd:.1f} vol={cand_post.rating.vol:.4f} "
        f"games={cand_post.games} promoted={promoted} name={cand_post.commander_name}"
    )

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({"games": game_log}, indent=2), encoding="utf-8")
        print(f"wrote {log_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--league", type=str, default="runs/arena/league.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_add = sub.add_parser("add")
    p_add.add_argument("ckpt", type=str)
    p_add.add_argument("--kind", choices=["candidate", "commander"], default="candidate")
    p_add.add_argument("--name", type=str, default=None, help="Optional commander display name")
    p_add.add_argument("--device", type=str, default="auto")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list")
    p_list.set_defaults(func=cmd_list)

    p_eval = sub.add_parser("eval-candidate")
    p_eval.add_argument("ckpt", type=str)
    p_eval.add_argument("--device", type=str, default="auto")
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--matches", type=int, default=10)
    p_eval.add_argument("--top-k", type=int, default=20)
    p_eval.add_argument("--candidate-k", type=int, default=5)
    p_eval.add_argument("--log", type=str, default=None)
    p_eval.set_defaults(func=cmd_eval_candidate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
