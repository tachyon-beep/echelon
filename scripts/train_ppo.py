# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echelon import EchelonEnv, EnvConfig, WorldConfig
from echelon.agents.heuristic import HeuristicPolicy
from echelon.arena.glicko2 import GameResult
from echelon.arena.league import League, LeagueEntry
from echelon.arena.match import play_match
from echelon.constants import PACK_SIZE
from echelon.rl.model import ActorCriticLSTM


def stack_obs(obs: dict[str, np.ndarray], ids: list[str]) -> np.ndarray:
    return np.stack([obs[aid] for aid in ids], axis=0)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def find_latest_checkpoint(run_dir: Path) -> Path:
    best: tuple[int, Path] | None = None
    for path in run_dir.glob("ckpt_*.pt"):
        try:
            update = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if best is None or update > best[0]:
            best = (update, path)
    if best is None:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return best[1]


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_info(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            p = subprocess.run(
                args,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return None
        if p.returncode != 0:
            return None
        out = (p.stdout or "").strip()
        return out if out else None

    is_git = _run(["git", "rev-parse", "--is-inside-work-tree"])
    if is_git != "true":
        return {"is_git_repo": False}

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    describe = _run(["git", "describe", "--always", "--dirty", "--tags"])
    status = _run(["git", "status", "--porcelain=v1"]) or ""
    changed = [line[3:] for line in status.splitlines() if len(line) >= 4]
    return {
        "is_git_repo": True,
        "commit": commit,
        "branch": branch,
        "describe": describe,
        "dirty": bool(changed),
        "changed_paths": changed[:200],
    }


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    parts = [p for p in parts if p]
    return parts or None


def _read_wandb_run_id(run_dir: Path) -> str | None:
    try:
        value = (run_dir / "wandb_run_id.txt").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return value or None


def _write_wandb_run_id(run_dir: Path, run_id: str) -> None:
    (run_dir / "wandb_run_id.txt").write_text(f"{run_id}\n", encoding="utf-8")


def compute_gae(
    rewards: torch.Tensor,  # [T, B]
    values: torch.Tensor,  # [T, B]
    dones: torch.Tensor,  # [T, B] done at start of step (like CleanRL)
    next_value: torch.Tensor,  # [B]
    next_done: torch.Tensor,  # [B] done at end of rollout / start of next step
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(rewards.size(1), device=rewards.device)
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - next_done.float()
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1].float()
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def push_replay(url: str, replay: dict) -> None:
    if not url:
        return
    try:
        import requests
        requests.post(url, json=replay, timeout=5)
    except Exception as e:
        print(f"Failed to push replay: {e}")


@torch.no_grad()
def evaluate_vs_heuristic(
    model: ActorCriticLSTM,
    env_cfg: EnvConfig,
    episodes: int,
    seed: int,
    device: torch.device,
    *,
    seeds: list[int] | None = None,
    record: bool = False,
) -> tuple[dict, dict | None]:
    # Set record_replay if requested
    env_cfg = replace(env_cfg, record_replay=record)
    env = EchelonEnv(env_cfg)
    heuristic = HeuristicPolicy()

    blue_ids = env.blue_ids
    red_ids = env.red_ids

    if seeds is None:
        seeds = [int(seed) + int(i) for i in range(int(episodes))]
    wins = {"blue": 0, "red": 0, "draw": 0}
    hp_margins: list[float] = []
    last_replay = None

    for ep, ep_seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(ep_seed))
        lstm_state = model.initial_state(batch_size=len(blue_ids), device=device)
        done = torch.zeros(len(blue_ids), device=device)

        while True:
            obs_b = torch.from_numpy(stack_obs(obs, blue_ids)).to(device)
            action_b, _, _, _, lstm_state = model.get_action_and_value(obs_b, lstm_state, done)
            action_np = action_b.cpu().numpy()

            actions = {bid: action_np[i] for i, bid in enumerate(blue_ids)}
            for rid in red_ids:
                actions[rid] = heuristic.act(env, rid)

            obs, rewards, terminations, truncations, infos = env.step(actions)

            blue_alive = env.sim.team_alive("blue")
            red_alive = env.sim.team_alive("red")
            done = torch.tensor(
                [terminations[bid] or truncations[bid] for bid in blue_ids], dtype=torch.float32, device=device
            )

            if any(truncations.values()) or (not blue_alive) or (not red_alive):
                outcome = env.last_outcome or {"winner": "draw", "hp": env.team_hp()}
                winner = outcome.get("winner", "draw")
                wins[winner] += 1
                hp = outcome.get("hp") or env.team_hp()
                hp_margins.append(float(hp.get("blue", 0.0) - hp.get("red", 0.0)))
                
                if record and ep == 0:
                    last_replay = env.get_replay()
                break

    wins["episodes"] = int(len(seeds))
    wins["win_rate"] = float(wins["blue"] / max(1, len(seeds)))
    wins["mean_hp_margin"] = float(np.mean(hp_margins)) if hp_margins else 0.0
    return wins, last_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--packs-per-team", type=int, default=2, help="Number of 10-mech packs (1 Heavy, 5 Med, 3 Light, 1 Scout) per team")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "partial"])
    parser.add_argument("--dt-sim", type=float, default=0.05)
    parser.add_argument("--decision-repeat", type=int, default=5)
    parser.add_argument("--episode-seconds", type=float, default=60.0)
    parser.add_argument("--comm-dim", type=int, default=8, help="Pack-local comm message size (0 disables)")
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default="",
        help="Optional comma-separated eval seeds (overrides --eval-episodes/seed schedule)",
    )
    parser.add_argument(
        "--save-eval-replay",
        action="store_true",
        help="Save one replay JSON from each evaluation to <run-dir>/replays/",
    )
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--run-dir", type=str, default="runs/train")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path, or 'latest' (from run-dir)")
    parser.add_argument("--push-url", type=str, default="http://127.0.0.1:8090/push", help="URL to POST replays to")
    parser.add_argument(
        "--train-mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "arena"],
        help="Opponent for red team during training: heuristic | arena",
    )
    parser.add_argument("--arena-league", type=str, default="runs/arena/league.json", help="League file (arena mode)")
    parser.add_argument("--arena-top-k", type=int, default=20, help="Sample opponents from top-K commanders (arena mode)")
    parser.add_argument("--arena-candidate-k", type=int, default=5, help="Plus K recent candidates (arena mode)")
    parser.add_argument(
        "--arena-refresh-episodes",
        type=int,
        default=20,
        help="Reload league/pool every N episodes (0 disables refresh)",
    )
    parser.add_argument(
        "--arena-submit",
        action="store_true",
        help="After training, run Glicko-2 evaluation matches and update the arena league (arena mode)",
    )
    parser.add_argument("--arena-submit-matches", type=int, default=10, help="Matches to play when --arena-submit")
    parser.add_argument("--arena-submit-log", type=str, default=None, help="Optional JSON log path for --arena-submit")

    # Weights & Biases (W&B)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging (implies --wandb-mode online)")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=["disabled", "offline", "online"],
        help="W&B mode: disabled | offline | online",
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project (default: echelon)")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (user/team)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group")
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags")
    parser.add_argument(
        "--wandb-id",
        type=str,
        default=None,
        help="W&B run id to resume (defaults to <run-dir>/wandb_run_id.txt if present)",
    )
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default="allow",
        choices=["allow", "must", "never"],
        help="W&B resume behavior when a run id is provided",
    )
    parser.add_argument(
        "--wandb-watch",
        type=str,
        default="off",
        choices=["off", "gradients", "all"],
        help="wandb.watch setting (off|gradients|all). Warning: expensive.",
    )
    parser.add_argument("--wandb-watch-freq", type=int, default=500, help="wandb.watch log frequency")
    parser.add_argument(
        "--wandb-artifacts",
        type=str,
        default="best",
        choices=["none", "best", "all"],
        help="W&B artifacts to log: none | best | all (includes eval replays)",
    )
    args = parser.parse_args()
    if args.wandb_project is not None and not args.wandb_project.strip():
        args.wandb_project = None
    if args.wandb_entity is not None and not args.wandb_entity.strip():
        args.wandb_entity = None
    if args.wandb_tags is not None and not args.wandb_tags.strip():
        args.wandb_tags = None
    if args.wandb_id is not None and not args.wandb_id.strip():
        args.wandb_id = None

    if args.wandb and args.wandb_mode == "disabled":
        args.wandb_mode = "online"
    if args.wandb_project and args.wandb_mode == "disabled":
        args.wandb_mode = "online"

    set_seed(args.seed)

    # Suppress noisy CPU-only CUDA warnings, but keep hard CUDA errors.
    warnings.filterwarnings("ignore", message=r"CUDA initialization:.*", category=UserWarning)

    device = resolve_device(args.device)
    print(f"device={device} cuda_available={torch.cuda.is_available()} torch={torch.__version__}")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_ckpt = None
    resume_path: Path | None = None
    if args.resume:
        resume_path = find_latest_checkpoint(run_dir) if args.resume == "latest" else Path(args.resume)
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        env_cfg_dict = dict(resume_ckpt["env_cfg"])
        world = WorldConfig(**env_cfg_dict["world"])
        env_cfg = EnvConfig(
            **{
                **{k: v for k, v in env_cfg_dict.items() if k != "world"},
                "world": world,
                "seed": int(args.seed),
                "record_replay": False,
            }
        )
        print(f"resuming from {resume_path} (update={resume_ckpt.get('update')})")
    else:
        env_cfg = EnvConfig(
            world=WorldConfig(size_x=args.size, size_y=args.size, size_z=20),
            num_packs=args.packs_per_team,
            dt_sim=args.dt_sim,
            decision_repeat=args.decision_repeat,
            max_episode_seconds=args.episode_seconds,
            observation_mode=args.mode,
            comm_dim=args.comm_dim,
            record_replay=False,
            seed=args.seed,
        )

    env = EchelonEnv(env_cfg)
    heuristic = HeuristicPolicy() if args.train_mode == "heuristic" else None

    # Run provenance (best-effort; safe to run outside git as well).
    repo_git = git_info(ROOT)

    obs, _ = env.reset(seed=args.seed)
    next_obs_dict = obs
    blue_ids = env.blue_ids
    red_ids = env.red_ids

    obs_dim = int(next(iter(obs.values())).shape[0])
    action_dim = env.ACTION_DIM

    model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    def _env_signature(env_cfg0: EnvConfig) -> dict:
        d = asdict(env_cfg0)
        d.pop("seed", None)
        d.pop("record_replay", None)
        return d

    eval_seeds: list[int] | None = None
    if str(args.eval_seeds).strip():
        eval_seeds = [int(s.strip()) for s in str(args.eval_seeds).split(",") if s.strip()]

    arena_league_path = Path(args.arena_league)
    arena_rng = random.Random(int(args.seed) + 13_371)
    arena_pool: list[LeagueEntry] = []
    arena_cache: dict[str, ActorCriticLSTM] = {}
    arena_opponent_id: str | None = None
    arena_opponent: ActorCriticLSTM | None = None
    arena_lstm_state = None
    arena_done = torch.zeros(len(red_ids), device=device)

    def _arena_refresh_pool() -> None:
        nonlocal arena_pool
        if not arena_league_path.exists():
            raise FileNotFoundError(f"missing arena league: {arena_league_path}")
        league = League.load(arena_league_path)
        sig = _env_signature(env_cfg)
        if league.env_signature is not None and league.env_signature != sig:
            raise RuntimeError("training env_cfg does not match league env_signature")
        pool = league.top_commanders(int(args.arena_top_k)) + league.recent_candidates(int(args.arena_candidate_k))
        by_id = {e.entry_id: e for e in pool}
        arena_pool = list(by_id.values())
        if not arena_pool:
            raise RuntimeError("arena league has no eligible opponents (need commanders/candidates)")

    def _arena_load_model(ckpt_path: str) -> ActorCriticLSTM:
        ckpt = torch.load(ckpt_path, map_location=device)
        opp = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim).to(device)
        opp.load_state_dict(ckpt["model_state"])
        opp.eval()
        return opp

    def _arena_sample_opponent(reset_hidden: bool) -> None:
        nonlocal arena_opponent_id, arena_opponent, arena_lstm_state, arena_done
        entry = arena_rng.choice(arena_pool)
        arena_opponent_id = entry.entry_id
        if arena_opponent_id not in arena_cache:
            arena_cache[arena_opponent_id] = _arena_load_model(entry.ckpt_path)
        arena_opponent = arena_cache[arena_opponent_id]
        if reset_hidden:
            arena_lstm_state = arena_opponent.initial_state(batch_size=len(red_ids), device=device)
            arena_done = torch.ones(len(red_ids), device=device)

    if args.train_mode == "arena":
        _arena_refresh_pool()
        _arena_sample_opponent(reset_hidden=True)

    if not (run_dir / "config.json").exists():
        (run_dir / "config.json").write_text(json.dumps({"env": asdict(env_cfg)}, indent=2), encoding="utf-8")

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        optimizer_to_device(optimizer, device)
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    batch_size = len(blue_ids)
    lstm_state = model.initial_state(batch_size=batch_size, device=device)
    next_obs = torch.from_numpy(stack_obs(obs, blue_ids)).to(device)
    next_done = torch.zeros(batch_size, device=device)

    if resume_ckpt is not None:
        global_step = int(resume_ckpt.get("global_step", 0))
        episodes = int(resume_ckpt.get("episodes", 0))
        start_update = int(resume_ckpt.get("update", 0)) + 1
        end_update = start_update + int(args.updates) - 1
    else:
        global_step = 0
        episodes = 0
        start_update = 1
        end_update = int(args.updates)

    start_time = time.time()
    start_global_step = global_step
    episodic_returns: list[float] = []
    episodic_lengths: list[int] = []
    episodic_outcomes: list[str] = []
    current_ep_return = 0.0
    current_ep_len = 0

    metrics_f = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    metrics_f.write(
        json.dumps(
            {
                "type": "start",
                "time": time.time(),
                "device": str(device),
                "cuda_available": bool(torch.cuda.is_available()),
                "torch": torch.__version__,
                "git": repo_git,
                "run_dir": str(run_dir),
                "resume_path": str(resume_path) if resume_path is not None else None,
                "resume_update": int(resume_ckpt.get("update")) if resume_ckpt is not None else None,
                "start_update": int(start_update),
                "end_update": int(end_update),
                "global_step": int(global_step),
                "episodes": int(episodes),
                "obs_dim": int(obs_dim),
                "action_dim": int(action_dim),
                "env_signature": _env_signature(env_cfg),
                "env_constants": {
                    "PACK_SIZE": int(PACK_SIZE),
                    "CONTACT_SLOTS": int(getattr(env, "CONTACT_SLOTS", 0)),
                    "CONTACT_DIM": int(getattr(env, "CONTACT_DIM", 0)),
                    "LOCAL_MAP_R": int(getattr(env, "LOCAL_MAP_R", 0)),
                    "LOCAL_MAP_DIM": int(getattr(env, "LOCAL_MAP_DIM", 0)),
                    "BASE_ACTION_DIM": int(getattr(env, "BASE_ACTION_DIM", 0)),
                    "TARGET_DIM": int(getattr(env, "TARGET_DIM", 0)),
                    "EWAR_DIM": int(getattr(env, "EWAR_DIM", 0)),
                    "OBS_CTRL_DIM": int(getattr(env, "OBS_CTRL_DIM", 0)),
                    "COMM_DIM": int(getattr(env, "comm_dim", 0)),
                },
                "args": vars(args),
                "env": asdict(env_cfg),
                "eval_seeds": eval_seeds,
            }
        )
        + "\n"
    )
    metrics_f.flush()

    wandb_run = None
    wandb_log_best_model = False
    wandb_log_eval_replays = False
    if args.wandb_mode != "disabled":
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "W&B logging enabled but `wandb` is not installed. "
                "Install `wandb` or disable W&B via --wandb-mode disabled."
            ) from e

        wandb_log_best_model = args.wandb_artifacts in {"best", "all"}
        wandb_log_eval_replays = args.wandb_artifacts == "all"

        wandb_resume = None
        wandb_id = None
        if args.wandb_resume != "never":
            wandb_id = args.wandb_id or _read_wandb_run_id(run_dir)
            if wandb_id:
                wandb_resume = args.wandb_resume

        wandb_run = wandb.init(
            project=args.wandb_project or os.environ.get("WANDB_PROJECT") or "echelon",
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=_parse_csv_list(args.wandb_tags),
            mode=args.wandb_mode,
            id=wandb_id,
            resume=wandb_resume,
            config={
                "args": vars(args),
                "env_cfg": asdict(env_cfg),
                "git": repo_git,
                "device": str(device),
                "env_signature": _env_signature(env_cfg),
            },
            dir=str(run_dir),
        )

        if getattr(wandb_run, "id", None):
            _write_wandb_run_id(run_dir, str(wandb_run.id))

        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("eval/*", step_metric="train/global_step")
        wandb.define_metric("arena/*", step_metric="train/global_step")

        if args.wandb_watch != "off":
            wandb.watch(model, log=args.wandb_watch, log_freq=int(args.wandb_watch_freq))

    best_win_rate = -1.0
    best_mean_hp_margin = float("-inf")
    best_json_path = run_dir / "best.json"
    if best_json_path.exists():
        try:
            best_data = json.loads(best_json_path.read_text(encoding="utf-8"))
            best_win_rate = float(best_data.get("win_rate", best_win_rate))
            best_mean_hp_margin = float(best_data.get("mean_hp_margin", best_mean_hp_margin))
        except (ValueError, TypeError):
            pass

    for update in range(start_update, end_update + 1):
        obs_buf = torch.zeros(args.rollout_steps, batch_size, obs_dim, device=device)
        actions_buf = torch.zeros(args.rollout_steps, batch_size, action_dim, device=device)
        logprobs_buf = torch.zeros(args.rollout_steps, batch_size, device=device)
        rewards_buf = torch.zeros(args.rollout_steps, batch_size, device=device)
        dones_buf = torch.zeros(args.rollout_steps, batch_size, device=device)
        values_buf = torch.zeros(args.rollout_steps, batch_size, device=device)
        entropies_buf = torch.zeros(args.rollout_steps, batch_size, device=device)

        init_state = lstm_state

        for step in range(args.rollout_steps):
            global_step += batch_size
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, entropy, value, lstm_state = model.get_action_and_value(
                    next_obs, lstm_state, next_done
                )

            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value
            entropies_buf[step] = entropy

            action_np = action.cpu().numpy()
            action_dict = {bid: action_np[i] for i, bid in enumerate(blue_ids)}
            if args.train_mode == "heuristic":
                assert heuristic is not None
                for rid in red_ids:
                    action_dict[rid] = heuristic.act(env, rid)
            else:
                assert arena_opponent is not None
                assert arena_lstm_state is not None
                obs_r = torch.from_numpy(stack_obs(next_obs_dict, red_ids)).to(device)
                with torch.no_grad():
                    act_r, _, _, _, arena_lstm_state = arena_opponent.get_action_and_value(
                        obs_r, arena_lstm_state, arena_done
                    )
                act_r_np = act_r.cpu().numpy()
                for i, rid in enumerate(red_ids):
                    action_dict[rid] = act_r_np[i]

            next_obs_dict, rewards, terminations, truncations, infos = env.step(action_dict)

            blue_alive = env.sim.team_alive("blue")
            red_alive = env.sim.team_alive("red")
            ep_over = any(truncations.values()) or (not blue_alive) or (not red_alive)

            r_b = torch.tensor([rewards[bid] for bid in blue_ids], dtype=torch.float32, device=device)
            rewards_buf[step] = r_b
            current_ep_return += float(r_b.sum().item())
            current_ep_len += 1

            next_done = torch.tensor(
                [terminations[bid] or truncations[bid] for bid in blue_ids], dtype=torch.float32, device=device
            )
            if args.train_mode == "arena":
                arena_done = torch.tensor(
                    [terminations[rid] or truncations[rid] for rid in red_ids], dtype=torch.float32, device=device
                )

            if ep_over:
                ep_ret = float(current_ep_return)
                ep_len = int(current_ep_len)
                episodic_returns.append(current_ep_return)
                episodic_lengths.append(current_ep_len)
                
                outcome = env.last_outcome or {"winner": "unknown"}
                episodic_outcomes.append(outcome.get("winner", "unknown"))

                metrics_f.write(
                    json.dumps(
                        {
                            "type": "episode",
                            "time": time.time(),
                            "episode": int(episodes) + 1,
                            "return": float(ep_ret),
                            "len": int(ep_len),
                            "winner": str(outcome.get("winner", "unknown")),
                            "reason": outcome.get("reason"),
                            "hp": outcome.get("hp"),
                            "zone_score": outcome.get("zone_score"),
                            "stats": outcome.get("stats"),
                        }
                    )
                    + "\n"
                )

                current_ep_return = 0.0
                current_ep_len = 0
                episodes += 1
                next_obs_dict, _ = env.reset(seed=args.seed + episodes)
                # Ensure LSTM resets at the start of the next step (episode boundary).
                next_done = torch.ones(batch_size, device=device)
                if args.train_mode == "arena":
                    if args.arena_refresh_episodes > 0 and episodes % int(args.arena_refresh_episodes) == 0:
                        _arena_refresh_pool()
                    _arena_sample_opponent(reset_hidden=True)

            next_obs = torch.from_numpy(stack_obs(next_obs_dict, blue_ids)).to(device)

        with torch.no_grad():
            next_value, _ = model.get_value(next_obs, lstm_state, next_done)

        advantages, returns = compute_gae(
            rewards=rewards_buf,
            values=values_buf,
            dones=dones_buf,
            next_value=next_value,
            next_done=next_done,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO updates (recurrent full-batch for now).
        for epoch in range(args.update_epochs):
            new_logprobs = torch.zeros_like(logprobs_buf)
            new_values = torch.zeros_like(values_buf)
            new_entropies = torch.zeros_like(entropies_buf)

            lstm_state_train = init_state
            for t in range(args.rollout_steps):
                _, lp, ent, val, lstm_state_train = model.get_action_and_value(
                    obs_buf[t], lstm_state_train, dones_buf[t], action=actions_buf[t]
                )
                new_logprobs[t] = lp
                new_values[t] = val
                new_entropies[t] = ent

            logratio = new_logprobs - logprobs_buf
            ratio = logratio.exp()

            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * (returns - new_values).pow(2).mean()
            entropy_loss = new_entropies.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        sps = int((global_step - start_global_step) / max(1e-6, (time.time() - start_time)))
        
        # Stats Aggregation
        window = 20
        avg_return = float(np.mean(episodic_returns[-window:])) if episodic_returns else 0.0
        avg_len = float(np.mean(episodic_lengths[-window:])) if episodic_lengths else 0.0
        
        recent_outcomes = episodic_outcomes[-window:]
        n_out = len(recent_outcomes)
        if n_out > 0:
            win_rate_blue = recent_outcomes.count("blue") / n_out
            win_rate_red = recent_outcomes.count("red") / n_out
        else:
            win_rate_blue = 0.0
            win_rate_red = 0.0

        if update % 10 == 0 or update == start_update:
            print(
                f"update {update:05d} | steps {global_step} | episodes {episodes} | "
                f"ret {avg_return:.1f} | len {avg_len:.1f} | "
                f"win_blue {win_rate_blue:.2f} | win_red {win_rate_red:.2f} | "
                f"loss {loss.item():.3f} | sps {sps}"
            )

        eval_stats = None
        if args.eval_every > 0 and update % args.eval_every == 0:
            if eval_seeds is not None:
                seeds_now = list(eval_seeds)
                seed0 = int(seeds_now[0]) if seeds_now else int(args.seed)
                eval_episodes = int(len(seeds_now))
            else:
                seed0 = int(args.seed + 10_000 + update * 13)
                eval_episodes = int(args.eval_episodes)
                seeds_now = [seed0 + i for i in range(eval_episodes)]

            eval_stats, replay = evaluate_vs_heuristic(
                model=model,
                env_cfg=env_cfg,
                episodes=eval_episodes,
                seed=seed0,
                device=device,
                seeds=seeds_now,
                record=(bool(args.push_url) or bool(args.save_eval_replay)),
            )
            eval_stats["seeds"] = seeds_now
            print(f"eval vs heuristic: {eval_stats}")
            if replay is not None:
                replay["run"] = {
                    "kind": "eval_vs_heuristic",
                    "run_dir": str(run_dir),
                    "update": int(update),
                    "global_step": int(global_step),
                    "episodes": int(episodes),
                    "eval": dict(eval_stats),
                    "eval_seeds": seeds_now,
                    "git": dict(repo_git),
                    "env_signature": _env_signature(env_cfg),
                }
                if args.save_eval_replay:
                    replay_dir = run_dir / "replays"
                    replay_dir.mkdir(parents=True, exist_ok=True)
                    out_path = replay_dir / f"eval_update_{update:05d}_seed_{seed0}.json"
                    out_path.write_text(json.dumps(replay), encoding="utf-8")
                    print(f"saved eval replay: {out_path}")
                    if wandb_run is not None and wandb_log_eval_replays:
                        artifact = wandb.Artifact(
                            name=f"replay-{wandb_run.id}-update-{update:05d}",
                            type="replay",
                            metadata={
                                "update": int(update),
                                "global_step": int(global_step),
                                "seed": int(seed0),
                                "win_rate": float(eval_stats.get("win_rate", 0.0)),
                            },
                        )
                        artifact.add_file(str(out_path))
                        wandb_run.log_artifact(artifact)
                if args.push_url:
                    push_replay(args.push_url, replay)

        best_updated = False
        best_snapshot: str | None = None
        if eval_stats is not None:
            win_rate = float(eval_stats.get("win_rate", 0.0))
            mean_hp_margin = float(eval_stats.get("mean_hp_margin", 0.0))
            improved = (win_rate > best_win_rate + 1e-6) or (
                abs(win_rate - best_win_rate) <= 1e-6 and mean_hp_margin > best_mean_hp_margin + 1e-6
            )
            if improved:
                best_updated = True
                best_win_rate = win_rate
                best_mean_hp_margin = mean_hp_margin
                best_ckpt = {
                    "update": update,
                    "global_step": global_step,
                    "episodes": episodes,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "env_cfg": asdict(env_cfg),
                    "args": vars(args),
                    "eval": eval_stats,
                }
                best_pt = run_dir / "best.pt"
                torch.save(best_ckpt, best_pt)
                best_snapshot = str(best_pt)
                best_json_path.write_text(
                    json.dumps(
                        {
                            "update": int(update),
                            "global_step": int(global_step),
                            "episodes": int(episodes),
                            "win_rate": float(win_rate),
                            "mean_hp_margin": float(mean_hp_margin),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"new best: win_rate={win_rate:.3f} mean_hp_margin={mean_hp_margin:.2f} @ update {update}")

                if wandb_run is not None and wandb_log_best_model:
                    artifact = wandb.Artifact(
                        name=f"model-{wandb_run.id}",
                        type="model",
                        metadata={"update": int(update), "global_step": int(global_step), **best_ckpt["eval"]},
                    )
                    artifact.add_file(str(best_pt))
                    wandb_run.log_artifact(artifact, aliases=["best", f"update_{update:05d}"])

        saved_ckpt: str | None = None
        if args.save_every > 0 and update % args.save_every == 0:
            ckpt = {
                "update": update,
                "global_step": global_step,
                "episodes": episodes,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "env_cfg": asdict(env_cfg),
                "args": vars(args),
            }
            p = run_dir / f"ckpt_{update:05d}.pt"
            torch.save(ckpt, p)
            saved_ckpt = str(p)

        metrics_f.write(
            json.dumps(
                {
                    "type": "update",
                    "time": time.time(),
                    "update": int(update),
                    "global_step": int(global_step),
                    "episodes": int(episodes),
                    "avg_return_10": float(avg_return),
                    "loss": float(loss.item()),
                    "pg_loss": float(pg_loss.item()),
                    "v_loss": float(v_loss.item()),
                    "entropy": float(entropy_loss.item()),
                    "sps": int(sps),
                    "eval": eval_stats,
                    "best_updated": best_updated,
                    "best_snapshot": best_snapshot,
                    "saved_ckpt": saved_ckpt,
                }
            )
            + "\n"
        )
        metrics_f.flush()

        if wandb_run is not None:
            wandb_metrics = {
                "train/update": int(update),
                "train/loss": float(loss.item()),
                "train/pg_loss": float(pg_loss.item()),
                "train/v_loss": float(v_loss.item()),
                "train/entropy": float(entropy_loss.item()),
                "train/sps": int(sps),
                "train/global_step": int(global_step),
                "train/episodes": int(episodes),
                "train/avg_return": float(avg_return),
                "train/avg_len": float(avg_len),
                "train/win_rate_blue": win_rate_blue,
                "train/win_rate_red": win_rate_red,
            }
            if eval_stats:
                wandb_metrics.update(
                    {
                        "eval/win_rate": float(eval_stats.get("win_rate", 0.0)),
                        "eval/mean_hp_margin": float(eval_stats.get("mean_hp_margin", 0.0)),
                        "eval/episodes": int(eval_stats.get("episodes", 0)),
                    }
                )
            wandb_run.log(wandb_metrics, step=int(global_step))

    if args.train_mode == "arena" and args.arena_submit:
        # Save a snapshot of the final trained model as the tournament candidate.
        candidate_path = run_dir / "arena_candidate.pt"
        torch.save(
            {
                "update": int(end_update),
                "global_step": int(global_step),
                "episodes": int(episodes),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "env_cfg": asdict(env_cfg),
                "args": vars(args),
            },
            candidate_path,
        )

        # Evaluate candidate vs the league and apply one Glicko-2 rating period.
        league = League.load(arena_league_path)
        sig = _env_signature(env_cfg)
        if league.env_signature is None:
            league.env_signature = sig
        elif league.env_signature != sig:
            raise RuntimeError("training env_cfg does not match league env_signature")

        candidate_entry = league.upsert_checkpoint(candidate_path, kind="candidate")
        pool = league.top_commanders(int(args.arena_top_k)) + league.recent_candidates(
            int(args.arena_candidate_k), exclude_id=candidate_entry.entry_id
        )
        pool = [e for e in pool if e.entry_id != candidate_entry.entry_id]
        if not pool:
            raise RuntimeError("no opponents available (need commanders/candidates in league)")

        candidate_model = _arena_load_model(str(candidate_path))
        match_cfg = replace(env_cfg, record_replay=False)

        matches = int(args.arena_submit_matches)
        rng = random.Random(int(args.seed) * 1_000_000 + 54321)
        results: dict[str, list[GameResult]] = {}
        game_log: list[dict] = []

        seed0 = int(args.seed) * 1_000_000 + 12345
        for i in range(matches):
            opp_entry = rng.choice(pool)
            if opp_entry.entry_id not in arena_cache:
                arena_cache[opp_entry.entry_id] = _arena_load_model(opp_entry.ckpt_path)
            opp_model = arena_cache[opp_entry.entry_id]

            candidate_as_blue = (i % 2) == 0
            match_seed = seed0 + i * 9973
            if candidate_as_blue:
                out = play_match(
                    env_cfg=match_cfg,
                    blue_policy=candidate_model,
                    red_policy=opp_model,
                    seed=int(match_seed),
                    device=device,
                )
                cand_team = "blue"
            else:
                out = play_match(
                    env_cfg=match_cfg,
                    blue_policy=opp_model,
                    red_policy=candidate_model,
                    seed=int(match_seed),
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
                    "i": int(i),
                    "seed": int(match_seed),
                    "candidate_as": cand_team,
                    "opponent": opp_entry.entry_id,
                    "opponent_name": opp_entry.commander_name,
                    "winner": out.winner,
                    "candidate_score": float(cand_score),
                    "hp": out.hp,
                }
            )
            print(
                f"arena_submit match {i+1}/{matches}: opp={opp_entry.entry_id} cand_as={cand_team} "
                f"winner={out.winner} score={cand_score}"
            )

        league.apply_rating_period(results)
        promoted = league.promote_if_topk(candidate_entry.entry_id, top_k=int(args.arena_top_k))
        league.save(arena_league_path)

        cand_post = league.entries[candidate_entry.entry_id]
        print(
            f"arena_submit candidate rating: {cand_post.rating.rating:.1f}±{cand_post.rating.rd:.1f} "
            f"vol={cand_post.rating.vol:.4f} games={cand_post.games} promoted={promoted} "
            f"name={cand_post.commander_name}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "arena/rating": float(cand_post.rating.rating),
                    "arena/rd": float(cand_post.rating.rd),
                    "arena/games": int(cand_post.games),
                    "arena/promoted": int(promoted),
                },
                step=int(global_step),
            )

        if args.arena_submit_log:
            log_path = Path(args.arena_submit_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                json.dumps({"candidate": cand_post.as_dict(), "games": game_log}, indent=2),
                encoding="utf-8",
            )
            print(f"arena_submit wrote log: {log_path}")

    metrics_f.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
