#!/usr/bin/env python
# ruff: noqa: E402
"""PPO training script for Echelon.

Usage:
    uv run python scripts/train_ppo.py --total-steps 1000000 --num-envs 8
    uv run python scripts/train_ppo.py --size 100 --updates 200
    uv run python scripts/train_ppo.py --wandb --wandb-run-name "experiment-01"
    uv run python scripts/train_ppo.py --resume latest --total-steps 500000
"""

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

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echelon import EchelonEnv, EnvConfig, WorldConfig
from echelon.arena.glicko2 import GameResult
from echelon.arena.league import League, LeagueEntry
from echelon.arena.match import play_match
from echelon.constants import PACK_SIZE
from echelon.rl.model import ActorCriticLSTM, LSTMState
from echelon.training import PPOConfig, PPOTrainer, RolloutBuffer, VectorEnv, evaluate_vs_heuristic


def _role_for_agent(agent_id: str) -> str:
    """Get role name for an agent ID (e.g., 'blue_0' -> 'heavy')."""
    idx = int(agent_id.split("_")[1]) % PACK_SIZE
    if idx == 0:
        return "heavy"
    elif idx <= 2:
        return "medium"
    elif idx == 3:
        return "light"
    else:
        return "scout"


# ====================================================================
# Arena Self-Play Utilities
# ====================================================================


class LRUModelCache:
    """LRU cache for opponent models to prevent OOM (HIGH-10)."""

    def __init__(self, max_size: int = 10):
        self.max_size = max(1, max_size)
        self._cache: dict[str, ActorCriticLSTM] = {}
        self._order: list[str] = []

    def get(self, key: str) -> ActorCriticLSTM | None:
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, model: ActorCriticLSTM) -> None:
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self.max_size:
            evict_key = self._order.pop(0)
            evicted = self._cache.pop(evict_key)
            evicted.cpu()
            del evicted
        self._cache[key] = model
        self._order.append(key)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# ====================================================================
# Utility Functions
# ====================================================================


def stack_obs(obs: dict[str, np.ndarray], ids: list[str]) -> np.ndarray:
    """Stack observations for given agent IDs into a single array."""
    return np.stack([obs[aid] for aid in ids], axis=0)


def stack_obs_many(obs_list: list[dict[str, np.ndarray]], ids: list[str]) -> np.ndarray:
    """Stack observations from multiple environments."""
    if not obs_list:
        raise ValueError("obs_list must be non-empty")
    return np.concatenate([stack_obs(obs, ids) for obs in obs_list], axis=0)


def resolve_devices(device_arg: str) -> list[torch.device]:
    """Parse device argument into list of torch devices."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        return [torch.device("cpu")]
    parts = [s.strip() for s in device_arg.split(",")]
    return [torch.device(p) for p in parts]


def find_latest_checkpoint(run_dir: Path) -> Path:
    """Find the latest checkpoint in run directory."""
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
    """Move optimizer state tensors to device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def git_info(repo_root: Path) -> dict[str, Any]:
    """Gather git repository provenance information."""

    def _run(args: list[str]) -> str | None:
        try:
            p = subprocess.run(
                args, cwd=str(repo_root), capture_output=True, text=True, timeout=2, check=False
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


def push_replay(url: str, replay: dict, chunk_size: int = 100) -> None:
    """Push replay in chunks to server.

    Sends world once (deduplicated by hash), then streams frames in chunks.
    Each chunk is ~1-2 MB, keeping memory bounded on both ends.
    """
    if not url:
        return

    import gzip
    import hashlib
    import logging
    import uuid

    import requests  # type: ignore[import-untyped]

    logger = logging.getLogger(__name__)

    try:
        base_url = url.rsplit("/push", 1)[0]

        # 1. Push world if not cached
        world = replay.get("world")
        if not world:
            logger.warning("Replay has no world data, skipping push")
            return

        canonical = json.dumps(world, sort_keys=True, separators=(",", ":"))
        world_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]

        # Check server cache
        try:
            head_resp = requests.head(f"{base_url}/worlds/{world_hash}", timeout=2)
            world_cached = head_resp.status_code == 200
        except Exception:
            world_cached = False

        if not world_cached:
            world_data = gzip.compress(json.dumps(world).encode())
            requests.put(
                f"{base_url}/worlds/{world_hash}",
                data=world_data,
                headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
                timeout=10,
            )

        # 2. Push chunks
        frames = replay.get("frames", [])
        if not frames:
            return

        replay_id = uuid.uuid4().hex[:12]
        chunk_count = (len(frames) + chunk_size - 1) // chunk_size

        for i in range(chunk_count):
            chunk = {
                "replay_id": replay_id,
                "world_ref": world_hash,
                "chunk_index": i,
                "chunk_count": chunk_count,
                "frames": frames[i * chunk_size : (i + 1) * chunk_size],
            }
            if i == 0:
                chunk["meta"] = {
                    "episode": replay.get("episode"),
                    "update": replay.get("update"),
                }
                # Include run metadata if present
                if "run" in replay:
                    chunk["run"] = replay["run"]

            requests.post(url, json=chunk, timeout=10)

    except requests.RequestException as e:
        logger.warning(f"Failed to push replay chunk: {e}")
    except Exception as e:
        logger.error(f"Unexpected error pushing replay: {e}")


# ====================================================================
# Argument Parsing
# ====================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    # Environment config
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0,cuda:1")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--packs-per-team", type=int, default=2, help="Number of 10-mech packs per team")
    parser.add_argument("--size", type=int, default=100, help="Map size (x/y)")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "partial"])
    parser.add_argument("--dt-sim", type=float, default=0.05)
    parser.add_argument("--decision-repeat", type=int, default=5)
    parser.add_argument(
        "--game-length",
        type=int,
        default=500,
        help="Max rounds per game (each agent acts once per round)",
    )
    parser.add_argument("--comm-dim", type=int, default=8, help="Pack-local comm message size (0 disables)")

    # Feature toggles
    parser.add_argument("--disable-target-selection", action="store_true")
    parser.add_argument("--disable-ewar", action="store_true")
    parser.add_argument("--disable-obs-control", action="store_true")
    parser.add_argument("--disable-comm", action="store_true")

    # Training duration (pick ONE: --games, --total-steps, or --updates)
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Target completed games. Training runs until this many episodes finish. "
        "Simplest way to specify duration - ignores PPO internals.",
    )
    parser.add_argument("--total-steps", type=int, default=None, help="Total env steps (overrides --updates)")
    parser.add_argument(
        "--updates", type=int, default=200, help="PPO update cycles (ignored if --total-steps or --games set)"
    )

    # PPO hyperparameters (you probably don't need to touch these)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8, help="TBPTT minibatches per epoch")
    parser.add_argument("--tbptt-chunk-length", type=int, default=0, help="TBPTT chunk length (0=auto)")

    # Evaluation
    # 10 fixed eval seeds for consistent comparison across runs
    DEFAULT_EVAL_SEEDS = "42,137,256,314,512,777,1024,1337,2048,3141"
    parser.add_argument(
        "--eval-after-updates",
        type=int,
        default=50,
        dest="eval_every",
        help="Run evaluation every N training updates (0 to disable)",
    )
    parser.add_argument("--eval-episodes", type=int, default=20, help="Games to play per evaluation")
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default=DEFAULT_EVAL_SEEDS,
        help="Comma-separated eval seeds (default: 10 fixed seeds)",
    )
    parser.add_argument("--save-eval-replay", action="store_true")

    # Checkpointing
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--run-dir", type=str, default="runs/train")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path or 'latest'")

    # Training mode
    parser.add_argument("--push-url", type=str, default="http://127.0.0.1:8090/push")
    parser.add_argument("--train-mode", type=str, default="heuristic", choices=["heuristic", "arena"])
    parser.add_argument("--arena-league", type=str, default="runs/arena/league.json")
    parser.add_argument("--arena-top-k", type=int, default=20)
    parser.add_argument("--arena-candidate-k", type=int, default=5)
    parser.add_argument("--arena-refresh-episodes", type=int, default=20)
    parser.add_argument("--arena-submit", action="store_true")
    parser.add_argument("--arena-submit-matches", type=int, default=10)
    parser.add_argument("--arena-submit-log", type=str, default=None)

    # Opponent curriculum (heuristic mode only)
    parser.add_argument(
        "--opfor-weapon-start",
        type=float,
        default=0.2,
        help="Initial weapon fire probability for heuristic opponent (0-1)",
    )
    parser.add_argument(
        "--opfor-weapon-end",
        type=float,
        default=1.0,
        help="Final weapon fire probability for heuristic opponent (0-1)",
    )
    parser.add_argument(
        "--opfor-ramp-updates",
        type=int,
        default=0,
        help="Updates to ramp from start to end (0 = instant full lethality)",
    )

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--wandb-mode", type=str, default="disabled", choices=["disabled", "offline", "online"]
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated tags")
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--wandb-resume", type=str, default="allow", choices=["allow", "must", "never"])
    parser.add_argument("--wandb-watch", type=str, default="off", choices=["off", "gradients", "all"])
    parser.add_argument("--wandb-watch-freq", type=int, default=500)
    parser.add_argument("--wandb-artifacts", type=str, default="best", choices=["none", "best", "all"])

    args = parser.parse_args()

    # Normalize W&B args
    if args.wandb and args.wandb_mode == "disabled":
        args.wandb_mode = "online"
    if args.wandb_project and args.wandb_mode == "disabled":
        args.wandb_mode = "online"

    return args


# ====================================================================
# Configuration Building
# ====================================================================


def build_env_config(args: argparse.Namespace, resume_ckpt: dict | None) -> EnvConfig:
    """Build environment configuration from args and optional resume checkpoint."""
    if resume_ckpt is not None:
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
    else:
        # Convert game length (rounds) to seconds: rounds * dt_sim * decision_repeat
        max_episode_seconds = args.game_length * args.dt_sim * args.decision_repeat
        env_cfg = EnvConfig(
            world=WorldConfig(size_x=args.size, size_y=args.size, size_z=20),
            num_packs=args.packs_per_team,
            dt_sim=args.dt_sim,
            decision_repeat=args.decision_repeat,
            max_episode_seconds=max_episode_seconds,
            observation_mode=args.mode,
            comm_dim=args.comm_dim,
            record_replay=False,
            seed=args.seed,
        )

    # Feature toggles (safe to override even when resuming)
    env_cfg = replace(
        env_cfg,
        enable_target_selection=not args.disable_target_selection,
        enable_ewar=not args.disable_ewar,
        enable_obs_control=not args.disable_obs_control,
        enable_comm=not args.disable_comm,
    )

    return env_cfg


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    """Build PPO configuration from args."""
    return PPOConfig(
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        rollout_steps=args.rollout_steps,
        num_minibatches=args.num_minibatches,
        tbptt_chunk_length=args.tbptt_chunk_length,
    )


# ====================================================================
# W&B Setup
# ====================================================================


def setup_wandb(args: argparse.Namespace, run_dir: Path, env_cfg: EnvConfig, device: torch.device) -> Any:
    """Initialize Weights & Biases logging."""
    if args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as e:
        raise RuntimeError(
            "W&B logging enabled but `wandb` is not installed. "
            "Install `wandb` or disable W&B via --wandb-mode disabled."
        ) from e

    # Read or generate run ID
    wandb_id = None
    if args.wandb_resume != "never":
        wandb_id = args.wandb_id
        if not wandb_id and (run_dir / "wandb_run_id.txt").exists():
            wandb_id = (run_dir / "wandb_run_id.txt").read_text(encoding="utf-8").strip() or None

    wandb_run = wandb.init(
        project=args.wandb_project or os.environ.get("WANDB_PROJECT") or "echelon",
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None,
        mode=args.wandb_mode,
        id=wandb_id,
        resume=args.wandb_resume if wandb_id else None,
        config={
            "args": vars(args),
            "env_cfg": asdict(env_cfg),
            "git": git_info(ROOT),
            "device": str(device),
        },
        dir=str(run_dir),
    )

    if getattr(wandb_run, "id", None):
        (run_dir / "wandb_run_id.txt").write_text(f"{wandb_run.id}\n", encoding="utf-8")

    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("eval/*", step_metric="train/global_step")
    wandb.define_metric("arena/*", step_metric="train/global_step")

    return wandb_run


# ====================================================================
# Main Training Loop
# ====================================================================


def main() -> None:
    args = parse_args()

    # Seed and device setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    warnings.filterwarnings("ignore", message=r"CUDA initialization:.*", category=UserWarning)

    devices = resolve_devices(args.device)
    device = devices[0]
    opponent_device = devices[1] if len(devices) > 1 else device

    print(f"devices={devices} cuda_available={torch.cuda.is_available()} torch={torch.__version__}")
    print(f"learning_device={device} opponent_device={opponent_device}")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resume checkpoint if requested
    resume_ckpt = None
    resume_path: Path | None = None
    if args.resume:
        resume_path = find_latest_checkpoint(run_dir) if args.resume == "latest" else Path(args.resume)
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        print(f"resuming from {resume_path} (update={resume_ckpt.get('update')})")

    # Build configurations
    env_cfg = build_env_config(args, resume_ckpt)
    ppo_cfg = build_ppo_config(args)

    # Initialize environments
    num_envs = args.num_envs
    if num_envs < 1:
        raise ValueError("--num-envs must be >= 1")

    # Determine initial weapon probability for curriculum
    initial_weapon_prob = args.opfor_weapon_start if args.train_mode == "heuristic" else 0.5

    print(f"Initializing {num_envs} environments (mode={args.mode}, size={args.size})...")
    if args.train_mode == "heuristic" and args.opfor_ramp_updates > 0:
        print(
            f"Opponent curriculum: weapon prob {args.opfor_weapon_start:.0%} → "
            f"{args.opfor_weapon_end:.0%} over {args.opfor_ramp_updates} updates"
        )
    venv = VectorEnv(num_envs, env_cfg, initial_weapon_prob=initial_weapon_prob)

    # Extract metadata from a temporary environment
    temp_env = EchelonEnv(env_cfg)
    blue_ids = temp_env.blue_ids
    red_ids = temp_env.red_ids
    action_dim = temp_env.ACTION_DIM

    # Reset environments
    env_episode_counts = [0] * num_envs
    print("Resetting environments...")
    next_obs_dicts = venv.reset([args.seed + i * 1_000_000 for i in range(num_envs)])[0]
    print("Environments ready.")

    obs_dim = int(next(iter(next_obs_dicts[0].values())).shape[0])

    # Initialize model and trainer
    model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim).to(device)
    trainer = PPOTrainer(model, ppo_cfg, device)

    # Save config
    if not (run_dir / "config.json").exists():
        (run_dir / "config.json").write_text(json.dumps({"env": asdict(env_cfg)}, indent=2), encoding="utf-8")

    # Load checkpoint state
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state"])
        trainer.optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        optimizer_to_device(trainer.optimizer, device)
        for group in trainer.optimizer.param_groups:
            group["lr"] = args.lr
        if "return_rms" in resume_ckpt:
            trainer.return_normalizer.rms.load_state_dict(resume_ckpt["return_rms"])

    # Initialize training state
    batch_size_per_env = len(blue_ids)
    batch_size = batch_size_per_env * num_envs
    lstm_state = model.initial_state(batch_size=batch_size, device=device)
    next_obs = torch.from_numpy(stack_obs_many(next_obs_dicts, blue_ids)).to(device)
    next_done = torch.zeros(batch_size, device=device)

    # Compute training duration from --games, --total-steps, or --updates (in priority order)
    steps_per_update = ppo_cfg.rollout_steps * num_envs * batch_size_per_env
    target_games: int | None = None  # If set, stop when this many episodes complete

    if args.games is not None:
        # User specified target games - run indefinitely until target reached
        # Set a very high upper bound; early exit happens when episodes >= target_games
        target_games = args.games
        num_updates = 1_000_000  # Effectively unlimited; we break when target_games reached
        print(f"--games {args.games} → training until {args.games} games complete")
    elif args.total_steps is not None:
        num_updates = max(1, args.total_steps // steps_per_update)
        print(
            f"--total-steps {args.total_steps:,} → {num_updates} updates ({steps_per_update:,} steps/update)"
        )
    else:
        num_updates = args.updates

    if resume_ckpt is not None:
        global_step = int(resume_ckpt.get("global_step", 0))
        episodes = int(resume_ckpt.get("episodes", 0))
        start_update = int(resume_ckpt.get("update", 0)) + 1
        end_update = start_update + num_updates - 1
    else:
        global_step = 0
        episodes = 0
        start_update = 1
        end_update = num_updates

    # Training metrics
    start_time = time.time()
    start_global_step = global_step
    episodic_returns: list[float] = []
    episodic_lengths: list[int] = []
    episodic_outcomes: list[str] = []
    current_ep_returns = [0.0] * num_envs
    current_ep_lens = [0] * num_envs

    # Per-role reward tracking (for blue team only - the learning policy)
    ROLES = ["heavy", "medium", "light", "scout"]
    current_ep_returns_by_role: list[dict[str, float]] = [dict.fromkeys(ROLES, 0.0) for _ in range(num_envs)]
    episodic_returns_by_role: list[dict[str, float]] = []  # list of per-episode {role: total}

    # Combat statistics tracking
    episodic_combat_stats: list[dict[str, float]] = []

    # Zone control metrics tracking
    episodic_zone_stats: list[dict[str, float]] = []

    # Coordination metrics tracking
    episodic_coord_stats: list[dict[str, float]] = []

    # Per-component reward tracking (aggregate across all blue agents)
    COMPONENTS = ["approach", "zone", "damage", "kill", "assist", "death", "terminal"]
    current_ep_components: list[dict[str, float]] = [dict.fromkeys(COMPONENTS, 0.0) for _ in range(num_envs)]
    episodic_components: list[dict[str, float]] = []  # list of per-episode {component: total}

    # Per-role action tracking (activation counts for weapon slots)
    # Action indices: 4=PRIMARY, 5=VENT, 6=SECONDARY, 7=TERTIARY, 8=SPECIAL
    ACTION_SLOTS = ["primary", "vent", "secondary", "tertiary", "special"]
    SLOT_INDICES = [4, 5, 6, 7, 8]  # Corresponding action indices
    # Track activation count and step count per role per slot
    current_ep_actions: list[dict[str, dict[str, int]]] = [
        {role: dict.fromkeys(ACTION_SLOTS, 0) for role in ROLES} for _ in range(num_envs)
    ]
    current_ep_action_steps: list[dict[str, int]] = [
        dict.fromkeys(ROLES, 0)
        for _ in range(num_envs)  # Steps per role
    ]
    episodic_actions: list[dict[str, dict[str, float]]] = []  # Per-episode activation rates

    # Setup logging
    metrics_f = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    metrics_f.write(
        json.dumps(
            {
                "type": "start",
                "time": time.time(),
                "device": str(device),
                "cuda_available": bool(torch.cuda.is_available()),
                "torch": torch.__version__,
                "git": git_info(ROOT),
                "run_dir": str(run_dir),
                "resume_path": str(resume_path) if resume_path else None,
                "resume_update": int(resume_ckpt.get("update")) if resume_ckpt else None,
                "start_update": int(start_update),
                "end_update": int(end_update),
                "global_step": int(global_step),
                "episodes": int(episodes),
                "obs_dim": int(obs_dim),
                "action_dim": int(action_dim),
                "args": vars(args),
                "env": asdict(env_cfg),
            }
        )
        + "\n"
    )
    metrics_f.flush()

    wandb_run = setup_wandb(args, run_dir, env_cfg, device)
    if wandb_run and args.wandb_watch != "off":
        import wandb

        wandb.watch(model, log=args.wandb_watch, log_freq=args.wandb_watch_freq)

    # Evaluation seeds
    eval_seeds: list[int] | None = None
    if args.eval_seeds.strip():
        eval_seeds = [int(s.strip()) for s in args.eval_seeds.split(",") if s.strip()]

    # Arena setup
    arena_league_path = Path(args.arena_league)
    arena_rng = random.Random(args.seed + 13_371)
    arena_pool: list[LeagueEntry] = []
    arena_cache = LRUModelCache(max_size=20)
    arena_opponent_id: str | None = None
    arena_opponent: ActorCriticLSTM | None = None
    arena_lstm_state = None
    arena_done = torch.zeros(num_envs * len(red_ids), device=opponent_device)

    def env_signature(cfg: EnvConfig) -> dict:
        d = asdict(cfg)
        d.pop("seed", None)
        d.pop("record_replay", None)
        return d

    def arena_refresh_pool() -> None:
        nonlocal arena_pool
        if not arena_league_path.exists():
            raise FileNotFoundError(f"missing arena league: {arena_league_path}")
        league = League.load(arena_league_path)
        sig = env_signature(env_cfg)
        if league.env_signature is not None and league.env_signature != sig:
            raise RuntimeError("training env_cfg does not match league env_signature")
        pool = league.top_commanders(args.arena_top_k) + league.recent_candidates(args.arena_candidate_k)
        by_id = {e.entry_id: e for e in pool}
        arena_pool = list(by_id.values())
        if not arena_pool:
            raise RuntimeError("arena league has no eligible opponents")

    def arena_load_model(ckpt_path: str) -> ActorCriticLSTM:
        ckpt = torch.load(ckpt_path, map_location=opponent_device, weights_only=False)
        opp = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim).to(opponent_device)
        opp.load_state_dict(ckpt["model_state"])
        opp.eval()
        return opp

    def arena_sample_opponent(reset_hidden: bool) -> None:
        nonlocal arena_opponent_id, arena_opponent, arena_lstm_state, arena_done
        entry = arena_rng.choice(arena_pool)
        arena_opponent_id = entry.entry_id
        cached = arena_cache.get(arena_opponent_id)
        if cached is None:
            cached = arena_load_model(entry.ckpt_path)
            arena_cache.put(arena_opponent_id, cached)
        arena_opponent = cached
        if reset_hidden:
            arena_lstm_state = arena_opponent.initial_state(
                batch_size=num_envs * len(red_ids), device=opponent_device
            )
            arena_done = torch.ones(num_envs * len(red_ids), device=opponent_device)

    if args.train_mode == "arena":
        arena_refresh_pool()
        arena_sample_opponent(reset_hidden=True)

    # Best model tracking
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

    if target_games is not None:
        print(f"Training until {target_games} games complete...")
    else:
        print(f"Starting {num_updates} training updates...")

    # ====================================================================
    # Training Loop
    # ====================================================================

    for update in range(start_update, end_update + 1):
        episodes_at_update_start = episodes

        # Update opponent curriculum (heuristic mode only)
        if args.train_mode == "heuristic" and args.opfor_ramp_updates > 0:
            progress = min(1.0, (update - 1) / args.opfor_ramp_updates)
            current_weapon_prob = args.opfor_weapon_start + progress * (
                args.opfor_weapon_end - args.opfor_weapon_start
            )
            venv.set_heuristic_weapon_prob(current_weapon_prob)
        else:
            current_weapon_prob = args.opfor_weapon_end if args.train_mode == "heuristic" else 1.0

        # Create rollout buffer
        buffer = RolloutBuffer.create(
            num_steps=args.rollout_steps,
            num_agents=batch_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )

        init_state = lstm_state

        # Action statistics accumulators
        update_action_sum = np.zeros(action_dim, dtype=np.float64)
        update_action_sq_sum = np.zeros(action_dim, dtype=np.float64)
        update_action_count = 0
        update_saturation_count = 0

        # Collect rollout
        for step in range(args.rollout_steps):
            global_step += batch_size
            buffer.obs[step] = next_obs
            buffer.dones[step] = next_done

            with torch.no_grad():
                action, logprob, _entropy, value, lstm_state = model.get_action_and_value(
                    next_obs, lstm_state, next_done
                )

            buffer.actions[step] = action
            buffer.logprobs[step] = logprob
            buffer.values[step] = value

            action_np = action.detach().cpu().numpy()

            # Accumulate action statistics for this update
            actions_flat_np = action_np  # Already flat from model output
            update_action_sum += actions_flat_np.sum(axis=0)
            update_action_sq_sum += (actions_flat_np**2).sum(axis=0)
            update_action_count += actions_flat_np.shape[0]
            update_saturation_count += int((np.abs(actions_flat_np) > 0.95).sum())

            # Prepare action dictionaries
            all_actions_dicts: list[dict[str, np.ndarray]] = [{} for _ in range(num_envs)]

            # Blue team (learning policy)
            for env_idx in range(num_envs):
                off = env_idx * batch_size_per_env
                blue_act_slice = action_np[off : off + batch_size_per_env]
                for i, bid in enumerate(blue_ids):
                    all_actions_dicts[env_idx][bid] = blue_act_slice[i]
                    # Track action activations per role
                    role = _role_for_agent(bid)
                    act = blue_act_slice[i]
                    for slot_name, slot_idx in zip(ACTION_SLOTS, SLOT_INDICES, strict=True):
                        if act[slot_idx] > 0:  # Positive value = activation
                            current_ep_actions[env_idx][role][slot_name] += 1
                    current_ep_action_steps[env_idx][role] += 1

            # Red team (opponent)
            if args.train_mode == "arena":
                obs_r_many = stack_obs_many(next_obs_dicts, red_ids)
                obs_r_torch = torch.from_numpy(obs_r_many).to(opponent_device)
                with torch.no_grad():
                    assert arena_opponent is not None  # Initialized in arena mode
                    assert arena_lstm_state is not None
                    act_r, _, _, _, arena_lstm_state = arena_opponent.get_action_and_value(
                        obs_r_torch, arena_lstm_state, arena_done
                    )
                act_r_np = act_r.detach().cpu().numpy()
                for env_idx in range(num_envs):
                    for i, rid in enumerate(red_ids):
                        all_actions_dicts[env_idx][rid] = act_r_np[env_idx * len(red_ids) + i]
            elif args.train_mode == "heuristic":
                heuristic_acts_list = venv.get_heuristic_actions(red_ids)
                for env_idx in range(num_envs):
                    all_actions_dicts[env_idx].update(heuristic_acts_list[env_idx])

            # Step environments
            next_obs_dicts, rewards_list, terminations_list, truncations_list, infos_list = venv.step(
                all_actions_dicts
            )

            rewards_flat = np.zeros(batch_size, dtype=np.float32)
            done_flat = np.zeros(batch_size, dtype=np.float32)
            arena_done_np = np.zeros(num_envs * len(red_ids), dtype=np.float32)

            team_alive_list = venv.get_team_alive()
            last_outcomes = venv.get_last_outcomes()

            for env_idx in range(num_envs):
                off = env_idx * batch_size_per_env

                # Learning policy rewards and dones
                r_i = np.asarray([rewards_list[env_idx][bid] for bid in blue_ids], dtype=np.float32)
                rewards_flat[off : off + batch_size_per_env] = r_i
                current_ep_returns[env_idx] += float(r_i.sum())
                current_ep_lens[env_idx] += 1

                # Per-role reward accumulation (blue team only)
                for bid in blue_ids:
                    role = _role_for_agent(bid)
                    current_ep_returns_by_role[env_idx][role] += float(rewards_list[env_idx][bid])

                # Per-component reward accumulation
                env_infos = infos_list[env_idx]
                for bid in blue_ids:
                    rc = env_infos.get(bid, {}).get("reward_components", {})
                    for comp in COMPONENTS:
                        current_ep_components[env_idx][comp] += float(rc.get(comp, 0.0))

                d_i = np.asarray(
                    [terminations_list[env_idx][bid] or truncations_list[env_idx][bid] for bid in blue_ids],
                    dtype=np.float32,
                )
                done_flat[off : off + batch_size_per_env] = d_i

                # Opponent dones
                arena_d_i = np.asarray(
                    [terminations_list[env_idx][rid] or truncations_list[env_idx][rid] for rid in red_ids],
                    dtype=np.float32,
                )
                arena_done_np[env_idx * len(red_ids) : (env_idx + 1) * len(red_ids)] = arena_d_i

                # Check episode termination
                # NOTE: Zone wins set terminations=True (objective achieved).
                #       Time limits set truncations=True (time ran out).
                #       Eliminations are detected via team_alive.
                blue_alive = team_alive_list[env_idx]["blue"]
                red_alive = team_alive_list[env_idx]["red"]
                has_termination = any(terminations_list[env_idx].values())
                has_truncation = any(truncations_list[env_idx].values())
                ep_over = has_termination or has_truncation or (not blue_alive) or (not red_alive)

                if ep_over:
                    ep_ret = float(current_ep_returns[env_idx])
                    ep_len = int(current_ep_lens[env_idx])
                    episodic_returns.append(current_ep_returns[env_idx])
                    episodic_lengths.append(current_ep_lens[env_idx])

                    outcome = last_outcomes[env_idx] or {"winner": "unknown"}
                    episodic_outcomes.append(outcome.get("winner", "unknown"))

                    metrics_f.write(
                        json.dumps(
                            {
                                "type": "episode",
                                "time": time.time(),
                                "episode": episodes + 1,
                                "env_idx": env_idx,
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

                    current_ep_returns[env_idx] = 0.0
                    current_ep_lens[env_idx] = 0

                    # Store and reset per-role/component stats
                    episodic_returns_by_role.append(dict(current_ep_returns_by_role[env_idx]))
                    episodic_components.append(dict(current_ep_components[env_idx]))
                    current_ep_returns_by_role[env_idx] = dict.fromkeys(ROLES, 0.0)
                    current_ep_components[env_idx] = dict.fromkeys(COMPONENTS, 0.0)

                    # Extract combat stats from episode outcome
                    ep_combat = {
                        "damage_blue": 0.0,
                        "damage_red": 0.0,
                        "kills_blue": 0.0,
                        "kills_red": 0.0,
                        "assists_blue": 0.0,
                        "deaths_blue": 0.0,
                    }
                    if outcome is not None:
                        stats = outcome.get("stats", {})
                        ep_combat["damage_blue"] = float(stats.get("damage_blue", 0.0))
                        ep_combat["damage_red"] = float(stats.get("damage_red", 0.0))
                        ep_combat["kills_blue"] = float(stats.get("kills_blue", 0.0))
                        ep_combat["kills_red"] = float(stats.get("kills_red", 0.0))
                        ep_combat["assists_blue"] = float(stats.get("assists_blue", 0.0))
                        # Count deaths from blue team alive status
                        deaths = sum(
                            1 for bid in blue_ids if not infos_list[env_idx].get(bid, {}).get("alive", True)
                        )
                        ep_combat["deaths_blue"] = float(deaths)
                    episodic_combat_stats.append(ep_combat)

                    # Extract zone stats from episode outcome
                    ep_zone = {
                        "zone_ticks_blue": float(stats.get("zone_ticks_blue", 0.0)),
                        "zone_ticks_red": float(stats.get("zone_ticks_red", 0.0)),
                        "contested_ticks": float(stats.get("contested_ticks", 0.0)),
                        "first_zone_entry": float(stats.get("first_zone_entry_step", -1.0)),
                        "episode_length": float(ep_len),
                    }
                    episodic_zone_stats.append(ep_zone)

                    # Extract coordination stats from episode outcome
                    ep_coord = {
                        "pack_dispersion_sum": float(stats.get("pack_dispersion_sum", 0.0)),
                        "pack_dispersion_count": float(stats.get("pack_dispersion_count", 1.0)),
                        "centroid_zone_dist_sum": float(stats.get("centroid_zone_dist_sum", 0.0)),
                        "centroid_zone_dist_count": float(stats.get("centroid_zone_dist_count", 1.0)),
                    }
                    episodic_coord_stats.append(ep_coord)

                    # Store action activation rates per role and reset
                    ep_action_rates: dict[str, dict[str, float]] = {}
                    for role in ROLES:
                        steps = current_ep_action_steps[env_idx][role]
                        if steps > 0:
                            ep_action_rates[role] = {
                                slot: current_ep_actions[env_idx][role][slot] / steps for slot in ACTION_SLOTS
                            }
                        else:
                            ep_action_rates[role] = dict.fromkeys(ACTION_SLOTS, 0.0)
                    episodic_actions.append(ep_action_rates)
                    current_ep_actions[env_idx] = {role: dict.fromkeys(ACTION_SLOTS, 0) for role in ROLES}
                    current_ep_action_steps[env_idx] = dict.fromkeys(ROLES, 0)

                    episodes += 1
                    env_episode_counts[env_idx] += 1

                    # Reset environment
                    reset_obs, _ = venv.reset(
                        [args.seed + env_idx * 1_000_000 + env_episode_counts[env_idx]], indices=[env_idx]
                    )
                    next_obs_dicts[env_idx] = reset_obs[0]

                    # Reset LSTM states
                    done_flat[off : off + batch_size_per_env] = 1.0
                    arena_done_np[env_idx * len(red_ids) : (env_idx + 1) * len(red_ids)] = 1.0

                    if args.train_mode == "arena":
                        if args.arena_refresh_episodes > 0 and episodes % args.arena_refresh_episodes == 0:
                            arena_refresh_pool()
                            arena_sample_opponent(reset_hidden=True)
                        else:
                            arena_sample_opponent(reset_hidden=False)

            buffer.rewards[step] = torch.from_numpy(rewards_flat).to(device)
            next_done = torch.from_numpy(done_flat).to(device)
            arena_done = torch.from_numpy(arena_done_np).to(opponent_device)
            next_obs = torch.from_numpy(stack_obs_many(next_obs_dicts, blue_ids)).to(device)

        # Action statistics for this update
        action_mean = update_action_sum / max(update_action_count, 1)
        action_var = (update_action_sq_sum / max(update_action_count, 1)) - (action_mean**2)
        action_std = np.sqrt(np.maximum(action_var, 0))
        saturation_rate = update_saturation_count / max(update_action_count * action_dim, 1)

        # Compute GAE
        with torch.no_grad():
            detached_state = LSTMState(h=lstm_state.h.detach(), c=lstm_state.c.detach())
            next_value, _ = model.get_value(next_obs, detached_state, next_done)

        buffer.compute_gae(
            next_value=next_value, next_done=next_done, gamma=args.gamma, gae_lambda=args.gae_lambda
        )

        # Compute advantage and value statistics for logging
        # (advantages and returns are guaranteed non-None after compute_gae)
        assert buffer.advantages is not None
        assert buffer.returns is not None
        adv_mean = float(buffer.advantages.mean().item())
        adv_std = float(buffer.advantages.std().item())
        val_mean = float(buffer.values.mean().item())
        val_std = float(buffer.values.std().item())

        # Explained variance: how well values predict returns
        # ev = 1 - Var(returns - values) / Var(returns)
        with torch.no_grad():
            returns_var = buffer.returns.var()
            residual_var = (buffer.returns - buffer.values).var()
            explained_var_t = 1.0 - (residual_var / (returns_var + 1e-8))
            explained_var = float(explained_var_t.item())

        # PPO update
        metrics = trainer.update(buffer, init_state)

        # Extract metrics
        pg_loss = metrics["pg_loss"]
        v_loss = metrics["vf_loss"]
        entropy_loss = metrics["entropy"]
        grad_norm = metrics["grad_norm"]
        approx_kl = metrics["approx_kl"]
        clipfrac = metrics["clipfrac"]
        loss = metrics["loss"]

        sps = int((global_step - start_global_step) / max(1e-6, (time.time() - start_time)))

        # Stats aggregation
        window = 20
        avg_return = float(np.mean(episodic_returns[-window:])) if episodic_returns else 0.0
        avg_len = float(np.mean(episodic_lengths[-window:])) if episodic_lengths else 0.0

        # Return distribution statistics
        if len(episodic_returns) >= 5:
            returns_arr = np.array(episodic_returns[-window:])  # Use recent window
            return_median = float(np.median(returns_arr))
            return_p10 = float(np.percentile(returns_arr, 10))
            return_p90 = float(np.percentile(returns_arr, 90))
            return_std = float(np.std(returns_arr))
        else:
            return_median = avg_return
            return_p10 = avg_return
            return_p90 = avg_return
            return_std = 0.0

        recent_outcomes = episodic_outcomes[-window:]
        n_out = len(recent_outcomes)
        win_rate_blue = recent_outcomes.count("blue") / n_out if n_out > 0 else 0.0
        win_rate_red = recent_outcomes.count("red") / n_out if n_out > 0 else 0.0

        # Per-role average rewards (from recent episodes)
        recent_roles = episodic_returns_by_role[-window:]
        avg_by_role: dict[str, float] = dict.fromkeys(ROLES, 0.0)
        if recent_roles:
            for role in ROLES:
                avg_by_role[role] = float(np.mean([ep[role] for ep in recent_roles]))

        # Per-component average rewards (from recent episodes)
        recent_comps = episodic_components[-window:]
        avg_by_comp: dict[str, float] = dict.fromkeys(COMPONENTS, 0.0)
        if recent_comps:
            for c in COMPONENTS:
                avg_by_comp[c] = float(np.mean([ep[c] for ep in recent_comps]))

        # Compute reward component percentages (of total absolute reward)
        total_abs = sum(abs(v) for v in avg_by_comp.values())
        comp_pct: dict[str, float] = {}
        if total_abs > 1e-6:
            for c in COMPONENTS:
                comp_pct[c] = 100.0 * abs(avg_by_comp[c]) / total_abs
        else:
            comp_pct = dict.fromkeys(COMPONENTS, 0.0)

        # Per-role action activation rates (from recent episodes)
        recent_actions = episodic_actions[-window:]
        avg_actions_by_role: dict[str, dict[str, float]] = {
            role: dict.fromkeys(ACTION_SLOTS, 0.0) for role in ROLES
        }
        if recent_actions:
            for role in ROLES:
                for slot in ACTION_SLOTS:
                    vals = [ep.get(role, {}).get(slot, 0.0) for ep in recent_actions]
                    avg_actions_by_role[role][slot] = float(np.mean(vals)) if vals else 0.0

        # Combat statistics
        recent_combat = episodic_combat_stats[-window:]
        if recent_combat:
            avg_damage_blue = float(np.mean([s["damage_blue"] for s in recent_combat]))
            avg_damage_red = float(np.mean([s["damage_red"] for s in recent_combat]))
            damage_ratio = avg_damage_blue / max(avg_damage_red, 1.0)

            avg_kills = float(np.mean([s["kills_blue"] for s in recent_combat]))
            avg_deaths = float(np.mean([s["deaths_blue"] for s in recent_combat]))
            avg_assists = float(np.mean([s["assists_blue"] for s in recent_combat]))

            # Kill participation = (kills + assists) / total_team_kills
            total_kills = avg_kills + float(np.mean([s["kills_red"] for s in recent_combat]))
            kill_participation = (avg_kills + avg_assists) / max(total_kills, 1.0)

            # Survival rate = agents alive at end / total agents
            survival_rate = 1.0 - (avg_deaths / len(blue_ids))
        else:
            damage_ratio = 1.0
            kill_participation = 0.0
            survival_rate = 1.0
            avg_damage_blue = 0.0

        # Zone control metrics
        recent_zone = episodic_zone_stats[-window:]
        if recent_zone:
            # Zone control margin (blue - red normalized by episode length)
            margins = [
                (z["zone_ticks_blue"] - z["zone_ticks_red"]) / max(z["episode_length"], 1)
                for z in recent_zone
            ]
            zone_margin = float(np.mean(margins))

            # Contested ratio
            contested_ratios = [z["contested_ticks"] / max(z["episode_length"], 1) for z in recent_zone]
            contested_ratio = float(np.mean(contested_ratios))

            # Time to first zone entry (normalized)
            entries = [
                z["first_zone_entry"] / max(z["episode_length"], 1)
                for z in recent_zone
                if z["first_zone_entry"] >= 0
            ]
            time_to_zone = float(np.mean(entries)) if entries else 1.0
        else:
            zone_margin = 0.0
            contested_ratio = 0.0
            time_to_zone = 1.0

        # Print primary line
        opfor_str = f" | opfor {current_weapon_prob:.0%}" if args.train_mode == "heuristic" else ""
        episodes_this_update = episodes - episodes_at_update_start
        episodes_str = f"{episodes} (+{episodes_this_update})" if episodes_this_update > 0 else str(episodes)
        print(
            f"update {update:05d} | steps {global_step} | games {episodes_str} | "
            f"ret {avg_return:.1f} | len {avg_len:.1f} | "
            f"win_blue {win_rate_blue:.2f} | win_red {win_rate_red:.2f} | "
            f"loss {loss:.3f} | sps {sps}{opfor_str}"
        )

        # Print per-role rewards (compact)
        role_str = " | ".join([f"{r[0].upper()}:{avg_by_role[r]:.2f}" for r in ROLES])
        print(f"         roles: {role_str}")

        # Print reward component breakdown with percentages
        comp_parts = []
        for c in COMPONENTS:
            if abs(avg_by_comp[c]) > 1e-6 or comp_pct[c] > 1.0:
                comp_parts.append(f"{c[:4]}:{avg_by_comp[c]:.3f}({comp_pct[c]:.0f}%)")
        if comp_parts:
            print(f"         comps: {' | '.join(comp_parts)}")

        # Print per-role action activation rates (compact: role -> primary/secondary/tertiary)
        # Skip vent/special as they're universal and less interesting
        if recent_actions:
            action_parts = []
            for role in ROLES:
                rates = avg_actions_by_role[role]
                role_char = role[0].upper()
                # Show prim/sec/tert as percentages (0-100%)
                prim_pct = rates["primary"] * 100
                sec_pct = rates["secondary"] * 100
                tert_pct = rates["tertiary"] * 100
                action_parts.append(f"{role_char}:p{prim_pct:.0f}/s{sec_pct:.0f}/t{tert_pct:.0f}")
            print(f"       actions: {' | '.join(action_parts)}")

        # Evaluation
        eval_stats: dict[str, Any] | None = None
        if args.eval_every > 0 and update % args.eval_every == 0:
            if eval_seeds is not None:
                seeds_now = list(eval_seeds)
                seed0 = int(seeds_now[0]) if seeds_now else args.seed
                eval_episodes = len(seeds_now)
            else:
                seed0 = args.seed + 10_000 + update * 13
                eval_episodes = args.eval_episodes
                seeds_now = [seed0 + i for i in range(eval_episodes)]

            print(f"running eval ({eval_episodes} episodes)...", flush=True)
            eval_stats_obj, replay = evaluate_vs_heuristic(
                model=model,
                env_cfg=env_cfg,
                episodes=eval_episodes,
                seeds=seeds_now,
                device=device,
                save_replay=(bool(args.push_url) or bool(args.save_eval_replay)),
            )
            eval_stats = {
                "episodes": eval_stats_obj.episodes,
                "win_rate": eval_stats_obj.win_rate,
                "mean_hp_margin": eval_stats_obj.mean_hp_margin,
                "mean_episode_length": eval_stats_obj.mean_episode_length,
                "seeds": seeds_now,
            }
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
                    "git": dict(git_info(ROOT)),
                    "env_signature": env_signature(env_cfg),
                }
                if args.save_eval_replay:
                    replay_dir = run_dir / "replays"
                    replay_dir.mkdir(parents=True, exist_ok=True)
                    out_path = replay_dir / f"eval_update_{update:05d}_seed_{seed0}.json"
                    out_path.write_text(json.dumps(replay), encoding="utf-8")
                    print(f"saved eval replay: {out_path}")
                    if wandb_run and args.wandb_artifacts == "all":
                        import wandb

                        artifact = wandb.Artifact(
                            name=f"replay-{wandb_run.id}-update-{update:05d}",
                            type="replay",
                            metadata={
                                "update": update,
                                "global_step": global_step,
                                "seed": seed0,
                                "win_rate": eval_stats["win_rate"],
                            },
                        )
                        artifact.add_file(str(out_path))
                        wandb_run.log_artifact(artifact)
                if args.push_url:
                    push_replay(args.push_url, replay)

        # Best model tracking
        best_updated = False
        best_snapshot: str | None = None
        if eval_stats is not None:
            win_rate: float = eval_stats["win_rate"]
            mean_hp_margin: float = eval_stats["mean_hp_margin"]
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
                    "optimizer_state": trainer.optimizer.state_dict(),
                    "return_rms": trainer.return_normalizer.rms.state_dict(),
                    "env_cfg": asdict(env_cfg),
                    "args": vars(args),
                    "eval": eval_stats,
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                }
                best_pt = run_dir / "best.pt"
                torch.save(best_ckpt, best_pt)
                best_snapshot = str(best_pt)
                best_json_path.write_text(
                    json.dumps(
                        {
                            "update": update,
                            "global_step": global_step,
                            "episodes": episodes,
                            "win_rate": win_rate,
                            "mean_hp_margin": mean_hp_margin,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"new best: win_rate={win_rate:.3f} mean_hp_margin={mean_hp_margin:.2f} @ update {update}"
                )

                if wandb_run and args.wandb_artifacts in {"best", "all"}:
                    import wandb

                    artifact = wandb.Artifact(
                        name=f"model-{wandb_run.id}",
                        type="model",
                        metadata={
                            "update": update,
                            "global_step": global_step,
                            "win_rate": win_rate,
                            "mean_hp_margin": mean_hp_margin,
                        },
                    )
                    artifact.add_file(str(best_pt))
                    wandb_run.log_artifact(artifact, aliases=["best", f"update_{update:05d}"])

        # Periodic checkpointing
        saved_ckpt: str | None = None
        if args.save_every > 0 and update % args.save_every == 0:
            ckpt = {
                "update": update,
                "global_step": global_step,
                "episodes": episodes,
                "model_state": model.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
                "return_rms": trainer.return_normalizer.rms.state_dict(),
                "env_cfg": asdict(env_cfg),
                "args": vars(args),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
            }
            p = run_dir / f"ckpt_{update:05d}.pt"
            torch.save(ckpt, p)
            saved_ckpt = str(p)

        # Write metrics
        metrics_f.write(
            json.dumps(
                {
                    "type": "update",
                    "time": time.time(),
                    "update": update,
                    "global_step": global_step,
                    "episodes": episodes,
                    "avg_return_10": avg_return,
                    "loss": loss,
                    "pg_loss": pg_loss,
                    "v_loss": v_loss,
                    "entropy": entropy_loss,
                    "grad_norm": grad_norm,
                    "sps": sps,
                    "eval": eval_stats,
                    "best_updated": best_updated,
                    "best_snapshot": best_snapshot,
                    "saved_ckpt": saved_ckpt,
                }
            )
            + "\n"
        )
        metrics_f.flush()

        # W&B logging
        if wandb_run is not None:
            import wandb

            wandb_metrics = {
                "train/update": update,
                "train/loss": loss,
                "train/pg_loss": pg_loss,
                "train/v_loss": v_loss,
                "train/entropy": entropy_loss,
                "train/grad_norm": grad_norm,
                "train/approx_kl": approx_kl,
                "train/clipfrac": clipfrac,
                "train/learning_rate": trainer.optimizer.param_groups[0]["lr"],
                "train/sps": sps,
                "train/global_step": global_step,
                "train/episodes": episodes,
                "train/avg_return": avg_return,
                "train/avg_len": avg_len,
                "train/win_rate_blue": win_rate_blue,
                "train/win_rate_red": win_rate_red,
                "policy/advantage_mean": adv_mean,
                "policy/advantage_std": adv_std,
                "policy/value_mean": val_mean,
                "policy/value_std": val_std,
                "policy/explained_variance": explained_var,
                "policy/action_mean_norm": float(np.linalg.norm(action_mean)),
                "policy/action_std_mean": float(action_std.mean()),
                "policy/saturation_rate": saturation_rate,
            }
            wandb_metrics.update(
                {
                    "returns/mean": avg_return,
                    "returns/median": return_median,
                    "returns/p10": return_p10,
                    "returns/p90": return_p90,
                    "returns/std": return_std,
                }
            )
            # Add per-role rewards
            for role in ROLES:
                wandb_metrics[f"train/reward_{role}"] = avg_by_role[role]
            # Add per-component rewards and percentages
            for c in COMPONENTS:
                wandb_metrics[f"reward/{c}"] = avg_by_comp[c]
                wandb_metrics[f"reward_pct/{c}"] = comp_pct[c]
            # Add per-role action activation rates
            for role in ROLES:
                for slot in ACTION_SLOTS:
                    wandb_metrics[f"actions/{role}/{slot}"] = avg_actions_by_role[role][slot]
            # Add opponent curriculum metrics
            if args.train_mode == "heuristic":
                wandb_metrics["curriculum/opfor_weapon_prob"] = current_weapon_prob
            # Add combat statistics
            wandb_metrics.update(
                {
                    "combat/damage_dealt": avg_damage_blue,
                    "combat/damage_ratio": damage_ratio,
                    "combat/kill_participation": kill_participation,
                    "combat/survival_rate": survival_rate,
                }
            )
            # Add zone control metrics
            wandb_metrics.update(
                {
                    "zone/control_margin": zone_margin,
                    "zone/contested_ratio": contested_ratio,
                    "zone/time_to_entry": time_to_zone,
                }
            )
            # Coordination metrics - only compute every 10 updates
            if update % 10 == 0:
                recent_coord = episodic_coord_stats[-window:]
                if recent_coord:
                    avg_dispersion = float(
                        np.mean(
                            [
                                s["pack_dispersion_sum"] / max(s["pack_dispersion_count"], 1)
                                for s in recent_coord
                            ]
                        )
                    )
                    avg_centroid_dist = float(
                        np.mean(
                            [
                                s["centroid_zone_dist_sum"] / max(s["centroid_zone_dist_count"], 1)
                                for s in recent_coord
                            ]
                        )
                    )
                else:
                    avg_dispersion = 0.0
                    avg_centroid_dist = 0.0

                wandb_metrics.update(
                    {
                        "coordination/pack_dispersion": avg_dispersion,
                        "coordination/centroid_zone_dist": avg_centroid_dist,
                    }
                )
            if eval_stats:
                wandb_metrics["eval/win_rate"] = eval_stats["win_rate"]
                wandb_metrics["eval/mean_hp_margin"] = eval_stats["mean_hp_margin"]
                wandb_metrics["eval/episodes"] = eval_stats["episodes"]
            wandb_run.log(wandb_metrics, step=global_step)

        # Early exit if target games reached
        if target_games is not None and episodes >= target_games:
            print(f"\n✓ Target reached: {episodes} games completed (target: {target_games})")
            break

    # ====================================================================
    # Arena Submission (if requested)
    # ====================================================================

    if args.train_mode == "arena" and args.arena_submit:
        # Save candidate checkpoint
        candidate_path = run_dir / "arena_candidate.pt"
        torch.save(
            {
                "update": end_update,
                "global_step": global_step,
                "episodes": episodes,
                "model_state": model.state_dict(),
                "optimizer_state": trainer.optimizer.state_dict(),
                "return_rms": trainer.return_normalizer.rms.state_dict(),
                "env_cfg": asdict(env_cfg),
                "args": vars(args),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
            },
            candidate_path,
        )

        # Load league and validate
        league = League.load(arena_league_path)
        sig = env_signature(env_cfg)
        if league.env_signature is None:
            league.env_signature = sig
        elif league.env_signature != sig:
            raise RuntimeError("training env_cfg does not match league env_signature")

        candidate_entry = league.upsert_checkpoint(candidate_path, kind="candidate")
        pool = league.top_commanders(args.arena_top_k) + league.recent_candidates(
            args.arena_candidate_k, exclude_id=candidate_entry.entry_id
        )
        pool = [e for e in pool if e.entry_id != candidate_entry.entry_id]
        if not pool:
            raise RuntimeError("no opponents available in league")

        candidate_model = arena_load_model(str(candidate_path))
        match_cfg = replace(env_cfg, record_replay=False)

        matches = args.arena_submit_matches
        rng = random.Random(args.seed * 1_000_000 + 54321)
        results: dict[str, list[GameResult]] = {}
        game_log: list[dict] = []

        seed0 = args.seed * 1_000_000 + 12345
        for i in range(matches):
            opp_entry = rng.choice(pool)
            opp_model = arena_cache.get(opp_entry.entry_id)
            if opp_model is None:
                opp_model = arena_load_model(opp_entry.ckpt_path)
                arena_cache.put(opp_entry.entry_id, opp_model)

            candidate_as_blue = (i % 2) == 0
            match_seed = seed0 + i * 9973
            if candidate_as_blue:
                out = play_match(
                    env_cfg=match_cfg,
                    blue_policy=candidate_model,
                    red_policy=opp_model,
                    seed=match_seed,
                    device=device,
                )
                cand_team = "blue"
            else:
                out = play_match(
                    env_cfg=match_cfg,
                    blue_policy=opp_model,
                    red_policy=candidate_model,
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
                    "seed": match_seed,
                    "candidate_as": cand_team,
                    "opponent": opp_entry.entry_id,
                    "opponent_name": opp_entry.commander_name,
                    "winner": out.winner,
                    "candidate_score": cand_score,
                    "hp": out.hp,
                }
            )
            print(
                f"arena_submit match {i + 1}/{matches}: opp={opp_entry.entry_id} cand_as={cand_team} "
                f"winner={out.winner} score={cand_score}"
            )

        league.apply_rating_period(results)
        promoted = league.promote_if_topk(candidate_entry.entry_id, top_k=args.arena_top_k)
        league.save(arena_league_path)

        cand_post = league.entries[candidate_entry.entry_id]
        print(
            f"arena_submit candidate rating: {cand_post.rating.rating:.1f}±{cand_post.rating.rd:.1f} "
            f"vol={cand_post.rating.vol:.4f} games={cand_post.games} promoted={promoted} "
            f"name={cand_post.commander_name}"
        )

        if wandb_run is not None:
            import wandb

            wandb_run.log(
                {
                    "arena/rating": cand_post.rating.rating,
                    "arena/rd": cand_post.rating.rd,
                    "arena/games": cand_post.games,
                    "arena/promoted": int(promoted),
                },
                step=global_step,
            )

        if args.arena_submit_log:
            log_path = Path(args.arena_submit_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                json.dumps({"candidate": cand_post.as_dict(), "games": game_log}, indent=2), encoding="utf-8"
            )
            print(f"arena_submit wrote log: {log_path}")

    # Cleanup
    venv.close()  # Properly shut down worker processes
    metrics_f.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
