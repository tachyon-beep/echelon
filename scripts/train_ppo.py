# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echelon import EchelonEnv, EnvConfig, WorldConfig
from echelon.agents.heuristic import HeuristicPolicy
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
    record: bool = False,
) -> tuple[dict, dict | None]:
    # Set record_replay if requested
    env_cfg = replace(env_cfg, record_replay=record)
    env = EchelonEnv(env_cfg)
    heuristic = HeuristicPolicy()

    blue_ids = env.blue_ids
    red_ids = env.red_ids

    wins = {"blue": 0, "red": 0, "draw": 0}
    hp_margins: list[float] = []
    last_replay = None

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
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

    wins["episodes"] = episodes
    wins["win_rate"] = float(wins["blue"] / max(1, episodes))
    wins["mean_hp_margin"] = float(np.mean(hp_margins)) if hp_margins else 0.0
    return wins, last_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--packs-per-team", type=int, default=2, help="Number of 5-mech packs (1 Heavy, 2 Med, 2 Light) per team")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "partial"])
    parser.add_argument("--dt-sim", type=float, default=0.05)
    parser.add_argument("--decision-repeat", type=int, default=5)
    parser.add_argument("--episode-seconds", type=float, default=60.0)
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
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--run-dir", type=str, default="runs/train")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path, or 'latest' (from run-dir)")
    parser.add_argument("--push-url", type=str, default="http://localhost:8090/push", help="URL to POST replays to")
    args = parser.parse_args()

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
            record_replay=False,
            seed=args.seed,
        )

    env = EchelonEnv(env_cfg)
    heuristic = HeuristicPolicy()

    obs, _ = env.reset(seed=args.seed)
    blue_ids = env.blue_ids
    red_ids = env.red_ids

    obs_dim = int(next(iter(obs.values())).shape[0])
    action_dim = env.ACTION_DIM

    model = ActorCriticLSTM(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

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
                "run_dir": str(run_dir),
                "resume_path": str(resume_path) if resume_path is not None else None,
                "resume_update": int(resume_ckpt.get("update")) if resume_ckpt is not None else None,
                "start_update": int(start_update),
                "end_update": int(end_update),
                "global_step": int(global_step),
                "episodes": int(episodes),
                "args": vars(args),
                "env": asdict(env_cfg),
            }
        )
        + "\n"
    )
    metrics_f.flush()

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
            for rid in red_ids:
                action_dict[rid] = heuristic.act(env, rid)

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

            if ep_over:
                episodic_returns.append(current_ep_return)
                episodic_lengths.append(current_ep_len)
                
                outcome = env.last_outcome or {"winner": "unknown"}
                episodic_outcomes.append(outcome.get("winner", "unknown"))

                current_ep_return = 0.0
                current_ep_len = 0
                episodes += 1
                next_obs_dict, _ = env.reset(seed=args.seed + episodes)
                # Ensure LSTM resets at the start of the next step (episode boundary).
                next_done = torch.ones(batch_size, device=device)

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
            eval_stats, replay = evaluate_vs_heuristic(
                model=model,
                env_cfg=env_cfg,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000 + update * 13,
                device=device,
                record=(args.push_url is not None)
            )
            print(f"eval vs heuristic: {eval_stats}")
            if replay and args.push_url:
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


if __name__ == "__main__":
    main()
