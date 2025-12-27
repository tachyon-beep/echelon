"""Vectorized environment for parallel experience collection.

This module provides multiprocessing-based vectorization for EchelonEnv,
enabling efficient parallel rollout collection for PPO training.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from echelon import EnvConfig


def _make_env_thunk(env_cfg: EnvConfig):
    """Thunk for lazy environment creation in worker process."""
    from echelon import EchelonEnv

    return EchelonEnv(env_cfg)


def _env_worker(remote: Connection, env_fn, env_cfg: EnvConfig, initial_weapon_prob: float = 0.5) -> None:
    """Worker process that runs a single environment.

    Protocol:
        ("step", actions_dict) -> (obs, reward, term, trunc, info)
        ("reset", seed) -> (obs, info)
        ("get_team_alive", None) -> {"blue": bool, "red": bool}
        ("get_last_outcome", None) -> dict | None
        ("get_heuristic_actions", agent_ids) -> {agent_id: action}
        ("set_heuristic_weapon_prob", float) -> None (updates weapon fire probability)
        ("set_curriculum", dict) -> None (updates curriculum params)
        ("get_curriculum", None) -> dict (returns current curriculum)
        ("close", None) -> exits

    Args:
        remote: Parent connection pipe for communication
        env_fn: Environment factory function
        env_cfg: Environment configuration
        initial_weapon_prob: Initial weapon fire probability for heuristic (curriculum)
    """
    try:
        import random
        from dataclasses import replace

        from echelon.agents.heuristic import HeuristicPolicy

        env = env_fn(env_cfg)
        heuristic = HeuristicPolicy(weapon_fire_prob=initial_weapon_prob)

        # Store base config for curriculum modifications
        base_env_cfg = env_cfg

        # Curriculum state
        curriculum: dict = {
            "weapon_prob": initial_weapon_prob,
            "map_size_range": (env_cfg.world.size_x, env_cfg.world.size_x),  # Default: fixed size
        }

        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, term, trunc, info = env.step(data)
                remote.send((obs, reward, term, trunc, info))
            elif cmd == "reset":
                # Apply curriculum: randomize map size within range
                min_size, max_size = curriculum["map_size_range"]
                if min_size != max_size:
                    new_size = random.randint(min_size, max_size)
                    current_size = env.config.world.size_x
                    # Only recreate env if size actually changed
                    if new_size != current_size:
                        new_world = replace(
                            base_env_cfg.world,
                            size_x=new_size,
                            size_y=new_size,
                        )
                        new_cfg = replace(base_env_cfg, world=new_world)
                        # Close old env to prevent resource leak (P1 fix)
                        env.close()
                        env = env_fn(new_cfg)
                obs, info = env.reset(seed=data)
                remote.send((obs, info))
            elif cmd == "get_team_alive":
                remote.send({team: env.sim.team_alive(team) for team in ("blue", "red")})
            elif cmd == "get_last_outcome":
                remote.send(env.last_outcome)
            elif cmd == "get_heuristic_actions":
                res = {rid: heuristic.act(env, rid) for rid in data}
                remote.send(res)
            elif cmd == "set_heuristic_weapon_prob":
                heuristic.weapon_fire_prob = float(data)
                remote.send(None)
            elif cmd == "set_curriculum":
                curriculum.update(data)
                if "weapon_prob" in data:
                    heuristic.weapon_fire_prob = data["weapon_prob"]
                remote.send(None)
            elif cmd == "get_curriculum":
                remote.send(dict(curriculum))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except Exception:
        print(f"Worker process encountered error:\n{traceback.format_exc()}")
        remote.close()


class VectorEnv:
    """Vectorized environment using multiprocessing.

    Each env runs in a separate process to avoid GIL contention
    and CUDA fork issues (uses spawn context).

    Usage:
        vec_env = VectorEnv(num_envs=4, env_cfg=cfg)
        obs_list, infos = vec_env.reset(seeds=[0, 1, 2, 3])
        obs_list, rewards, terms, truncs, infos = vec_env.step(actions_list)
        vec_env.close()

        # Or use as context manager:
        with VectorEnv(num_envs=4, env_cfg=cfg) as vec_env:
            obs_list, infos = vec_env.reset(seeds=[0, 1, 2, 3])
            ...
    """

    def __init__(self, num_envs: int, env_cfg: EnvConfig, initial_weapon_prob: float = 0.5):
        """Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments
            env_cfg: Environment configuration (shared across all envs)
            initial_weapon_prob: Initial weapon fire probability for heuristic opponent
        """
        self.num_envs = num_envs
        # Use spawn to avoid CUDA fork issues
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)], strict=False)
        self.ps = [
            ctx.Process(target=_env_worker, args=(work_remote, _make_env_thunk, env_cfg, initial_weapon_prob))
            for work_remote in self.work_remotes
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(
        self, actions_list: list[dict]
    ) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
        """Step all environments in parallel.

        Args:
            actions_list: List of action dicts, one per environment

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos),
            each as a list with one entry per environment
        """
        for remote, action in zip(self.remotes, actions_list, strict=False):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, terms, truncs, infos = zip(*results, strict=False)
        return list(obs), list(rews), list(terms), list(truncs), list(infos)

    def reset(self, seeds: list[int], indices: list[int] | None = None) -> tuple[list[dict], list[dict]]:
        """Reset environments.

        Args:
            seeds: List of seeds, one per environment to reset
            indices: Optional list of environment indices to reset.
                     If None, resets first len(seeds) environments.

        Returns:
            Tuple of (observations, infos) as lists
        """
        if indices is None:
            indices = list(range(len(seeds)))
        for i, seed in zip(indices, seeds, strict=False):
            self.remotes[i].send(("reset", seed))
        results = [self.remotes[i].recv() for i in indices]
        obs, infos = zip(*results, strict=False)
        return list(obs), list(infos)

    def get_team_alive(self) -> list[dict[str, bool]]:
        """Get team alive status from all environments.

        Returns:
            List of dicts with {"blue": bool, "red": bool} per environment
        """
        for remote in self.remotes:
            remote.send(("get_team_alive", None))
        return [remote.recv() for remote in self.remotes]

    def get_last_outcomes(self) -> list[dict | None]:
        """Get last outcome from all environments.

        Returns:
            List of outcome dicts (or None) per environment
        """
        for remote in self.remotes:
            remote.send(("get_last_outcome", None))
        return [remote.recv() for remote in self.remotes]

    def get_heuristic_actions(self, red_ids: list[str]) -> list[dict]:
        """Get heuristic actions for red team from all environments.

        Args:
            red_ids: List of red agent IDs

        Returns:
            List of action dicts, one per environment
        """
        for remote in self.remotes:
            remote.send(("get_heuristic_actions", red_ids))
        return [remote.recv() for remote in self.remotes]

    def set_heuristic_weapon_prob(self, prob: float) -> None:
        """Update heuristic weapon fire probability (for curriculum learning).

        Args:
            prob: New weapon fire probability (0.0 to 1.0)
        """
        for remote in self.remotes:
            remote.send(("set_heuristic_weapon_prob", prob))
        for remote in self.remotes:
            remote.recv()  # Wait for acknowledgment

    def set_curriculum(
        self,
        weapon_prob: float | None = None,
        map_size_range: tuple[int, int] | None = None,
    ) -> None:
        """Update curriculum parameters across all environments.

        Args:
            weapon_prob: Heuristic weapon fire probability (0.0 to 1.0)
            map_size_range: (min_size, max_size) for random map sizes on reset
        """
        update: dict = {}
        if weapon_prob is not None:
            update["weapon_prob"] = weapon_prob
        if map_size_range is not None:
            min_size, max_size = map_size_range
            if min_size > max_size:
                raise ValueError(f"map_size_range min ({min_size}) must be <= max ({max_size})")
            update["map_size_range"] = map_size_range

        if update:
            for remote in self.remotes:
                remote.send(("set_curriculum", update))
            for remote in self.remotes:
                remote.recv()

    def get_curriculum(self) -> dict:
        """Get current curriculum parameters from first environment.

        Returns:
            Dict with weapon_prob, map_size_range
        """
        self.remotes[0].send(("get_curriculum", None))
        result: dict = self.remotes[0].recv()
        return result

    def close(self) -> None:
        """Close all environments and terminate worker processes.

        Sends close command to all workers and waits for them to terminate.
        """
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()

    def __enter__(self) -> VectorEnv:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()
