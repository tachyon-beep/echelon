# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echelon import EchelonEnv, EnvConfig, WorldConfig
from echelon.actions import ActionIndex


def check_speed_cap(env: EchelonEnv, seed: int) -> None:
    env.reset(seed=seed)
    # [fwd, strafe, vert, yaw, laser, vent, missile, paint, kinetic]
    actions = {}
    for aid in env.agents:
        a = np.zeros(env.ACTION_DIM, dtype=np.float32)
        a[ActionIndex.FORWARD] = 1.0
        a[ActionIndex.STRAFE] = 1.0
        actions[aid] = a
    env.step(actions)

    assert env.sim is not None
    for aid in env.agents:
        mech = env.sim.mechs[aid]
        if not mech.alive:
            continue
        speed_xy = float(np.linalg.norm(mech.vel[:2]))
        max_speed = float(mech.spec.max_speed)
        assert speed_xy <= max_speed + 1e-5, f"{aid}: speed_xy={speed_xy:.4f} > max_speed={max_speed:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-per-team", type=int, default=2)
    parser.add_argument("--size", type=int, default=20)
    args = parser.parse_args()

    env = EchelonEnv(
        EnvConfig(
            world=WorldConfig(size_x=args.size, size_y=args.size, size_z=args.size),
            num_packs=max(1, args.num_per_team // 10),  # Pack size is now 10
            seed=args.seed,
        )
    )
    check_speed_cap(env, seed=args.seed)
    print("ok")


if __name__ == "__main__":
    main()
