# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echelon.config import WorldConfig
from echelon.gen.objective import clear_capture_zone
from echelon.gen.recipe import build_recipe
from echelon.gen.transforms import apply_transform_voxels
from echelon.gen.validator import ConnectivityValidator
from echelon.sim.world import VoxelWorld


def clear_spawn_corner(world: VoxelWorld, *, corner: str, spawn_clear: int) -> None:
    if corner == "BL":
        world.set_box_solid(0, 0, 0, spawn_clear, spawn_clear, world.size_z, False)
    elif corner == "BR":
        world.set_box_solid(world.size_x - spawn_clear, 0, 0, world.size_x, spawn_clear, world.size_z, False)
    elif corner == "TL":
        world.set_box_solid(0, world.size_y - spawn_clear, 0, spawn_clear, world.size_y, world.size_z, False)
    elif corner == "TR":
        world.set_box_solid(world.size_x - spawn_clear, world.size_y - spawn_clear, 0, world.size_x, world.size_y, world.size_z, False)
    else:
        raise ValueError(f"Unknown corner: {corner!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=str)
    args = parser.parse_args()

    replay_path = Path(args.replay)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))

    world = replay.get("world", {})
    meta = world.get("meta", {}) or {}
    recipe = meta.get("recipe")
    if recipe is None:
        raise SystemExit("Replay has no `world.meta.recipe`; cannot reproduce.")

    seed = recipe.get("seed")
    if seed is None:
        raise SystemExit("Recipe has no `seed`; cannot reproduce.")
    seed = int(seed)

    gen = recipe.get("generator", {})
    generator_id = str(gen.get("id", "legacy_voxel_archetypes"))
    generator_version = str(gen.get("version", "1"))
    cfg_dict = gen.get("config")
    if not isinstance(cfg_dict, dict):
        raise SystemExit("Recipe has no `generator.config`; cannot reproduce.")
    world_cfg = WorldConfig(**cfg_dict)

    variant = recipe.get("variant", {}) or {}
    transform = str(variant.get("transform", "identity"))
    spawn_corners = dict(variant.get("spawn", {"blue": "BL", "red": "TR"}))

    regions = recipe.get("regions", {}) or {}
    objective = regions.get("objective", {}) or {}
    spawns = regions.get("spawns", {}) or {}
    spawn_clear = int((spawns.get("blue") or {}).get("spawn_clear", 0))
    if spawn_clear <= 0:
        spawn_clear = int(world.get("spawn_clear", 0)) or max(25, int(min(world_cfg.size_x, world_cfg.size_y) * 0.25))
    spawn_clear = min(spawn_clear, world_cfg.size_x, world_cfg.size_y)

    seq = np.random.SeedSequence(seed)
    rng_world = np.random.default_rng(seq.spawn(1)[0])

    repro_world = VoxelWorld.generate(world_cfg, rng_world)
    repro_world.voxels = apply_transform_voxels(repro_world.voxels, transform)
    repro_world.meta["transform"] = transform
    repro_world.meta["spawn_corners"] = dict(spawn_corners)
    repro_world.meta["spawn_clear"] = int(spawn_clear)

    obj_center = objective.get("center")
    obj_radius = objective.get("radius")
    if (
        str(objective.get("shape", "circle")) != "circle"
        or not isinstance(obj_center, (list, tuple))
        or len(obj_center) < 2
        or not isinstance(obj_center[0], (int, float))
        or not isinstance(obj_center[1], (int, float))
        or not isinstance(obj_radius, (int, float))
    ):
        raise SystemExit("Recipe has no `regions.objective` circle; cannot reproduce.")
    repro_world.meta["capture_zone"] = {"center": [float(obj_center[0]), float(obj_center[1])], "radius": float(obj_radius)}

    clear_spawn_corner(repro_world, corner=spawn_corners.get("blue", "BL"), spawn_clear=spawn_clear)
    clear_spawn_corner(repro_world, corner=spawn_corners.get("red", "TR"), spawn_clear=spawn_clear)
    clear_capture_zone(repro_world, meta=repro_world.meta)

    if world_cfg.ensure_connectivity:
        validator = ConnectivityValidator(
            (repro_world.size_z, repro_world.size_y, repro_world.size_x),
            clearance_z=world_cfg.connectivity_clearance_z,
            obstacle_inflate_radius=world_cfg.connectivity_obstacle_inflate_radius,
            wall_cost=world_cfg.connectivity_wall_cost,
            penalty_radius=world_cfg.connectivity_penalty_radius,
            penalty_cost=world_cfg.connectivity_penalty_cost,
            carve_width=world_cfg.connectivity_carve_width,
        )
        repro_world.voxels = validator.validate_and_fix(
            repro_world.voxels,
            spawn_corners=spawn_corners,
            spawn_clear=spawn_clear,
            meta=repro_world.meta,
        )

    repro_recipe = build_recipe(
        generator_id=generator_id,
        generator_version=generator_version,
        seed=seed,
        world_config=world_cfg,
        world_meta=repro_world.meta,
        solids_zyx=repro_world.voxels,
    )

    expected_hashes = recipe.get("hashes", {}) or {}
    exp_solid = expected_hashes.get("solid")
    exp_recipe = expected_hashes.get("recipe")

    got_solid = repro_recipe.get("hashes", {}).get("solid")
    got_recipe = repro_recipe.get("hashes", {}).get("recipe")

    ok = True
    if exp_solid is not None and exp_solid != got_solid:
        ok = False
        print(f"solid hash mismatch: expected={exp_solid} got={got_solid}")
    if exp_recipe is not None and exp_recipe != got_recipe:
        ok = False
        print(f"recipe hash mismatch: expected={exp_recipe} got={got_recipe}")

    if ok:
        print("ok: reproduced hashes match")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
