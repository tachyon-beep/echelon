# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echelon.config import WorldConfig
from echelon.env.env import default_mech_classes
from echelon.gen.corridors import carve_macro_corridors
from echelon.gen.objective import clear_capture_zone, sample_capture_zone
from echelon.gen.recipe import build_recipe
from echelon.gen.transforms import apply_transform_voxels, list_transforms, opposite_corner, transform_corner
from echelon.gen.validator import ConnectivityValidator
from echelon.sim.world import VoxelWorld


def _spawn_clear_for_packs(world_cfg: WorldConfig, packs_per_team: int) -> int:
    mech_classes = default_mech_classes()
    max_hs_x = max(float(spec.size_voxels[0] * 0.5) for spec in mech_classes.values())
    max_hs_y = max(float(spec.size_voxels[1] * 0.5) for spec in mech_classes.values())
    cols = 5
    n_per_team = max(1, int(packs_per_team) * 10)
    cols_used = min(cols, n_per_team)
    rows = int((n_per_team + cols - 1) // cols)
    required_x = 2.0 + float(max(0, cols_used - 1)) * 3.5 + 2.0 * max_hs_x
    required_y = 2.0 + float(max(0, rows - 1)) * 3.5 + 2.0 * max_hs_y
    required_clear = int(np.ceil(max(required_x, required_y) + 1.0))
    spawn_clear = max(25, int(min(world_cfg.size_x, world_cfg.size_y) * 0.25), required_clear)
    return min(spawn_clear, world_cfg.size_x, world_cfg.size_y)


def _clear_corner(world: VoxelWorld, corner: str, spawn_clear: int) -> None:
    if corner == "BL":
        world.set_box_solid(0, 0, 0, spawn_clear, spawn_clear, world.size_z, False)
    elif corner == "BR":
        world.set_box_solid(world.size_x - spawn_clear, 0, 0, world.size_x, spawn_clear, world.size_z, False)
    elif corner == "TL":
        world.set_box_solid(0, world.size_y - spawn_clear, 0, spawn_clear, world.size_y, world.size_z, False)
    elif corner == "TR":
        world.set_box_solid(
            world.size_x - spawn_clear,
            world.size_y - spawn_clear,
            0,
            world.size_x,
            world.size_y,
            world.size_z,
            False,
        )
    else:
        raise ValueError(f"Unknown corner: {corner!r}")


def generate_hashes(seed: int, world_cfg: WorldConfig, *, packs_per_team: int) -> dict:
    seq = np.random.SeedSequence(int(seed))
    child = seq.spawn(3)
    rng_world = np.random.default_rng(child[0])
    rng_variants = np.random.default_rng(child[2])

    world = VoxelWorld.generate(world_cfg, rng_world)

    blue_canon = str(rng_variants.choice(["BL", "BR", "TL", "TR"]))
    red_canon = opposite_corner(blue_canon)
    transform = str(rng_variants.choice(list_transforms()))
    world.voxels = apply_transform_voxels(world.voxels, transform)
    spawn_corners = {
        "blue": transform_corner(blue_canon, transform),
        "red": transform_corner(red_canon, transform),
    }

    spawn_clear = _spawn_clear_for_packs(world_cfg, packs_per_team)
    world.meta["transform"] = transform
    world.meta["spawn_corners"] = dict(spawn_corners)
    world.meta["spawn_clear"] = int(spawn_clear)

    _clear_corner(world, spawn_corners["blue"], spawn_clear)
    _clear_corner(world, spawn_corners["red"], spawn_clear)

    world.meta["capture_zone"] = sample_capture_zone(
        world, rng_variants, spawn_clear=spawn_clear, spawn_corners=spawn_corners
    )
    clear_capture_zone(world, meta=world.meta)

    carve_macro_corridors(
        world,
        spawn_corners=spawn_corners,
        spawn_clear=spawn_clear,
        meta=world.meta,
        rng=rng_variants,
    )

    if world_cfg.ensure_connectivity:
        validator = ConnectivityValidator(
            (world.size_z, world.size_y, world.size_x),
            clearance_z=world_cfg.connectivity_clearance_z,
            obstacle_inflate_radius=world_cfg.connectivity_obstacle_inflate_radius,
            wall_cost=world_cfg.connectivity_wall_cost,
            penalty_radius=world_cfg.connectivity_penalty_radius,
            penalty_cost=world_cfg.connectivity_penalty_cost,
            carve_width=world_cfg.connectivity_carve_width,
        )
        world.voxels = validator.validate_and_fix(
            world.voxels,
            spawn_corners=spawn_corners,
            spawn_clear=spawn_clear,
            meta=world.meta,
        )

    recipe = build_recipe(
        generator_id="legacy_voxel_archetypes",
        generator_version="1",
        seed=seed,
        world_config=world_cfg,
        world_meta=world.meta,
        solids_zyx=world.voxels,
    )
    return {
        "seed": int(seed),
        "transform": transform,
        "spawn_corners": dict(spawn_corners),
        "capture_zone": dict(world.meta.get("capture_zone", {})),
        "hashes": dict(recipe.get("hashes", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=str, default="runs/terrain_golden.json")
    parser.add_argument(
        "--write", action="store_true", help="Write/update the golden file instead of checking it"
    )
    parser.add_argument("--seeds", type=str, default="1,2,3,12345,99991")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--z", type=int, default=20)
    parser.add_argument("--packs-per-team", type=int, default=1)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("No seeds provided")

    world_cfg = WorldConfig(size_x=args.size, size_y=args.size, size_z=args.z)
    out_path = Path(args.golden)

    if not args.write and not out_path.exists():
        raise SystemExit(f"Golden file not found: {out_path} (run with --write to create)")

    generated = [generate_hashes(s, world_cfg, packs_per_team=args.packs_per_team) for s in seeds]

    payload = {
        "generator": {"id": "legacy_voxel_archetypes", "version": "1"},
        "world_config": asdict(world_cfg),
        "packs_per_team": int(args.packs_per_team),
        "entries": generated,
    }

    if args.write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
        return

    golden = json.loads(out_path.read_text(encoding="utf-8"))
    golden_entries = {int(e["seed"]): e for e in golden.get("entries", [])}

    ok = True
    for e in generated:
        seed = int(e["seed"])
        g = golden_entries.get(seed)
        if g is None:
            ok = False
            print(f"missing seed in golden: {seed}")
            continue

        exp = g.get("hashes") or {}
        got = e.get("hashes") or {}
        if exp.get("solid") != got.get("solid"):
            ok = False
            print(f"seed {seed}: solid hash mismatch expected={exp.get('solid')} got={got.get('solid')}")
        if exp.get("recipe") != got.get("recipe"):
            ok = False
            print(f"seed {seed}: recipe hash mismatch expected={exp.get('recipe')} got={got.get('recipe')}")

    if ok:
        print("ok: all golden hashes match")
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
