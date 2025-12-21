# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echelon.env.env import default_mech_classes
from echelon.config import WorldConfig
from echelon.gen.objective import capture_zone_params, clear_capture_zone, sample_capture_zone
from echelon.gen.corridors import carve_macro_corridors
from echelon.gen.recipe import build_recipe
from echelon.gen.transforms import apply_transform_voxels, list_transforms, opposite_corner, transform_corner
from echelon.gen.validator import ConnectivityValidator
from echelon.sim.world import VoxelWorld


def compute_density(solids: np.ndarray, clearance_z: int, *, meta: dict) -> dict[str, float]:
    z = int(max(1, min(clearance_z, solids.shape[0])))
    footprint = np.any(solids[:z, :, :], axis=0)
    density_total = float(np.mean(footprint))

    sy, sx = int(solids.shape[1]), int(solids.shape[2])
    cx, cy, r = capture_zone_params(meta, size_x=sx, size_y=sy)
    r = max(0.0, float(r))
    x0 = max(0, int(np.floor(cx - r)))
    x1 = min(sx, int(np.ceil(cx + r + 1)))
    y0 = max(0, int(np.floor(cy - r)))
    y1 = min(sy, int(np.ceil(cy + r + 1)))

    sub = footprint[y0:y1, x0:x1]
    if sub.size:
        xs = np.arange(x0, x1, dtype=np.float32)[None, :]
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        mask = (xs - float(cx)) ** 2 + (ys - float(cy)) ** 2 <= float(r) ** 2
        obj = sub[mask]
        density_obj = float(np.mean(obj)) if obj.size else 0.0
    else:
        density_obj = 0.0
    return {"total": density_total, "objective": density_obj}


def badness_score(entry: dict) -> float:
    # Higher is worse. Keep simple and tunable.
    score = 0.0
    score += float(entry.get("fixups_count", 0)) * 5.0
    score += float(entry.get("carved_ratio", 0.0)) * 500.0

    dens = entry.get("density", {})
    dt = float(dens.get("total", 0.0))
    if dt < 0.08:
        score += (0.08 - dt) * 200.0
    if dt > 0.35:
        score += (dt - 0.35) * 200.0

    dobj = float(dens.get("objective", 0.0))
    if dobj > 0.20:
        score += (dobj - 0.20) * 300.0

    paths = entry.get("paths", {})
    for key in ("blue_to_objective", "red_to_objective"):
        p = paths.get(key, {})
        if not p.get("found", False):
            score += 10_000.0
            continue
        overlap = p.get("overlap")
        if overlap is not None and float(overlap) > 0.65:
            score += (float(overlap) - 0.65) * 200.0
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100, help="World size (x=y), z fixed by --z")
    parser.add_argument("--z", type=int, default=20)
    parser.add_argument("--packs-per-team", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/terrain_audit.json")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    world_cfg = WorldConfig(size_x=args.size, size_y=args.size, size_z=args.z)
    mech_classes = default_mech_classes()
    max_hs_x = max(float(spec.size_voxels[0] * 0.5) for spec in mech_classes.values())
    max_hs_y = max(float(spec.size_voxels[1] * 0.5) for spec in mech_classes.values())
    cols = 5
    n_per_team = max(1, int(args.packs_per_team) * 10)
    cols_used = min(cols, n_per_team)
    rows = int((n_per_team + cols - 1) // cols)
    required_x = 2.0 + float(max(0, cols_used - 1)) * 3.5 + 2.0 * max_hs_x
    required_y = 2.0 + float(max(0, rows - 1)) * 3.5 + 2.0 * max_hs_y
    required_clear = int(np.ceil(max(required_x, required_y) + 1.0))
    spawn_clear = max(25, int(min(world_cfg.size_x, world_cfg.size_y) * 0.25), required_clear)
    spawn_clear = min(spawn_clear, world_cfg.size_x, world_cfg.size_y)

    validator = ConnectivityValidator(
        (world_cfg.size_z, world_cfg.size_y, world_cfg.size_x),
        clearance_z=world_cfg.connectivity_clearance_z,
        obstacle_inflate_radius=world_cfg.connectivity_obstacle_inflate_radius,
        wall_cost=world_cfg.connectivity_wall_cost,
        penalty_radius=world_cfg.connectivity_penalty_radius,
        penalty_cost=world_cfg.connectivity_penalty_cost,
        carve_width=world_cfg.connectivity_carve_width,
    )

    entries: list[dict] = []
    t0 = time.time()
    for i in range(args.seeds):
        seed = args.seed_start + i
        seq = np.random.SeedSequence(int(seed))
        child = seq.spawn(3)
        rng_world = np.random.default_rng(child[0])
        rng_variants = np.random.default_rng(child[2])
        world = VoxelWorld.generate(world_cfg, rng_world)

        blue_canon = str(rng_variants.choice(["BL", "BR", "TL", "TR"]))
        red_canon = opposite_corner(blue_canon)
        transform = str(rng_variants.choice(list_transforms()))
        world.voxels = apply_transform_voxels(world.voxels, transform)
        spawn_corners = {"blue": transform_corner(blue_canon, transform), "red": transform_corner(red_canon, transform)}

        # Apply spawn clears (matching env defaults today).
        if spawn_corners["blue"] == "BL":
            world.set_box_solid(0, 0, 0, spawn_clear, spawn_clear, world.size_z, False)
        elif spawn_corners["blue"] == "BR":
            world.set_box_solid(world.size_x - spawn_clear, 0, 0, world.size_x, spawn_clear, world.size_z, False)
        elif spawn_corners["blue"] == "TL":
            world.set_box_solid(0, world.size_y - spawn_clear, 0, spawn_clear, world.size_y, world.size_z, False)
        elif spawn_corners["blue"] == "TR":
            world.set_box_solid(
                world.size_x - spawn_clear, world.size_y - spawn_clear, 0, world.size_x, world.size_y, world.size_z, False
            )
        else:
            raise ValueError(spawn_corners["blue"])

        if spawn_corners["red"] == "BL":
            world.set_box_solid(0, 0, 0, spawn_clear, spawn_clear, world.size_z, False)
        elif spawn_corners["red"] == "BR":
            world.set_box_solid(world.size_x - spawn_clear, 0, 0, world.size_x, spawn_clear, world.size_z, False)
        elif spawn_corners["red"] == "TL":
            world.set_box_solid(0, world.size_y - spawn_clear, 0, spawn_clear, world.size_y, world.size_z, False)
        elif spawn_corners["red"] == "TR":
            world.set_box_solid(
                world.size_x - spawn_clear, world.size_y - spawn_clear, 0, world.size_x, world.size_y, world.size_z, False
            )
        else:
            raise ValueError(spawn_corners["red"])

        solids_before = world.solid.copy()
        world.meta["spawn_clear"] = int(spawn_clear)
        world.meta["spawn_corners"] = dict(spawn_corners)
        world.meta["transform"] = transform
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

        world.voxels = validator.validate_and_fix(
            world.voxels,
            spawn_corners=spawn_corners,
            spawn_clear=spawn_clear,
            meta=world.meta,
        )

        carved = int(np.count_nonzero(np.logical_and(solids_before, np.logical_not(world.solid))))
        total_vox = int(world.solid.size)
        carved_ratio = float(carved / max(1, total_vox))

        dens = compute_density(world.solid, clearance_z=world_cfg.connectivity_clearance_z, meta=world.meta)
        recipe = build_recipe(
            generator_id="legacy_voxel_archetypes",
            generator_version="1",
            seed=seed,
            world_config=world_cfg,
            world_meta=world.meta,
            solids_zyx=world.voxels,
        )

        paths = dict(world.meta.get("stats", {}).get("paths", {}))
        entry = {
            "seed": int(seed),
            "size": [int(world.size_x), int(world.size_y), int(world.size_z)],
            "archetype": world.meta.get("archetype_name", world.meta.get("archetype")),
            "transform": transform,
            "spawn_corners": dict(spawn_corners),
            "density": dens,
            "walls": int(np.count_nonzero(world.solid)),
            "carved_voxels": carved,
            "carved_ratio": carved_ratio,
            "fixups_count": int(len(world.meta.get("fixups", []))),
            "paths": paths,
            "hashes": recipe.get("hashes", {}),
        }
        entry["badness"] = badness_score(entry)
        entries.append(entry)

    elapsed = time.time() - t0
    entries_sorted = sorted(entries, key=lambda e: float(e.get("badness", 0.0)), reverse=True)
    worst = entries_sorted[: max(1, int(args.topk))]

    summary = {
        "world_config": asdict(world_cfg),
        "spawn_clear": int(spawn_clear),
        "packs_per_team": int(args.packs_per_team),
        "count": int(len(entries)),
        "elapsed_s": float(elapsed),
        "badness_max": float(worst[0]["badness"]) if worst else 0.0,
        "badness_p95": float(np.percentile([e["badness"] for e in entries], 95)) if entries else 0.0,
        "fixups_mean": float(np.mean([e["fixups_count"] for e in entries])) if entries else 0.0,
        "carved_ratio_mean": float(np.mean([e["carved_ratio"] for e in entries])) if entries else 0.0,
        "density_total_mean": float(np.mean([e["density"]["total"] for e in entries])) if entries else 0.0,
    }

    out = {"summary": summary, "worst": worst}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path} (worst={len(worst)})")


if __name__ == "__main__":
    main()
