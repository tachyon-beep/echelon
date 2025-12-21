from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from typing import Any

import numpy as np

from ..config import WorldConfig
from .objective import capture_zone_params


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_solids(solids_zyx: np.ndarray) -> str:
    arr = np.asarray(solids_zyx, dtype=np.uint8).ravel(order="C")
    packed = np.packbits(arr)
    return sha256_hex(packed.tobytes())


def build_recipe(
    *,
    generator_id: str,
    generator_version: str,
    seed: int | None,
    world_config: WorldConfig,
    world_meta: dict[str, Any],
    solids_zyx: np.ndarray,
) -> dict[str, Any]:
    sx, sy, sz = int(solids_zyx.shape[2]), int(solids_zyx.shape[1]), int(solids_zyx.shape[0])
    spawn_clear = int(world_meta.get("spawn_clear", max(25, int(sx * 0.25))))
    spawn_corners = dict(world_meta.get("spawn_corners", {"blue": "BL", "red": "TR"}))
    obj_x, obj_y, obj_r = capture_zone_params(world_meta, size_x=sx, size_y=sy)

    recipe: dict[str, Any] = {
        "schema_version": 2,
        "generator": {
            "id": str(generator_id),
            "version": str(generator_version),
            "config": dataclasses.asdict(world_config),
        },
        "seed": int(seed) if seed is not None else None,
        "variant": {
            "transform": str(world_meta.get("transform", "identity")),
            "biome_perm": world_meta.get("biome_perm"),
            "spawn": spawn_corners,
        },
        "regions": {
            "objective": {"shape": "circle", "center": [float(obj_x), float(obj_y)], "radius": float(obj_r)},
            "spawns": {
                "blue": {"corner": spawn_corners.get("blue", "BL"), "spawn_clear": spawn_clear},
                "red": {"corner": spawn_corners.get("red", "TR"), "spawn_clear": spawn_clear},
            },
        },
        "stats": copy.deepcopy(world_meta.get("stats", {})),
        "fixups": list(world_meta.get("fixups", [])),
        "hashes": {},
    }

    # Avoid hashing the hash fields themselves.
    recipe_for_hash = copy.deepcopy(recipe)
    recipe_for_hash.pop("hashes", None)
    recipe_hash = sha256_hex(canonical_json_bytes(recipe_for_hash))
    solid_hash = hash_solids(solids_zyx)
    recipe["hashes"] = {"recipe": recipe_hash, "solid": solid_hash}
    return recipe
