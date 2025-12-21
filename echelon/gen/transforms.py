from __future__ import annotations

from typing import Literal

import numpy as np

TransformType = Literal["identity", "flip_x", "flip_y", "flip_xy"]


def list_transforms() -> tuple[TransformType, ...]:
    return ("identity", "flip_x", "flip_y", "flip_xy")


def apply_transform_grid(grid_zyx: np.ndarray, transform: TransformType) -> np.ndarray:
    if transform == "identity":
        return grid_zyx
    if transform == "flip_x":
        return np.ascontiguousarray(np.flip(grid_zyx, axis=2))
    if transform == "flip_y":
        return np.ascontiguousarray(np.flip(grid_zyx, axis=1))
    if transform == "flip_xy":
        return np.ascontiguousarray(np.flip(grid_zyx, axis=(1, 2)))
    raise ValueError(f"Unknown transform: {transform!r}")


def apply_transform_voxels(voxels_zyx: np.ndarray, transform: TransformType) -> np.ndarray:
    return apply_transform_grid(voxels_zyx, transform)


def apply_transform_solids(solids_zyx: np.ndarray, transform: TransformType) -> np.ndarray:
    return apply_transform_grid(solids_zyx, transform)


def transform_corner(corner: str, transform: TransformType) -> str:
    if transform == "identity":
        mapping = {"BL": "BL", "BR": "BR", "TL": "TL", "TR": "TR"}
    elif transform == "flip_x":
        mapping = {"BL": "BR", "BR": "BL", "TL": "TR", "TR": "TL"}
    elif transform == "flip_y":
        mapping = {"BL": "TL", "TL": "BL", "BR": "TR", "TR": "BR"}
    elif transform == "flip_xy":
        mapping = {"BL": "TR", "TR": "BL", "BR": "TL", "TL": "BR"}
    else:
        raise ValueError(f"Unknown transform: {transform!r}")

    out = mapping.get(corner)
    if out is None:
        raise ValueError(f"Unknown corner: {corner!r}")
    return out


def opposite_corner(corner: str) -> str:
    mapping = {"BL": "TR", "TR": "BL", "BR": "TL", "TL": "BR"}
    out = mapping.get(corner)
    if out is None:
        raise ValueError(f"Unknown corner: {corner!r}")
    return out
