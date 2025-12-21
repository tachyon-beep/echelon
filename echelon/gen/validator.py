from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import heapq

import numpy as np

from .objective import capture_zone_anchor


@dataclass(frozen=True)
class PathResult:
    path: list[tuple[int, int]]  # [(y, x), ...]
    cost: float
    digs_needed: int
    found: bool


def _inflate_obstacles_8(blocked: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return blocked
    out = blocked
    for _ in range(radius):
        m = out
        expanded = m.copy()
        expanded[:-1, :] |= m[1:, :]
        expanded[1:, :] |= m[:-1, :]
        expanded[:, :-1] |= m[:, 1:]
        expanded[:, 1:] |= m[:, :-1]
        expanded[:-1, :-1] |= m[1:, 1:]
        expanded[:-1, 1:] |= m[1:, :-1]
        expanded[1:, :-1] |= m[:-1, 1:]
        expanded[1:, 1:] |= m[:-1, :-1]
        out = expanded
    return out


def _corner_anchor(corner: str, spawn_clear: int, size_y: int, size_x: int) -> tuple[int, int]:
    half = max(1, int(spawn_clear // 2))
    if corner == "BL":
        return (half, half)
    if corner == "BR":
        return (half, max(0, size_x - 1 - half))
    if corner == "TL":
        return (max(0, size_y - 1 - half), half)
    if corner == "TR":
        return (max(0, size_y - 1 - half), max(0, size_x - 1 - half))
    raise ValueError(f"Unknown corner: {corner!r}")


class ConnectivityValidator:
    def __init__(
        self,
        world_shape_zyx: tuple[int, int, int],
        *,
        clearance_z: int = 4,
        obstacle_inflate_radius: int = 1,
        wall_cost: float = 80.0,
        penalty_radius: int = 3,
        penalty_cost: float = 20.0,
        carve_width: int = 5,
    ) -> None:
        self.size_z, self.size_y, self.size_x = (int(world_shape_zyx[0]), int(world_shape_zyx[1]), int(world_shape_zyx[2]))
        self.clearance_z = max(1, int(min(clearance_z, self.size_z)))
        self.obstacle_inflate_radius = max(0, int(obstacle_inflate_radius))
        self.wall_cost = float(wall_cost)
        self.penalty_radius = max(0, int(penalty_radius))
        self.penalty_cost = float(penalty_cost)
        self.carve_width = max(1, int(carve_width))

    def validate_and_fix(
        self,
        solids: np.ndarray,
        *,
        spawn_corners: dict[str, str],
        spawn_clear: int,
        meta: dict[str, Any],
    ) -> np.ndarray:
        if solids.shape != (self.size_z, self.size_y, self.size_x):
            raise ValueError(f"solids has shape {solids.shape}, expected {(self.size_z, self.size_y, self.size_x)}")

        fixups: list[str] = meta.setdefault("fixups", [])
        stats: dict[str, Any] = meta.setdefault("stats", {})
        paths_stats: dict[str, Any] = stats.setdefault("paths", {})
        meta.setdefault("validator", {})
        meta["validator"].update(
            {
                "type": "connectivity_dig_astar_v1",
                "clearance_z": int(self.clearance_z),
                "obstacle_inflate_radius": int(self.obstacle_inflate_radius),
                "wall_cost": float(self.wall_cost),
                "penalty_radius": int(self.penalty_radius),
                "penalty_cost": float(self.penalty_cost),
                "carve_width": int(self.carve_width),
            }
        )

        blue_corner = spawn_corners.get("blue", "BL")
        red_corner = spawn_corners.get("red", "TR")
        blue = _corner_anchor(blue_corner, spawn_clear, self.size_y, self.size_x)
        red = _corner_anchor(red_corner, spawn_clear, self.size_y, self.size_x)
        objective = capture_zone_anchor(meta, size_x=self.size_x, size_y=self.size_y)

        solids = self._ensure_two_paths(solids, blue, objective, label="blue_to_objective", fixups=fixups, paths_stats=paths_stats)
        solids = self._ensure_two_paths(solids, red, objective, label="red_to_objective", fixups=fixups, paths_stats=paths_stats)
        return solids

    def _build_nav_grid(self, solids: np.ndarray) -> np.ndarray:
        footprint = np.any(solids[: self.clearance_z, :, :], axis=0)
        return _inflate_obstacles_8(footprint, self.obstacle_inflate_radius)

    def _ensure_one_path(
        self,
        solids: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        *,
        label: str,
        fixups: list[str],
        paths_stats: dict[str, Any],
    ) -> np.ndarray:
        nav = self._build_nav_grid(solids)
        res = self._astar_dig(nav, start, goal, penalty=None)
        paths_stats[label] = {
            "found": bool(res.found),
            "len": int(len(res.path)) if res.found else 0,
            "digs": int(res.digs_needed),
        }
        if res.found and res.digs_needed > 0:
            solids = self._apply_carve(solids, res.path)
            fixups.append(f"{label}:dig={res.digs_needed}")
        return solids

    def _ensure_two_paths(
        self,
        solids: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        *,
        label: str,
        fixups: list[str],
        paths_stats: dict[str, Any],
    ) -> np.ndarray:
        nav = self._build_nav_grid(solids)

        res1 = self._astar_dig(nav, start, goal, penalty=None)
        if not res1.found:
            fixups.append(f"{label}:path1:failed")
            paths_stats[label] = {"found": False}
            return solids
        if res1.digs_needed > 0:
            solids = self._apply_carve(solids, res1.path)
            fixups.append(f"{label}:path1:dig={res1.digs_needed}")
            nav = self._build_nav_grid(solids)

        penalty = self._penalty_grid(res1.path)
        res2 = self._astar_dig(nav, start, goal, penalty=penalty)
        if not res2.found:
            fixups.append(f"{label}:path2:failed")
            paths_stats[label] = {
                "found": True,
                "len_a": int(len(res1.path)),
                "digs_a": int(res1.digs_needed),
                "found_b": False,
            }
            return solids
        if res2.digs_needed > 0:
            solids = self._apply_carve(solids, res2.path)
            fixups.append(f"{label}:path2:dig={res2.digs_needed}")
            overlap = self._path_overlap_ratio(res1.path, res2.path)
            paths_stats[label] = {
                "found": True,
                "len_a": int(len(res1.path)),
                "len_b": int(len(res2.path)),
                "overlap": float(overlap),
                "digs_a": int(res1.digs_needed),
                "digs_b": int(res2.digs_needed),
            }
            return solids

        overlap = self._path_overlap_ratio(res1.path, res2.path)
        paths_stats[label] = {
            "found": True,
            "len_a": int(len(res1.path)),
            "len_b": int(len(res2.path)),
            "overlap": float(overlap),
            "digs_a": int(res1.digs_needed),
            "digs_b": int(res2.digs_needed),
        }
        return solids

    def _penalty_grid(self, path: list[tuple[int, int]]) -> np.ndarray:
        if self.penalty_radius <= 0 or self.penalty_cost <= 0.0 or not path:
            return np.zeros((self.size_y, self.size_x), dtype=np.float32)
        pen = np.zeros((self.size_y, self.size_x), dtype=np.float32)
        r = self.penalty_radius
        for y, x in path:
            y0 = max(0, y - r)
            y1 = min(self.size_y, y + r + 1)
            x0 = max(0, x - r)
            x1 = min(self.size_x, x + r + 1)
            pen[y0:y1, x0:x1] += self.penalty_cost
        return pen

    def _path_overlap_ratio(self, a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
        if not a or not b:
            return 0.0
        sa = set(a)
        shared = sum((p in sa) for p in b)
        return float(shared / max(1, len(b)))

    def _astar_dig(
        self,
        blocked: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        *,
        penalty: np.ndarray | None,
    ) -> PathResult:
        sy, sx = start
        gy, gx = goal
        if not (0 <= sy < self.size_y and 0 <= sx < self.size_x):
            return PathResult([], float("inf"), 0, False)
        if not (0 <= gy < self.size_y and 0 <= gx < self.size_x):
            return PathResult([], float("inf"), 0, False)

        def h(y: int, x: int) -> float:
            return float(abs(y - gy) + abs(x - gx))

        frontier: list[tuple[float, int, int]] = []
        heapq.heappush(frontier, (0.0, sy, sx))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {(sy, sx): None}
        cost_so_far: dict[tuple[int, int], float] = {(sy, sx): 0.0}

        found = False
        while frontier:
            _, cy, cx = heapq.heappop(frontier)
            if (cy, cx) == (gy, gx):
                found = True
                break

            base_cost = cost_so_far[(cy, cx)]
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if not (0 <= ny < self.size_y and 0 <= nx < self.size_x):
                    continue
                step = 1.0 + (self.wall_cost if blocked[ny, nx] else 0.0)
                if penalty is not None:
                    step += float(penalty[ny, nx])
                new_cost = base_cost + step
                prev = cost_so_far.get((ny, nx))
                if prev is None or new_cost < prev:
                    cost_so_far[(ny, nx)] = new_cost
                    priority = new_cost + h(ny, nx)
                    heapq.heappush(frontier, (priority, ny, nx))
                    came_from[(ny, nx)] = (cy, cx)

        if not found:
            return PathResult([], float("inf"), 0, False)

        path: list[tuple[int, int]] = []
        digs = 0
        cur: tuple[int, int] | None = (gy, gx)
        while cur is not None:
            path.append(cur)
            if blocked[cur[0], cur[1]]:
                digs += 1
            cur = came_from[cur]
        path.reverse()
        return PathResult(path, float(cost_so_far[(gy, gx)]), digs, True)

    def _apply_carve(self, solids: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
        if not path:
            return solids
        radius = max(1, int(self.carve_width // 2))
        for y, x in path:
            y0 = max(0, y - radius)
            y1 = min(self.size_y, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(self.size_x, x + radius + 1)
            solids[:, y0:y1, x0:x1] = False
        return solids
