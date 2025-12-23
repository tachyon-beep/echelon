from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..nav.graph import NavGraph
from ..nav.planner import Planner
from ..sim.world import VoxelWorld
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
        self.size_z, self.size_y, self.size_x = (
            int(world_shape_zyx[0]),
            int(world_shape_zyx[1]),
            int(world_shape_zyx[2]),
        )
        self.clearance_z = max(1, int(min(clearance_z, self.size_z)))
        self.obstacle_inflate_radius = max(0, int(obstacle_inflate_radius))
        self.wall_cost = float(wall_cost)
        self.penalty_radius = max(0, int(penalty_radius))
        self.penalty_cost = float(penalty_cost)
        self.carve_width = max(1, int(carve_width))

    def validate_and_fix(
        self,
        voxels: np.ndarray,
        *,
        spawn_corners: dict[str, str],
        spawn_clear: int,
        meta: dict[str, Any],
    ) -> np.ndarray:
        if voxels.shape != (self.size_z, self.size_y, self.size_x):
            raise ValueError(
                f"voxels has shape {voxels.shape}, expected {(self.size_z, self.size_y, self.size_x)}"
            )

        fixups: list[str] = meta.setdefault("fixups", [])
        stats: dict[str, Any] = meta.setdefault("stats", {})
        paths_stats: dict[str, Any] = stats.setdefault("paths", {})
        meta.setdefault("validator", {})
        meta["validator"].update(
            {
                "type": "connectivity_nav_graph_v1",
                "clearance_z": int(self.clearance_z),
                "carve_width": int(self.carve_width),
            }
        )

        blue_corner = spawn_corners.get("blue", "BL")
        red_corner = spawn_corners.get("red", "TR")
        blue = _corner_anchor(blue_corner, spawn_clear, self.size_y, self.size_x)
        red = _corner_anchor(red_corner, spawn_clear, self.size_y, self.size_x)
        objective = capture_zone_anchor(meta, size_x=self.size_x, size_y=self.size_y)

        # Convert anchors to world pos for NavGraph
        # Note: NavGraph assumes voxel_size=5.0 by default or reads from world.
        # We construct a temp world to build the graph.

        # We need to loop: Check Nav -> If bad, Fix 2D -> Recheck.

        for _team, start, label in [
            ("blue", blue, "blue_to_objective"),
            ("red", red, "red_to_objective"),
        ]:
            voxels = self._ensure_path_nav(
                voxels, start, objective, label=label, fixups=fixups, paths_stats=paths_stats
            )

        return voxels

    def _ensure_path_nav(
        self,
        voxels: np.ndarray,
        start_yx: tuple[int, int],
        goal_yx: tuple[int, int],
        *,
        label: str,
        fixups: list[str],
        paths_stats: dict[str, Any],
    ) -> np.ndarray:
        # Retry loop
        for attempt in range(3):
            # 1. Build Nav Graph
            tmp_world = VoxelWorld(voxels=voxels, voxel_size_m=5.0)
            # Ensure ground layer so z=-1 nodes exist?
            # VoxelWorld.generate calls ensure_ground_layer.
            # But here we just wrap voxels.
            # NavGraph respects DIRT logic (z=0 is non-colliding).
            # If voxels has DIRT at 0, NavGraph works.
            # If voxels has AIR at 0, NavGraph works (floor is -1).

            graph = NavGraph.build(
                tmp_world, clearance_z=self.clearance_z, mech_radius=self.obstacle_inflate_radius
            )
            planner = Planner(graph)

            # 2. Find Nodes
            # We assume flat ground at start/goal?
            # We search for nearest node to (x, y, 0).
            start_pos = (float(start_yx[1] + 0.5) * 5.0, float(start_yx[0] + 0.5) * 5.0, 0.0)
            goal_pos = (float(goal_yx[1] + 0.5) * 5.0, float(goal_yx[0] + 0.5) * 5.0, 0.0)

            start_node = graph.get_nearest_node(start_pos)
            goal_node = graph.get_nearest_node(goal_pos)

            found = False
            if start_node and goal_node:
                _path, pstats = planner.find_path(start_node, goal_node)
                found = pstats.found

            paths_stats.setdefault(label, {})["attempt_" + str(attempt)] = found

            if found:
                return voxels

            # 3. If fail, use 2D digger
            # We find a path on the 2D projected grid (where obstacles are high cost)
            nav_2d = self._build_nav_grid(voxels)
            forbidden = self._forbidden_footprint()
            res = self._astar_dig(nav_2d, start_yx, goal_yx, penalty=None, forbidden=forbidden)

            if res.found:
                # Use Staircase Digger: Create explicit ramps along the 2D path
                voxels = self._apply_staircase_carve(voxels, res.path)
                fixups.append(f"{label}:attempt={attempt}:staircase_dig={res.digs_needed}")
            else:
                fixups.append(f"{label}:attempt={attempt}:2d_failed")
                break

        raise RuntimeError(
            f"ConnectivityValidator failed to find path for {label} after {attempt + 1} attempts."
        )

    def _apply_staircase_carve(self, voxels: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
        """
        Creates a 3D traversable ramp along a 2D path.
        Forces floor and headroom at each step.
        """
        if not path:
            return voxels

        radius = max(1, int(self.carve_width // 2))
        sz, sy, sx = voxels.shape

        # We start at bedrock (z=-1) and maintain current_z
        cur_z = -1

        for i in range(len(path)):
            y, x = path[i]
            y0, y1 = max(0, y - radius), min(sy, y + radius + 1)
            x0, x1 = max(0, x - radius), min(sx, x + radius + 1)

            # 1. Place Floor (SOLID) if above bedrock
            if cur_z >= 0:
                voxels[cur_z, y0:y1, x0:x1] = 1  # SOLID

            # 2. Clear Headroom (AIR)
            # Stand at cur_z, so clear from cur_z + 1
            z_start = max(0, cur_z + 1)
            z_top = min(sz, z_start + self.clearance_z)
            voxels[z_start:z_top, y0:y1, x0:x1] = 0  # AIR

        return voxels

    def _build_nav_grid(self, voxels: np.ndarray) -> np.ndarray:
        # Treat SOLID (1) as blocked. Passable hazards (Lava/Water) are NOT blocked for nav here.
        # This can be refined later if we want mechs to avoid Lava.
        footprint = np.any(voxels[: self.clearance_z, :, :] == 1, axis=0)
        return _inflate_obstacles_8(footprint, self.obstacle_inflate_radius)

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

    def _forbidden_footprint(self) -> np.ndarray:
        r = max(0, int(self.obstacle_inflate_radius))
        forbidden = np.zeros((self.size_y, self.size_x), dtype=bool)
        if r <= 0:
            return forbidden
        forbidden[:r, :] = True
        forbidden[-r:, :] = True
        forbidden[:, :r] = True
        forbidden[:, -r:] = True
        return forbidden

    def _astar_dig(
        self,
        blocked: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        *,
        penalty: np.ndarray | None,
        forbidden: np.ndarray | None,
    ) -> PathResult:
        sy, sx = start
        gy, gx = goal
        if not (0 <= sy < self.size_y and 0 <= sx < self.size_x):
            return PathResult([], float("inf"), 0, False)
        if not (0 <= gy < self.size_y and 0 <= gx < self.size_x):
            return PathResult([], float("inf"), 0, False)
        if forbidden is not None:
            if forbidden.shape != (self.size_y, self.size_x):
                raise ValueError(
                    f"forbidden has shape {forbidden.shape}, expected {(self.size_y, self.size_x)}"
                )
            if forbidden[sy, sx] or forbidden[gy, gx]:
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
                if forbidden is not None and forbidden[ny, nx]:
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

    def _apply_carve(self, voxels: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
        if not path:
            return voxels
        radius = max(1, int(self.carve_width // 2))
        for y, x in path:
            y0 = max(0, y - radius)
            y1 = min(self.size_y, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(self.size_x, x + radius + 1)
            # Clear 0..clearance_z + 1 to ensure standing room
            voxels[0 : self.clearance_z + 1, y0:y1, x0:x1] = 0  # AIR
        return voxels
