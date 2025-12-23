from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import NavGraph, NodeID


@dataclass
class PathStats:
    found: bool
    length: int
    cost: float
    visited_count: int


class Planner:
    """
    Pathfinding utility over the NavGraph.
    """

    def __init__(self, graph: NavGraph):
        self.graph = graph

    def find_path(
        self, start_id: NodeID, goal_id: NodeID, max_visited: int = 5000
    ) -> tuple[list[NodeID], PathStats]:
        """
        Standard A* search on the nav graph.
        """
        if start_id not in self.graph.nodes:
            return [], PathStats(False, 0, 0.0, 0)
        if goal_id not in self.graph.nodes:
            # Maybe find nearest valid neighbor to goal?
            # For now strict.
            return [], PathStats(False, 0, 0.0, 0)

        self.graph.nodes[start_id]
        goal_node = self.graph.nodes[goal_id]

        # Heuristic: Euclidean distance
        def heuristic(nid: NodeID) -> float:
            n = self.graph.nodes[nid]
            g = goal_node
            dx = n.pos[0] - g.pos[0]
            dy = n.pos[1] - g.pos[1]
            dz = n.pos[2] - g.pos[2]
            return float((dx * dx + dy * dy + dz * dz) ** 0.5)

        frontier: list[tuple[float, NodeID]] = []
        heapq.heappush(frontier, (0.0, start_id))

        came_from: dict[NodeID, NodeID | None] = {start_id: None}
        cost_so_far: dict[NodeID, float] = {start_id: 0.0}

        visited = 0
        found = False

        while frontier:
            if visited >= max_visited:
                break

            _, current_id = heapq.heappop(frontier)
            visited += 1

            if current_id == goal_id:
                found = True
                break

            current_node = self.graph.nodes[current_id]
            for next_id, edge_cost in current_node.edges:
                new_cost = cost_so_far[current_id] + edge_cost
                if next_id not in cost_so_far or new_cost < cost_so_far[next_id]:
                    cost_so_far[next_id] = new_cost
                    priority = new_cost + heuristic(next_id)
                    heapq.heappush(frontier, (priority, next_id))
                    came_from[next_id] = current_id

        if not found:
            return [], PathStats(False, 0, 0.0, visited)

        # Reconstruct
        path = []
        cur: NodeID | None = goal_id
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()

        return path, PathStats(True, len(path), cost_so_far[goal_id], visited)
