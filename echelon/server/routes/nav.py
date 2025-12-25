# echelon/server/routes/nav.py
"""Navigation graph and pathfinding endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from echelon.nav.planner import Planner

from ..models import PathRequest
from ..nav_cache import nav_cache
from ..world_cache import world_cache

logger = logging.getLogger("echelon.server")

router = APIRouter(prefix="/nav")


@router.post("/graph")
async def get_nav_graph(request: dict):
    """Get or build NavGraph for a world (cached by hash)."""
    world_hash = request.get("world_hash")
    if not world_hash:
        raise HTTPException(status_code=400, detail="world_hash required")

    world_data = world_cache.get(world_hash)
    if world_data is None:
        raise HTTPException(status_code=404, detail=f"World {world_hash} not in cache")

    try:
        graph = await nav_cache.get_or_build(world_hash, world_data)
        return graph.to_dict()
    except Exception as e:
        logger.error(f"Failed to build NavGraph: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/path")
async def plan_path(req: PathRequest):
    """Compute A* path using cached NavGraph."""
    world_data = world_cache.get(req.world_hash)
    if world_data is None:
        raise HTTPException(status_code=404, detail=f"World {req.world_hash} not in cache")

    try:
        graph = await nav_cache.get_or_build(req.world_hash, world_data)
        planner = Planner(graph)

        start_node = graph.get_nearest_node((req.start_pos[0], req.start_pos[1], req.start_pos[2]))
        goal_node = graph.get_nearest_node((req.goal_pos[0], req.goal_pos[1], req.goal_pos[2]))

        if not start_node or not goal_node:
            return {"found": False, "error": "Start or goal not on nav graph"}

        path_ids, stats = planner.find_path(start_node, goal_node)
        path_pos = [list(graph.nodes[nid].pos) for nid in path_ids]

        return {
            "found": stats.found,
            "path": path_pos,
            "node_ids": [list(nid) for nid in path_ids],
            "stats": {
                "length": stats.length,
                "cost": stats.cost,
                "visited": stats.visited_count,
            },
        }
    except Exception as e:
        logger.error(f"Pathfinding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
