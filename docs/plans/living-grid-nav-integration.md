# Nav Graph Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate the existing NavGraph/Planner with the simulation so heuristic agents navigate intelligently and mechs don't walk into walls.

**Architecture:** Cache the nav graph at env.reset(), wire it into HeuristicPolicy for waypoint-following, add optional nav-assist post-processing for movement actions to reduce "stuck" rates.

**Tech Stack:** NumPy, existing `echelon/nav/` modules

---

## Task 1: Add NavGraph Caching to EchelonEnv

**Files:**
- Modify: `echelon/env/env.py`

**Step 1: Add nav graph import and instance variable**

In `echelon/env/env.py`, add import at top:

```python
from echelon.nav.graph import NavGraph
from echelon.nav.planner import Planner
```

Add to `EchelonEnv.__init__()` after `self.world`:

```python
self.nav_graph: NavGraph | None = None
self.nav_planner: Planner | None = None
```

**Step 2: Build nav graph in reset()**

In `EchelonEnv.reset()`, after world generation and spawn clearing, add:

```python
# Build navigation graph
self.nav_graph = NavGraph.build(
    self.world,
    clearance_z=4,  # Mech height ~20m = 4 voxels
    mech_radius=1,  # Heavy mechs ~10m = 1 voxel radius
)
self.nav_planner = Planner(self.nav_graph)
```

**Step 3: Run existing tests**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_env*.py -v`
Expected: All existing tests PASS (nav graph is additive)

**Step 4: Commit**

```bash
git add echelon/env/env.py
git commit -m "feat(nav): cache NavGraph in EchelonEnv.reset()"
```

---

## Task 2: Add Nav Graph Stats to Replay Metadata

**Files:**
- Modify: `echelon/env/env.py`

**Step 1: Add nav stats to world.meta**

In `EchelonEnv.reset()`, after building nav graph, add:

```python
if self.world.meta is None:
    self.world.meta = {}
self.world.meta["nav"] = {
    "node_count": len(self.nav_graph.nodes),
    "edge_count": sum(len(n.edges) for n in self.nav_graph.nodes.values()),
    "clearance_z": self.nav_graph.clearance_z,
}
```

**Step 2: Verify replay includes nav stats**

Run: `PYTHONPATH=. uv run python scripts/smoke.py --episodes 1 --packs-per-team 1 --size 40 --mode full`
Check output for nav stats in world metadata.

**Step 3: Commit**

```bash
git add echelon/env/env.py
git commit -m "feat(nav): add nav graph stats to replay metadata"
```

---

## Task 3: Create NavController for Waypoint Following

**Files:**
- Create: `echelon/nav/controller.py`
- Create: `tests/unit/test_nav_controller.py`

**Step 1: Write failing test**

Create `tests/unit/test_nav_controller.py`:

```python
"""Tests for NavController waypoint following."""
from __future__ import annotations

import numpy as np
import pytest

from echelon.nav.controller import NavController, WaypointResult


class TestNavController:
    """Tests for NavController."""

    def test_compute_steering_toward_waypoint(self):
        """Controller should steer toward next waypoint."""
        controller = NavController()

        # Mech at origin, facing +X, waypoint at (10, 0)
        mech_pos = np.array([0.0, 0.0, 0.0])
        mech_yaw = 0.0  # Facing +X
        waypoint = np.array([10.0, 0.0, 0.0])

        result = controller.compute_steering(mech_pos, mech_yaw, waypoint)

        assert result.forward > 0.5  # Should move forward
        assert abs(result.yaw_rate) < 0.1  # Already facing target

    def test_compute_steering_turn_required(self):
        """Controller should turn when waypoint is to the side."""
        controller = NavController()

        # Mech at origin, facing +X, waypoint at (0, 10) (to the left)
        mech_pos = np.array([0.0, 0.0, 0.0])
        mech_yaw = 0.0  # Facing +X
        waypoint = np.array([0.0, 10.0, 0.0])

        result = controller.compute_steering(mech_pos, mech_yaw, waypoint)

        assert result.yaw_rate > 0.3  # Should turn left (positive yaw)

    def test_waypoint_reached_when_close(self):
        """Controller should report waypoint reached when close."""
        controller = NavController(arrival_threshold=2.0)

        mech_pos = np.array([9.5, 0.0, 0.0])
        waypoint = np.array([10.0, 0.0, 0.0])

        result = controller.compute_steering(mech_pos, 0.0, waypoint)

        assert result.reached is True
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_nav_controller.py -v`
Expected: FAIL with ImportError

**Step 3: Implement NavController**

Create `echelon/nav/controller.py`:

```python
"""Waypoint-following controller for mech navigation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class WaypointResult:
    """Result of waypoint steering computation."""
    forward: float  # [-1, 1] forward/backward
    strafe: float   # [-1, 1] left/right
    yaw_rate: float # [-1, 1] turn rate
    reached: bool   # True if waypoint reached


class NavController:
    """
    Simple waypoint-following controller.

    Converts a target waypoint into movement commands.
    """

    def __init__(
        self,
        arrival_threshold: float = 3.0,
        turn_gain: float = 2.0,
        forward_gain: float = 1.0,
    ):
        self.arrival_threshold = arrival_threshold
        self.turn_gain = turn_gain
        self.forward_gain = forward_gain

    def compute_steering(
        self,
        mech_pos: np.ndarray,
        mech_yaw: float,
        waypoint: np.ndarray,
    ) -> WaypointResult:
        """
        Compute steering commands to reach waypoint.

        Args:
            mech_pos: Current mech position [x, y, z]
            mech_yaw: Current mech heading in radians
            waypoint: Target waypoint [x, y, z]

        Returns:
            WaypointResult with movement commands
        """
        # Vector to waypoint (2D, ignore Z)
        dx = waypoint[0] - mech_pos[0]
        dy = waypoint[1] - mech_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # Check arrival
        if dist < self.arrival_threshold:
            return WaypointResult(forward=0.0, strafe=0.0, yaw_rate=0.0, reached=True)

        # Angle to waypoint
        target_yaw = math.atan2(dy, dx)

        # Angular error (wrapped to [-pi, pi])
        yaw_error = target_yaw - mech_yaw
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        # Steering commands
        yaw_rate = np.clip(yaw_error * self.turn_gain, -1.0, 1.0)

        # Forward speed based on alignment (slow down when turning)
        alignment = math.cos(yaw_error)
        forward = np.clip(alignment * self.forward_gain, 0.0, 1.0)

        # Strafe slightly to help with cornering
        strafe = np.clip(math.sin(yaw_error) * 0.3, -1.0, 1.0)

        return WaypointResult(
            forward=float(forward),
            strafe=float(strafe),
            yaw_rate=float(yaw_rate),
            reached=False,
        )
```

**Step 4: Run tests**

Run: `PYTHONPATH=. uv run pytest tests/unit/test_nav_controller.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add echelon/nav/controller.py tests/unit/test_nav_controller.py
git commit -m "feat(nav): add NavController for waypoint following"
```

---

## Task 4: Integrate NavController with HeuristicPolicy

**Files:**
- Modify: `echelon/agents/heuristic.py`

**Step 1: Add nav imports and path caching**

At top of `echelon/agents/heuristic.py`, add:

```python
from echelon.nav.graph import NavGraph
from echelon.nav.planner import Planner
from echelon.nav.controller import NavController, WaypointResult
```

Add to `HeuristicPolicy.__init__()`:

```python
self.nav_controller = NavController(arrival_threshold=5.0)
self._cached_paths: dict[str, list[tuple[int, int, int]]] = {}
self._path_update_interval = 30  # Recompute every 30 decision steps
self._step_counters: dict[str, int] = {}
```

**Step 2: Add path computation method**

Add to `HeuristicPolicy`:

```python
def _get_nav_waypoint(
    self,
    mech: MechState,
    goal_pos: np.ndarray,
    nav_graph: NavGraph | None,
    nav_planner: Planner | None,
) -> np.ndarray | None:
    """Get next waypoint from nav path, recomputing if needed."""
    if nav_graph is None or nav_planner is None:
        return None

    mech_id = mech.id

    # Update step counter
    self._step_counters[mech_id] = self._step_counters.get(mech_id, 0) + 1

    # Recompute path periodically or if no cached path
    if (
        mech_id not in self._cached_paths
        or self._step_counters[mech_id] >= self._path_update_interval
    ):
        self._step_counters[mech_id] = 0

        start_node = nav_graph.get_nearest_node((mech.pos[0], mech.pos[1], mech.pos[2]))
        goal_node = nav_graph.get_nearest_node((goal_pos[0], goal_pos[1], goal_pos[2]))

        if start_node is None or goal_node is None:
            self._cached_paths[mech_id] = []
            return None

        path, stats = nav_planner.find_path(start_node, goal_node)
        self._cached_paths[mech_id] = path

    path = self._cached_paths.get(mech_id, [])
    if not path:
        return None

    # Find next waypoint (skip nodes we've passed)
    mech_pos_2d = np.array([mech.pos[0], mech.pos[1]])
    for i, node_id in enumerate(path):
        node = nav_graph.nodes[node_id]
        node_pos_2d = np.array([node.pos[0], node.pos[1]])
        dist = np.linalg.norm(node_pos_2d - mech_pos_2d)
        if dist > 5.0:  # Found a waypoint ahead of us
            return np.array([node.pos[0], node.pos[1], node.pos[2]])

    # All waypoints passed, return goal
    return goal_pos
```

**Step 3: Use nav waypoint in movement decision**

In `HeuristicPolicy._decide_movement()` or equivalent, replace direct goal targeting with:

```python
# Get nav waypoint if available
nav_waypoint = self._get_nav_waypoint(mech, goal_pos, nav_graph, nav_planner)
if nav_waypoint is not None:
    steering = self.nav_controller.compute_steering(mech.pos, mech.yaw, nav_waypoint)
    action[0] = steering.forward
    action[1] = steering.strafe
    action[3] = steering.yaw_rate
else:
    # Fallback to direct targeting (existing logic)
    ...
```

**Step 4: Update HeuristicPolicy.act() signature**

Modify `HeuristicPolicy.act()` to accept optional nav parameters:

```python
def act(
    self,
    obs: dict[str, np.ndarray],
    sim: Sim,
    world: VoxelWorld,
    nav_graph: NavGraph | None = None,
    nav_planner: Planner | None = None,
) -> dict[str, np.ndarray]:
```

**Step 5: Run tests**

Run: `PYTHONPATH=. uv run pytest tests/ -v -x`
Expected: Tests pass (may need to update test fixtures)

**Step 6: Commit**

```bash
git add echelon/agents/heuristic.py
git commit -m "feat(nav): integrate NavController with HeuristicPolicy"
```

---

## Task 5: Wire Nav Graph to Heuristic in EchelonEnv

**Files:**
- Modify: `echelon/env/env.py`

**Step 1: Pass nav graph to heuristic policy**

In `EchelonEnv.step()`, where `HeuristicPolicy.act()` is called for opponent agents, update:

```python
red_actions = self.red_policy.act(
    red_obs,
    self.sim,
    self.world,
    nav_graph=self.nav_graph,
    nav_planner=self.nav_planner,
)
```

**Step 2: Run smoke test**

Run: `PYTHONPATH=. uv run python scripts/smoke.py --episodes 1 --packs-per-team 1 --size 40 --mode full`
Expected: Runs without errors, heuristic agents navigate

**Step 3: Commit**

```bash
git add echelon/env/env.py
git commit -m "feat(nav): wire NavGraph to HeuristicPolicy in env.step()"
```

---

## Task 6: Add Nav Integration Tests

**Files:**
- Create: `tests/integration/test_nav_integration.py`

**Step 1: Write integration test**

Create `tests/integration/test_nav_integration.py`:

```python
"""Integration tests for nav graph with environment."""
from __future__ import annotations

import numpy as np
import pytest

from echelon.env.env import EchelonEnv
from echelon.config import EnvConfig


class TestNavIntegration:
    """Test nav graph integration with EchelonEnv."""

    @pytest.fixture
    def env(self) -> EchelonEnv:
        cfg = EnvConfig(
            packs_per_team=1,
            size_x=50,
            size_y=50,
            size_z=10,
            seed=42,
        )
        return EchelonEnv(cfg)

    def test_nav_graph_built_on_reset(self, env: EchelonEnv):
        """Nav graph should be built during reset."""
        env.reset(seed=42)

        assert env.nav_graph is not None
        assert env.nav_planner is not None
        assert len(env.nav_graph.nodes) > 0

    def test_nav_stats_in_metadata(self, env: EchelonEnv):
        """Nav stats should appear in world metadata."""
        env.reset(seed=42)

        assert env.world is not None
        assert env.world.meta is not None
        assert "nav" in env.world.meta
        assert env.world.meta["nav"]["node_count"] > 0

    def test_spawns_reachable_via_nav(self, env: EchelonEnv):
        """Both spawns should be able to reach center via nav graph."""
        env.reset(seed=42)

        assert env.nav_graph is not None
        assert env.nav_planner is not None
        assert env.sim is not None

        # Get spawn positions
        blue_mechs = [m for m in env.sim.mechs.values() if m.team == "blue"]
        red_mechs = [m for m in env.sim.mechs.values() if m.team == "red"]

        assert len(blue_mechs) > 0
        assert len(red_mechs) > 0

        # Center of map
        center = np.array([
            env.world.size_x * env.world.voxel_size_m / 2,
            env.world.size_y * env.world.voxel_size_m / 2,
            0.0,
        ])

        # Check blue can reach center
        blue_pos = blue_mechs[0].pos
        blue_node = env.nav_graph.get_nearest_node(tuple(blue_pos))
        center_node = env.nav_graph.get_nearest_node(tuple(center))

        if blue_node and center_node:
            path, stats = env.nav_planner.find_path(blue_node, center_node)
            assert stats.found, "Blue spawn should reach center"

        # Check red can reach center
        red_pos = red_mechs[0].pos
        red_node = env.nav_graph.get_nearest_node(tuple(red_pos))

        if red_node and center_node:
            path, stats = env.nav_planner.find_path(red_node, center_node)
            assert stats.found, "Red spawn should reach center"

    def test_heuristic_uses_nav_graph(self, env: EchelonEnv):
        """Heuristic policy should use nav graph when available."""
        obs, _ = env.reset(seed=42)

        # Run a few steps and check heuristic doesn't get stuck
        stuck_count = 0
        prev_positions: dict[str, np.ndarray] = {}

        for _ in range(100):
            # Random actions for blue team
            actions = {}
            for aid in env.blue_ids:
                actions[aid] = np.zeros(env.ACTION_DIM, dtype=np.float32)
                actions[aid][0] = 0.5  # Move forward

            obs, rewards, terms, truncs, infos = env.step(actions)

            # Check red mechs are moving (using heuristic with nav)
            if env.sim is not None:
                for m in env.sim.mechs.values():
                    if m.team == "red" and m.hp > 0:
                        pos = tuple(m.pos)
                        if m.id in prev_positions:
                            moved = np.linalg.norm(m.pos - prev_positions[m.id])
                            if moved < 0.1:
                                stuck_count += 1
                        prev_positions[m.id] = m.pos.copy()

        # Allow some stuck frames but not too many
        assert stuck_count < 50, f"Too many stuck frames: {stuck_count}"
```

**Step 2: Run integration tests**

Run: `PYTHONPATH=. uv run pytest tests/integration/test_nav_integration.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/integration/test_nav_integration.py
git commit -m "test(nav): add nav graph integration tests"
```

---

## Summary

| Task | Description | Estimated Complexity |
|------|-------------|---------------------|
| 1 | Add NavGraph caching to EchelonEnv | Small |
| 2 | Add nav stats to replay metadata | Trivial |
| 3 | Create NavController for waypoint following | Medium |
| 4 | Integrate NavController with HeuristicPolicy | Medium |
| 5 | Wire nav graph to heuristic in env.step() | Small |
| 6 | Add nav integration tests | Small |

**Total: 6 tasks**
