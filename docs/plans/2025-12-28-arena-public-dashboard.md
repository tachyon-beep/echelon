# Arena Public Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add public-facing API, stats collection, and dashboard to make the arena system engaging for non-technical viewers.

**Architecture:** Extend existing FastAPI server with league/match endpoints. Capture match stats during training via info dict events. Store matches as JSON files. Build lightweight dashboard HTML that consumes the API.

**Tech Stack:** FastAPI (existing), vanilla JS + HTML (matching viewer.html pattern), JSON file storage

---

## Phase 1: Data Model

### Task 1: Define MatchRecord and TeamStats dataclasses

**Files:**
- Create: `echelon/arena/stats.py`
- Test: `tests/unit/arena/test_stats.py`

**Step 1: Write the failing test**

```python
# tests/unit/arena/test_stats.py
"""Tests for match statistics data structures."""

import pytest
from echelon.arena.stats import TeamStats, MatchRecord


def test_team_stats_defaults():
    """TeamStats initializes with zero values."""
    stats = TeamStats()
    assert stats.kills == 0
    assert stats.damage_dealt == 0.0
    assert stats.zone_ticks == 0
    assert stats.orders_issued == {}


def test_team_stats_weapon_counts():
    """TeamStats tracks weapon usage counts."""
    stats = TeamStats(primary_uses=10, secondary_uses=5, tertiary_uses=3)
    assert stats.primary_uses == 10
    assert stats.secondary_uses == 5
    assert stats.tertiary_uses == 3


def test_match_record_creation():
    """MatchRecord captures full match data."""
    blue = TeamStats(kills=3, deaths=2)
    red = TeamStats(kills=2, deaths=3)

    record = MatchRecord(
        match_id="test-123",
        timestamp=1703750400.0,
        blue_entry_id="contender",
        red_entry_id="heuristic",
        winner="blue",
        duration_steps=500,
        termination="zone",
        blue_stats=blue,
        red_stats=red,
    )

    assert record.winner == "blue"
    assert record.blue_stats.kills == 3
    assert record.termination == "zone"


def test_match_record_to_dict():
    """MatchRecord serializes to dict for JSON storage."""
    blue = TeamStats(kills=1)
    red = TeamStats(kills=2)

    record = MatchRecord(
        match_id="test-456",
        timestamp=1703750400.0,
        blue_entry_id="a",
        red_entry_id="b",
        winner="red",
        duration_steps=100,
        termination="elimination",
        blue_stats=blue,
        red_stats=red,
    )

    d = record.to_dict()
    assert d["match_id"] == "test-456"
    assert d["blue_stats"]["kills"] == 1
    assert d["red_stats"]["kills"] == 2


def test_match_record_from_dict():
    """MatchRecord deserializes from dict."""
    d = {
        "match_id": "test-789",
        "timestamp": 1703750400.0,
        "blue_entry_id": "x",
        "red_entry_id": "y",
        "winner": "draw",
        "duration_steps": 200,
        "termination": "timeout",
        "blue_stats": {"kills": 0, "deaths": 0},
        "red_stats": {"kills": 0, "deaths": 0},
    }

    record = MatchRecord.from_dict(d)
    assert record.match_id == "test-789"
    assert record.winner == "draw"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/arena/test_stats.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'echelon.arena.stats'"

**Step 3: Write minimal implementation**

```python
# echelon/arena/stats.py
"""Match statistics data structures for arena tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TeamStats:
    """Aggregate statistics for one team in a match."""

    # Combat
    kills: int = 0
    deaths: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0

    # Objective
    zone_ticks: int = 0

    # Weapon usage (counts)
    primary_uses: int = 0
    secondary_uses: int = 0
    tertiary_uses: int = 0

    # Utility usage
    vents: int = 0
    smokes: int = 0
    ecm_toggles: int = 0
    paints: int = 0

    # Resource events
    overheats: int = 0
    knockdowns: int = 0

    # Pack leader command stats
    orders_issued: dict[str, int] = field(default_factory=dict)
    orders_acknowledged: int = 0
    orders_overridden: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "kills": self.kills,
            "deaths": self.deaths,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "zone_ticks": self.zone_ticks,
            "primary_uses": self.primary_uses,
            "secondary_uses": self.secondary_uses,
            "tertiary_uses": self.tertiary_uses,
            "vents": self.vents,
            "smokes": self.smokes,
            "ecm_toggles": self.ecm_toggles,
            "paints": self.paints,
            "overheats": self.overheats,
            "knockdowns": self.knockdowns,
            "orders_issued": self.orders_issued,
            "orders_acknowledged": self.orders_acknowledged,
            "orders_overridden": self.orders_overridden,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TeamStats:
        """Deserialize from dict."""
        return cls(
            kills=d.get("kills", 0),
            deaths=d.get("deaths", 0),
            damage_dealt=d.get("damage_dealt", 0.0),
            damage_taken=d.get("damage_taken", 0.0),
            zone_ticks=d.get("zone_ticks", 0),
            primary_uses=d.get("primary_uses", 0),
            secondary_uses=d.get("secondary_uses", 0),
            tertiary_uses=d.get("tertiary_uses", 0),
            vents=d.get("vents", 0),
            smokes=d.get("smokes", 0),
            ecm_toggles=d.get("ecm_toggles", 0),
            paints=d.get("paints", 0),
            overheats=d.get("overheats", 0),
            knockdowns=d.get("knockdowns", 0),
            orders_issued=d.get("orders_issued", {}),
            orders_acknowledged=d.get("orders_acknowledged", 0),
            orders_overridden=d.get("orders_overridden", 0),
        )


@dataclass
class MatchRecord:
    """Complete record of a single match."""

    match_id: str
    timestamp: float  # Unix epoch
    blue_entry_id: str  # League entry ID or "training"
    red_entry_id: str  # League entry ID (e.g., "heuristic")
    winner: str  # "blue" | "red" | "draw"
    duration_steps: int
    termination: str  # "zone" | "elimination" | "timeout"

    blue_stats: TeamStats
    red_stats: TeamStats

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "match_id": self.match_id,
            "timestamp": self.timestamp,
            "blue_entry_id": self.blue_entry_id,
            "red_entry_id": self.red_entry_id,
            "winner": self.winner,
            "duration_steps": self.duration_steps,
            "termination": self.termination,
            "blue_stats": self.blue_stats.to_dict(),
            "red_stats": self.red_stats.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MatchRecord:
        """Deserialize from dict."""
        return cls(
            match_id=d["match_id"],
            timestamp=d["timestamp"],
            blue_entry_id=d["blue_entry_id"],
            red_entry_id=d["red_entry_id"],
            winner=d["winner"],
            duration_steps=d["duration_steps"],
            termination=d["termination"],
            blue_stats=TeamStats.from_dict(d.get("blue_stats", {})),
            red_stats=TeamStats.from_dict(d.get("red_stats", {})),
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/arena/test_stats.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add echelon/arena/stats.py tests/unit/arena/test_stats.py
git commit -m "feat(arena): add MatchRecord and TeamStats data models"
```

---

### Task 2: Add rating history to LeagueEntry

**Files:**
- Modify: `echelon/arena/league.py`
- Test: `tests/unit/arena/test_league.py` (extend existing)

**Step 1: Write the failing test**

```python
# Add to tests/unit/arena/test_league.py

def test_league_entry_rating_history():
    """LeagueEntry tracks rating history over time."""
    entry = LeagueEntry(
        entry_id="test",
        ckpt_path="",
        kind="commander",
        commander_name="Test",
        rating=Glicko2Rating(),
        games=0,
    )

    # Initially empty
    assert entry.rating_history == []

    # Record rating
    entry.record_rating(1703750400.0)
    assert len(entry.rating_history) == 1
    assert entry.rating_history[0] == (1703750400.0, 1500.0)

    # Update rating and record again
    entry.rating = Glicko2Rating(rating=1550.0)
    entry.record_rating(1703750500.0)
    assert len(entry.rating_history) == 2
    assert entry.rating_history[1] == (1703750500.0, 1550.0)


def test_league_entry_serializes_rating_history():
    """Rating history survives serialization round-trip."""
    entry = LeagueEntry(
        entry_id="test",
        ckpt_path="",
        kind="commander",
        commander_name="Test",
        rating=Glicko2Rating(),
        games=0,
        rating_history=[(1703750400.0, 1500.0), (1703750500.0, 1520.0)],
    )

    d = entry.to_dict()
    restored = LeagueEntry.from_dict(d)

    assert restored.rating_history == entry.rating_history
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/arena/test_league.py::test_league_entry_rating_history -v`
Expected: FAIL with "AttributeError: 'LeagueEntry' object has no attribute 'rating_history'"

**Step 3: Modify LeagueEntry in league.py**

Add to `LeagueEntry` dataclass:
```python
rating_history: list[tuple[float, float]] = field(default_factory=list)

def record_rating(self, timestamp: float) -> None:
    """Append current rating to history."""
    self.rating_history.append((timestamp, self.rating.rating))
```

Update `to_dict()`:
```python
"rating_history": self.rating_history,
```

Update `from_dict()`:
```python
rating_history=d.get("rating_history", []),
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/arena/test_league.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add echelon/arena/league.py tests/unit/arena/test_league.py
git commit -m "feat(arena): add rating history tracking to LeagueEntry"
```

---

### Task 3: Add aggregate stats to LeagueEntry

**Files:**
- Modify: `echelon/arena/league.py`
- Modify: `echelon/arena/stats.py` (import)
- Test: `tests/unit/arena/test_league.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/arena/test_league.py

from echelon.arena.stats import TeamStats


def test_league_entry_aggregate_stats():
    """LeagueEntry accumulates stats from matches."""
    entry = LeagueEntry(
        entry_id="test",
        ckpt_path="",
        kind="commander",
        commander_name="Test",
        rating=Glicko2Rating(),
        games=0,
    )

    # Initially zero
    assert entry.aggregate_stats.kills == 0

    # Add match stats
    match_stats = TeamStats(kills=3, deaths=1, damage_dealt=150.0)
    entry.add_match_stats(match_stats)

    assert entry.aggregate_stats.kills == 3
    assert entry.aggregate_stats.deaths == 1
    assert entry.aggregate_stats.damage_dealt == 150.0

    # Add another match
    match_stats2 = TeamStats(kills=2, deaths=2, damage_dealt=100.0)
    entry.add_match_stats(match_stats2)

    assert entry.aggregate_stats.kills == 5
    assert entry.aggregate_stats.deaths == 3
    assert entry.aggregate_stats.damage_dealt == 250.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/arena/test_league.py::test_league_entry_aggregate_stats -v`
Expected: FAIL

**Step 3: Modify LeagueEntry**

Add field:
```python
aggregate_stats: TeamStats = field(default_factory=TeamStats)
```

Add method:
```python
def add_match_stats(self, stats: TeamStats) -> None:
    """Accumulate stats from a match."""
    agg = self.aggregate_stats
    agg.kills += stats.kills
    agg.deaths += stats.deaths
    agg.damage_dealt += stats.damage_dealt
    agg.damage_taken += stats.damage_taken
    agg.zone_ticks += stats.zone_ticks
    agg.primary_uses += stats.primary_uses
    agg.secondary_uses += stats.secondary_uses
    agg.tertiary_uses += stats.tertiary_uses
    agg.vents += stats.vents
    agg.smokes += stats.smokes
    agg.ecm_toggles += stats.ecm_toggles
    agg.paints += stats.paints
    agg.overheats += stats.overheats
    agg.knockdowns += stats.knockdowns
    for order_type, count in stats.orders_issued.items():
        agg.orders_issued[order_type] = agg.orders_issued.get(order_type, 0) + count
    agg.orders_acknowledged += stats.orders_acknowledged
    agg.orders_overridden += stats.orders_overridden
```

Update serialization methods to include `aggregate_stats`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/arena/test_league.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add echelon/arena/league.py echelon/arena/stats.py tests/unit/arena/test_league.py
git commit -m "feat(arena): add aggregate stats tracking to LeagueEntry"
```

---

## Phase 2: Match History Storage

### Task 4: Create MatchHistory class for file-based storage

**Files:**
- Create: `echelon/arena/history.py`
- Test: `tests/unit/arena/test_history.py`

**Step 1: Write the failing test**

```python
# tests/unit/arena/test_history.py
"""Tests for match history storage."""

import pytest
from pathlib import Path
from echelon.arena.history import MatchHistory
from echelon.arena.stats import MatchRecord, TeamStats


@pytest.fixture
def tmp_history(tmp_path: Path) -> MatchHistory:
    """Create a MatchHistory in a temp directory."""
    return MatchHistory(tmp_path / "matches")


def test_match_history_save_and_load(tmp_history: MatchHistory):
    """MatchHistory saves and loads records."""
    record = MatchRecord(
        match_id="test-001",
        timestamp=1703750400.0,
        blue_entry_id="a",
        red_entry_id="b",
        winner="blue",
        duration_steps=100,
        termination="zone",
        blue_stats=TeamStats(kills=2),
        red_stats=TeamStats(kills=1),
    )

    tmp_history.save(record)
    loaded = tmp_history.load("test-001")

    assert loaded is not None
    assert loaded.match_id == "test-001"
    assert loaded.blue_stats.kills == 2


def test_match_history_list_recent(tmp_history: MatchHistory):
    """MatchHistory lists recent matches sorted by timestamp."""
    for i in range(5):
        record = MatchRecord(
            match_id=f"test-{i:03d}",
            timestamp=1703750400.0 + i * 100,
            blue_entry_id="a",
            red_entry_id="b",
            winner="blue",
            duration_steps=100,
            termination="zone",
            blue_stats=TeamStats(),
            red_stats=TeamStats(),
        )
        tmp_history.save(record)

    recent = tmp_history.list_recent(limit=3)

    assert len(recent) == 3
    # Most recent first
    assert recent[0].match_id == "test-004"
    assert recent[1].match_id == "test-003"
    assert recent[2].match_id == "test-002"


def test_match_history_filter_by_entry(tmp_history: MatchHistory):
    """MatchHistory filters by entry_id."""
    records = [
        MatchRecord("m1", 100.0, "alice", "bob", "blue", 50, "zone", TeamStats(), TeamStats()),
        MatchRecord("m2", 200.0, "alice", "carol", "red", 50, "zone", TeamStats(), TeamStats()),
        MatchRecord("m3", 300.0, "bob", "carol", "blue", 50, "zone", TeamStats(), TeamStats()),
    ]
    for r in records:
        tmp_history.save(r)

    alice_matches = tmp_history.list_recent(entry_id="alice")
    assert len(alice_matches) == 2

    bob_matches = tmp_history.list_recent(entry_id="bob")
    assert len(bob_matches) == 2  # m1 and m3
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/arena/test_history.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# echelon/arena/history.py
"""Match history file-based storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stats import MatchRecord


class MatchHistory:
    """File-based storage for match records.

    Stores one JSON file per match in a directory.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, match_id: str) -> Path:
        return self.directory / f"{match_id}.json"

    def save(self, record: "MatchRecord") -> None:
        """Save a match record to disk."""
        path = self._path_for(record.match_id)
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

    def load(self, match_id: str) -> "MatchRecord | None":
        """Load a match record by ID."""
        from .stats import MatchRecord

        path = self._path_for(match_id)
        if not path.exists():
            return None
        with open(path) as f:
            return MatchRecord.from_dict(json.load(f))

    def list_recent(
        self,
        limit: int = 50,
        entry_id: str | None = None,
    ) -> list["MatchRecord"]:
        """List recent matches, optionally filtered by entry_id."""
        from .stats import MatchRecord

        records: list[MatchRecord] = []
        for path in self.directory.glob("*.json"):
            with open(path) as f:
                record = MatchRecord.from_dict(json.load(f))
                if entry_id is None or entry_id in (record.blue_entry_id, record.red_entry_id):
                    records.append(record)

        # Sort by timestamp descending (most recent first)
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/arena/test_history.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add echelon/arena/history.py tests/unit/arena/test_history.py
git commit -m "feat(arena): add MatchHistory file-based storage"
```

---

## Phase 3: API Endpoints

### Task 5: Add league API routes

**Files:**
- Create: `echelon/server/routes/league.py`
- Modify: `echelon/server/__init__.py` (register router)
- Test: `tests/unit/server/test_league_routes.py`

**Step 1: Write the failing test**

```python
# tests/unit/server/test_league_routes.py
"""Tests for league API endpoints."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from echelon.server import create_app
from echelon.arena.league import League


@pytest.fixture
def app_with_league(tmp_path: Path):
    """Create app with a test league."""
    league_path = tmp_path / "league.json"
    league = League.create_empty(league_path)
    league.bootstrap_heuristic()
    league.save()

    app = create_app(league_path=league_path, matches_path=tmp_path / "matches")
    return TestClient(app)


def test_get_league(app_with_league):
    """GET /api/league returns league data."""
    response = app_with_league.get("/api/league")
    assert response.status_code == 200
    data = response.json()
    assert "entries" in data
    assert "heuristic" in data["entries"]


def test_get_standings(app_with_league):
    """GET /api/league/standings returns sorted leaderboard."""
    response = app_with_league.get("/api/league/standings")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1  # At least heuristic
    assert "entry_id" in data[0]
    assert "rating" in data[0]


def test_get_commander(app_with_league):
    """GET /api/commanders/{id} returns commander profile."""
    response = app_with_league.get("/api/commanders/heuristic")
    assert response.status_code == 200
    data = response.json()
    assert data["entry_id"] == "heuristic"
    assert data["commander_name"] == "Lieutenant Heuristic"
    assert "rating" in data
    assert "rating_history" in data
    assert "aggregate_stats" in data


def test_get_commander_not_found(app_with_league):
    """GET /api/commanders/{id} returns 404 for unknown commander."""
    response = app_with_league.get("/api/commanders/nonexistent")
    assert response.status_code == 404
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/server/test_league_routes.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# echelon/server/routes/league.py
"""League API endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Any

router = APIRouter(prefix="/api", tags=["league"])

# These will be set by the app factory
_league = None
_history = None


def init_league_routes(league, history) -> None:
    """Initialize routes with league and history instances."""
    global _league, _history
    _league = league
    _history = history


@router.get("/league")
def get_league() -> dict[str, Any]:
    """Get full league state."""
    if _league is None:
        raise HTTPException(503, "League not initialized")

    return {
        "schema_version": _league.schema_version,
        "entries": {
            eid: entry.to_dict() for eid, entry in _league.entries.items()
        },
    }


@router.get("/league/standings")
def get_standings() -> list[dict[str, Any]]:
    """Get sorted leaderboard."""
    if _league is None:
        raise HTTPException(503, "League not initialized")

    entries = list(_league.entries.values())
    # Sort by rating descending
    entries.sort(key=lambda e: e.rating.rating, reverse=True)

    return [
        {
            "entry_id": e.entry_id,
            "commander_name": e.commander_name,
            "kind": e.kind,
            "rating": e.rating.rating,
            "rd": e.rating.rd,
            "games": e.games,
            "wins": getattr(e, "wins", 0),
            "aggregate_stats": e.aggregate_stats.to_dict() if hasattr(e, "aggregate_stats") else {},
        }
        for e in entries
        if e.kind != "retired"
    ]


@router.get("/commanders/{entry_id}")
def get_commander(entry_id: str) -> dict[str, Any]:
    """Get commander profile with full stats."""
    if _league is None:
        raise HTTPException(503, "League not initialized")

    entry = _league.entries.get(entry_id)
    if entry is None:
        raise HTTPException(404, f"Commander '{entry_id}' not found")

    return {
        "entry_id": entry.entry_id,
        "commander_name": entry.commander_name,
        "kind": entry.kind,
        "rating": entry.rating.rating,
        "rd": entry.rating.rd,
        "games": entry.games,
        "rating_history": getattr(entry, "rating_history", []),
        "aggregate_stats": entry.aggregate_stats.to_dict() if hasattr(entry, "aggregate_stats") else {},
    }
```

Update `echelon/server/__init__.py` to register the router and add `create_app` factory.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/server/test_league_routes.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add echelon/server/routes/league.py echelon/server/__init__.py tests/unit/server/test_league_routes.py
git commit -m "feat(server): add league API endpoints"
```

---

### Task 6: Add match API routes

**Files:**
- Create: `echelon/server/routes/matches.py`
- Modify: `echelon/server/__init__.py` (register router)
- Test: `tests/unit/server/test_match_routes.py`

**Step 1: Write the failing test**

```python
# tests/unit/server/test_match_routes.py
"""Tests for match API endpoints."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from echelon.server import create_app
from echelon.arena.league import League
from echelon.arena.history import MatchHistory
from echelon.arena.stats import MatchRecord, TeamStats


@pytest.fixture
def app_with_matches(tmp_path: Path):
    """Create app with test league and matches."""
    league_path = tmp_path / "league.json"
    league = League.create_empty(league_path)
    league.bootstrap_heuristic()
    league.save()

    matches_path = tmp_path / "matches"
    history = MatchHistory(matches_path)

    # Add some test matches
    for i in range(3):
        record = MatchRecord(
            match_id=f"match-{i:03d}",
            timestamp=1703750400.0 + i * 100,
            blue_entry_id="contender",
            red_entry_id="heuristic",
            winner="blue" if i % 2 == 0 else "red",
            duration_steps=500 + i * 10,
            termination="zone",
            blue_stats=TeamStats(kills=2 + i, deaths=1),
            red_stats=TeamStats(kills=1, deaths=2 + i),
        )
        history.save(record)

    app = create_app(league_path=league_path, matches_path=matches_path)
    return TestClient(app)


def test_get_matches(app_with_matches):
    """GET /api/matches returns recent matches."""
    response = app_with_matches.get("/api/matches")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    # Most recent first
    assert data[0]["match_id"] == "match-002"


def test_get_matches_filtered(app_with_matches):
    """GET /api/matches?entry_id=X filters by commander."""
    response = app_with_matches.get("/api/matches?entry_id=heuristic")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3  # All matches involve heuristic


def test_get_match_detail(app_with_matches):
    """GET /api/matches/{id} returns full match detail."""
    response = app_with_matches.get("/api/matches/match-001")
    assert response.status_code == 200
    data = response.json()
    assert data["match_id"] == "match-001"
    assert "blue_stats" in data
    assert "red_stats" in data


def test_get_match_not_found(app_with_matches):
    """GET /api/matches/{id} returns 404 for unknown match."""
    response = app_with_matches.get("/api/matches/nonexistent")
    assert response.status_code == 404
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/server/test_match_routes.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# echelon/server/routes/matches.py
"""Match API endpoints."""

from fastapi import APIRouter, HTTPException, Query
from typing import Any

router = APIRouter(prefix="/api", tags=["matches"])

_history = None


def init_match_routes(history) -> None:
    """Initialize routes with history instance."""
    global _history
    _history = history


@router.get("/matches")
def get_matches(
    limit: int = Query(50, ge=1, le=200),
    entry_id: str | None = None,
) -> list[dict[str, Any]]:
    """Get recent matches."""
    if _history is None:
        raise HTTPException(503, "Match history not initialized")

    records = _history.list_recent(limit=limit, entry_id=entry_id)
    return [r.to_dict() for r in records]


@router.get("/matches/{match_id}")
def get_match(match_id: str) -> dict[str, Any]:
    """Get single match detail."""
    if _history is None:
        raise HTTPException(503, "Match history not initialized")

    record = _history.load(match_id)
    if record is None:
        raise HTTPException(404, f"Match '{match_id}' not found")

    return record.to_dict()
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/server/test_match_routes.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add echelon/server/routes/matches.py tests/unit/server/test_match_routes.py
git commit -m "feat(server): add match API endpoints"
```

---

### Task 7: Add live matchup endpoint

**Files:**
- Create: `echelon/server/routes/live.py`
- Test: `tests/unit/server/test_live_routes.py`

**Step 1: Write the failing test**

```python
# tests/unit/server/test_live_routes.py
"""Tests for live training status endpoints."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from echelon.server import create_app
from echelon.arena.league import League


@pytest.fixture
def app_client(tmp_path: Path):
    """Create app with test league."""
    league_path = tmp_path / "league.json"
    league = League.create_empty(league_path)
    league.bootstrap_heuristic()
    league.save()

    app = create_app(league_path=league_path, matches_path=tmp_path / "matches")
    return TestClient(app)


def test_get_live_matchup_no_training(app_client):
    """GET /api/live/matchup returns null when not training."""
    response = app_client.get("/api/live/matchup")
    assert response.status_code == 200
    data = response.json()
    assert data["active"] is False


def test_set_and_get_live_matchup(app_client):
    """Live matchup can be set and retrieved."""
    # POST to set current matchup (internal endpoint)
    response = app_client.post("/api/live/matchup", json={
        "blue_entry_id": "contender",
        "blue_rating": 1450.0,
        "red_entry_id": "heuristic",
        "red_rating": 1500.0,
    })
    assert response.status_code == 200

    # GET to retrieve
    response = app_client.get("/api/live/matchup")
    assert response.status_code == 200
    data = response.json()
    assert data["active"] is True
    assert data["blue_entry_id"] == "contender"
    assert data["red_entry_id"] == "heuristic"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/server/test_live_routes.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# echelon/server/routes/live.py
"""Live training status endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any

router = APIRouter(prefix="/api/live", tags=["live"])

# In-memory state for current matchup
_current_matchup: dict[str, Any] | None = None


class MatchupUpdate(BaseModel):
    """Payload for updating current matchup."""
    blue_entry_id: str
    blue_rating: float
    red_entry_id: str
    red_rating: float
    blue_stats_preview: dict[str, Any] | None = None
    red_stats_preview: dict[str, Any] | None = None


@router.get("/matchup")
def get_live_matchup() -> dict[str, Any]:
    """Get current training matchup."""
    if _current_matchup is None:
        return {"active": False}

    return {"active": True, **_current_matchup}


@router.post("/matchup")
def set_live_matchup(update: MatchupUpdate) -> dict[str, str]:
    """Set current training matchup (called by training loop)."""
    global _current_matchup
    _current_matchup = update.model_dump()
    return {"status": "ok"}


@router.delete("/matchup")
def clear_live_matchup() -> dict[str, str]:
    """Clear current matchup (training ended)."""
    global _current_matchup
    _current_matchup = None
    return {"status": "ok"}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/server/test_live_routes.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add echelon/server/routes/live.py tests/unit/server/test_live_routes.py
git commit -m "feat(server): add live matchup endpoint"
```

---

## Phase 4: Stats Collection in Training

### Task 8: Create MatchStatsCollector

**Files:**
- Create: `echelon/training/stats_collector.py`
- Test: `tests/unit/training/test_stats_collector.py`

**Step 1: Write the failing test**

```python
# tests/unit/training/test_stats_collector.py
"""Tests for match stats collection during training."""

import pytest
import numpy as np
from echelon.training.stats_collector import MatchStatsCollector


def test_collector_tracks_kills():
    """Collector accumulates kill events."""
    collector = MatchStatsCollector(num_envs=1)

    # Simulate kill event in info
    info = {
        "events": [
            {"type": "kill", "killer": "blue_0", "victim": "red_0", "team": "blue"}
        ]
    }
    collector.on_step(env_idx=0, info=info)

    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["kills"] == 1
    assert stats["red"]["deaths"] == 1


def test_collector_tracks_damage():
    """Collector accumulates damage events."""
    collector = MatchStatsCollector(num_envs=1)

    info = {
        "events": [
            {"type": "damage", "attacker": "blue_1", "target": "red_2",
             "amount": 25.0, "attacker_team": "blue"}
        ]
    }
    collector.on_step(env_idx=0, info=info)

    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["damage_dealt"] == 25.0
    assert stats["red"]["damage_taken"] == 25.0


def test_collector_tracks_weapon_usage():
    """Collector counts weapon activations."""
    collector = MatchStatsCollector(num_envs=1)

    info = {
        "events": [
            {"type": "weapon_fired", "agent": "blue_0", "weapon": "primary", "team": "blue"},
            {"type": "weapon_fired", "agent": "blue_0", "weapon": "primary", "team": "blue"},
            {"type": "weapon_fired", "agent": "blue_1", "weapon": "secondary", "team": "blue"},
        ]
    }
    collector.on_step(env_idx=0, info=info)

    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["primary_uses"] == 2
    assert stats["blue"]["secondary_uses"] == 1


def test_collector_resets_on_episode_end():
    """Collector resets stats when episode ends."""
    collector = MatchStatsCollector(num_envs=1)

    info = {"events": [{"type": "kill", "killer": "blue_0", "victim": "red_0", "team": "blue"}]}
    collector.on_step(env_idx=0, info=info)

    # End episode
    record = collector.on_episode_end(
        env_idx=0,
        winner="blue",
        termination="zone",
        duration_steps=100,
        blue_entry_id="contender",
        red_entry_id="heuristic",
    )

    assert record.blue_stats.kills == 1

    # Stats should be reset
    stats = collector.get_current_stats(env_idx=0)
    assert stats["blue"]["kills"] == 0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/unit/training/test_stats_collector.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write implementation**

```python
# echelon/training/stats_collector.py
"""Match statistics collection during training."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from echelon.arena.stats import MatchRecord, TeamStats


@dataclass
class EpisodeStats:
    """Accumulator for a single episode's stats."""

    blue: dict[str, Any] = field(default_factory=lambda: _empty_stats())
    red: dict[str, Any] = field(default_factory=lambda: _empty_stats())
    step_count: int = 0


def _empty_stats() -> dict[str, Any]:
    return {
        "kills": 0,
        "deaths": 0,
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "zone_ticks": 0,
        "primary_uses": 0,
        "secondary_uses": 0,
        "tertiary_uses": 0,
        "vents": 0,
        "smokes": 0,
        "ecm_toggles": 0,
        "paints": 0,
        "overheats": 0,
        "knockdowns": 0,
        "orders_issued": {},
        "orders_acknowledged": 0,
        "orders_overridden": 0,
    }


class MatchStatsCollector:
    """Collects match statistics from training environments."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.episodes: dict[int, EpisodeStats] = {
            i: EpisodeStats() for i in range(num_envs)
        }

    def on_step(self, env_idx: int, info: dict[str, Any]) -> None:
        """Process events from a single step."""
        ep = self.episodes[env_idx]
        ep.step_count += 1

        events = info.get("events", [])
        for event in events:
            self._process_event(ep, event)

    def _process_event(self, ep: EpisodeStats, event: dict[str, Any]) -> None:
        """Process a single event."""
        etype = event.get("type")

        if etype == "kill":
            team = event.get("team", "blue")
            target_team = "red" if team == "blue" else "blue"
            ep_stats = ep.blue if team == "blue" else ep.red
            target_stats = ep.red if team == "blue" else ep.blue
            ep_stats["kills"] += 1
            target_stats["deaths"] += 1

        elif etype == "damage":
            team = event.get("attacker_team", "blue")
            amount = event.get("amount", 0.0)
            attacker_stats = ep.blue if team == "blue" else ep.red
            target_stats = ep.red if team == "blue" else ep.blue
            attacker_stats["damage_dealt"] += amount
            target_stats["damage_taken"] += amount

        elif etype == "weapon_fired":
            team = event.get("team", "blue")
            weapon = event.get("weapon", "primary")
            stats = ep.blue if team == "blue" else ep.red
            key = f"{weapon}_uses"
            if key in stats:
                stats[key] += 1

        elif etype == "utility_used":
            team = event.get("team", "blue")
            utility = event.get("utility", "")
            stats = ep.blue if team == "blue" else ep.red
            if utility in stats:
                stats[utility] += 1

        elif etype == "zone_tick":
            team = event.get("team", "blue")
            stats = ep.blue if team == "blue" else ep.red
            stats["zone_ticks"] += 1

        elif etype == "order_issued":
            team = event.get("team", "blue")
            order_type = event.get("order_type", "unknown")
            stats = ep.blue if team == "blue" else ep.red
            stats["orders_issued"][order_type] = stats["orders_issued"].get(order_type, 0) + 1

        elif etype == "order_response":
            team = event.get("team", "blue")
            acknowledged = event.get("acknowledged", False)
            stats = ep.blue if team == "blue" else ep.red
            if acknowledged:
                stats["orders_acknowledged"] += 1
            else:
                stats["orders_overridden"] += 1

    def get_current_stats(self, env_idx: int) -> dict[str, dict[str, Any]]:
        """Get current accumulated stats for an environment."""
        ep = self.episodes[env_idx]
        return {"blue": ep.blue.copy(), "red": ep.red.copy()}

    def on_episode_end(
        self,
        env_idx: int,
        winner: str,
        termination: str,
        duration_steps: int,
        blue_entry_id: str,
        red_entry_id: str,
    ) -> MatchRecord:
        """Finalize episode stats and return MatchRecord."""
        ep = self.episodes[env_idx]

        record = MatchRecord(
            match_id=str(uuid.uuid4()),
            timestamp=time.time(),
            blue_entry_id=blue_entry_id,
            red_entry_id=red_entry_id,
            winner=winner,
            duration_steps=duration_steps,
            termination=termination,
            blue_stats=TeamStats(**{k: v for k, v in ep.blue.items()}),
            red_stats=TeamStats(**{k: v for k, v in ep.red.items()}),
        )

        # Reset for next episode
        self.episodes[env_idx] = EpisodeStats()

        return record
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/training/test_stats_collector.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add echelon/training/stats_collector.py tests/unit/training/test_stats_collector.py
git commit -m "feat(training): add MatchStatsCollector for training stats"
```

---

### Task 9: Emit events from EchelonEnv

**Files:**
- Modify: `echelon/env/env.py`
- Test: `tests/unit/env/test_env_events.py`

**Step 1: Write the failing test**

```python
# tests/unit/env/test_env_events.py
"""Tests for event emission from EchelonEnv."""

import pytest
from echelon.env.env import EchelonEnv
from echelon.config import EnvConfig, WorldConfig


@pytest.fixture
def env():
    """Create a test environment."""
    env = EchelonEnv(
        env_config=EnvConfig(world=WorldConfig(size=40)),
        render_mode=None,
    )
    env.reset()
    yield env
    env.close()


def test_env_emits_events_in_info(env):
    """Environment includes events list in info dict."""
    actions = {aid: env.action_space.sample() for aid in env.agents}
    _, _, _, _, infos = env.step(actions)

    # Should have events key (possibly empty)
    assert "events" in infos or all("events" in infos.get(aid, {}) for aid in env.agents)


def test_env_emits_damage_events(env):
    """Damage events are emitted when mechs take damage."""
    # This test requires setting up a scenario where damage occurs
    # For now, just verify the structure exists
    actions = {aid: env.action_space.sample() for aid in env.agents}

    # Run several steps to increase chance of combat
    all_events = []
    for _ in range(50):
        _, _, _, _, infos = env.step(actions)
        if "events" in infos:
            all_events.extend(infos["events"])

    # At minimum, structure should be correct
    for event in all_events:
        assert "type" in event
```

**Step 2: Run test to verify current behavior**

Run: `PYTHONPATH=. uv run pytest tests/unit/env/test_env_events.py -v`
Note: This may pass or fail depending on current event emission.

**Step 3: Add event emission to env.py**

Add an `_events` list that accumulates events during `step()`, and include it in the returned info dict.

Key emission points:
- `_process_combat()` → emit damage/kill events
- `_process_weapons()` → emit weapon_fired events
- `_process_zone()` → emit zone_tick events
- `_process_commands()` → emit order_issued events
- `_process_status_reports()` → emit order_response events

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/unit/env/test_env_events.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add echelon/env/env.py tests/unit/env/test_env_events.py
git commit -m "feat(env): emit structured events for stats collection"
```

---

## Phase 5: Dashboard UI

### Task 10: Create dashboard.html

**Files:**
- Create: `dashboard.html`

**Step 1: Create basic HTML structure**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Echelon Arena</title>
    <style>
        :root {
            --bg: #1a1a2e;
            --surface: #16213e;
            --primary: #0f3460;
            --accent: #e94560;
            --text: #eee;
            --text-dim: #888;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 2rem;
        }

        h1 {
            font-size: 2rem;
            margin-bottom: 1.5rem;
            color: var(--accent);
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        .card {
            background: var(--surface);
            border-radius: 8px;
            padding: 1.5rem;
        }

        .card h2 {
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-dim);
            margin-bottom: 1rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--primary);
        }

        th { color: var(--text-dim); font-weight: 500; }

        .rating {
            font-weight: bold;
            font-variant-numeric: tabular-nums;
        }

        .sparkline {
            width: 80px;
            height: 24px;
        }

        .match-list {
            list-style: none;
        }

        .match-list li {
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--primary);
            display: flex;
            justify-content: space-between;
        }

        .win { color: #4ade80; }
        .loss { color: var(--accent); }
        .draw { color: var(--text-dim); }

        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <h1>Echelon Arena</h1>

    <div class="grid">
        <div class="card">
            <h2>Leaderboard</h2>
            <table id="standings">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Commander</th>
                        <th>Rating</th>
                        <th>Games</th>
                        <th>K/D</th>
                        <th>History</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>

        <div class="card">
            <h2>Recent Matches</h2>
            <ul id="matches" class="match-list"></ul>
        </div>
    </div>

    <script>
        const API_BASE = '';  // Same origin

        async function fetchStandings() {
            const res = await fetch(`${API_BASE}/api/league/standings`);
            const data = await res.json();

            const tbody = document.querySelector('#standings tbody');
            tbody.innerHTML = data.map((entry, i) => {
                const kd = entry.aggregate_stats.deaths > 0
                    ? (entry.aggregate_stats.kills / entry.aggregate_stats.deaths).toFixed(2)
                    : entry.aggregate_stats.kills.toFixed(0);
                return `
                    <tr>
                        <td>${i + 1}</td>
                        <td>${entry.commander_name}</td>
                        <td class="rating">${Math.round(entry.rating)}</td>
                        <td>${entry.games}</td>
                        <td>${kd}</td>
                        <td><canvas class="sparkline" data-id="${entry.entry_id}"></canvas></td>
                    </tr>
                `;
            }).join('');

            // Fetch rating history for sparklines
            data.forEach(entry => fetchSparkline(entry.entry_id));
        }

        async function fetchSparkline(entryId) {
            const res = await fetch(`${API_BASE}/api/commanders/${entryId}`);
            const data = await res.json();

            const canvas = document.querySelector(`canvas[data-id="${entryId}"]`);
            if (!canvas || !data.rating_history?.length) return;

            const ctx = canvas.getContext('2d');
            const history = data.rating_history;
            const ratings = history.map(h => h[1]);
            const min = Math.min(...ratings) - 50;
            const max = Math.max(...ratings) + 50;

            ctx.strokeStyle = '#e94560';
            ctx.lineWidth = 2;
            ctx.beginPath();

            ratings.forEach((r, i) => {
                const x = (i / (ratings.length - 1)) * canvas.width;
                const y = canvas.height - ((r - min) / (max - min)) * canvas.height;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });

            ctx.stroke();
        }

        async function fetchMatches() {
            const res = await fetch(`${API_BASE}/api/matches?limit=10`);
            const data = await res.json();

            const ul = document.getElementById('matches');
            ul.innerHTML = data.map(match => {
                const blueWon = match.winner === 'blue';
                const redWon = match.winner === 'red';
                return `
                    <li>
                        <span class="${blueWon ? 'win' : redWon ? 'loss' : 'draw'}">
                            ${match.blue_entry_id}
                        </span>
                        <span>vs</span>
                        <span class="${redWon ? 'win' : blueWon ? 'loss' : 'draw'}">
                            ${match.red_entry_id}
                        </span>
                        <span>${match.termination}</span>
                    </li>
                `;
            }).join('');
        }

        // Initial load
        fetchStandings();
        fetchMatches();

        // Refresh every 30 seconds
        setInterval(() => {
            fetchStandings();
            fetchMatches();
        }, 30000);
    </script>
</body>
</html>
```

**Step 2: Test manually**

Start the server and open dashboard.html in browser.
Verify leaderboard loads and displays data.

**Step 3: Commit**

```bash
git add dashboard.html
git commit -m "feat: add arena dashboard HTML"
```

---

### Task 11: Add overlay to viewer.html

**Files:**
- Modify: `viewer.html`

**Step 1: Add overlay HTML/CSS**

Add a toggleable panel to the existing viewer:

```html
<!-- Add inside viewer.html body -->
<div id="matchup-overlay" class="overlay hidden">
    <div class="matchup-blue">
        <span class="team-label">BLUE</span>
        <span class="commander-name" id="blue-name">-</span>
        <span class="rating" id="blue-rating">-</span>
    </div>
    <div class="matchup-vs">VS</div>
    <div class="matchup-red">
        <span class="team-label">RED</span>
        <span class="commander-name" id="red-name">-</span>
        <span class="rating" id="red-rating">-</span>
    </div>
    <div class="stat-contrast" id="stat-contrast"></div>
</div>
```

```css
/* Add to viewer.html styles */
.overlay {
    position: fixed;
    bottom: 20px;
    left: 20px;
    background: rgba(0,0,0,0.8);
    padding: 1rem;
    border-radius: 8px;
    font-family: monospace;
    z-index: 1000;
}

.overlay.hidden { display: none; }

.matchup-blue, .matchup-red {
    display: flex;
    gap: 1rem;
    align-items: center;
}

.matchup-blue .team-label { color: #60a5fa; }
.matchup-red .team-label { color: #f87171; }

.matchup-vs {
    text-align: center;
    color: #888;
    padding: 0.25rem 0;
}

.rating {
    font-weight: bold;
}

.stat-contrast {
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #888;
}
```

**Step 2: Add JavaScript to poll live endpoint**

```javascript
// Add to viewer.html script
async function updateMatchupOverlay() {
    try {
        const res = await fetch('/api/live/matchup');
        const data = await res.json();

        const overlay = document.getElementById('matchup-overlay');

        if (!data.active) {
            overlay.classList.add('hidden');
            return;
        }

        overlay.classList.remove('hidden');
        document.getElementById('blue-name').textContent = data.blue_entry_id;
        document.getElementById('blue-rating').textContent = Math.round(data.blue_rating);
        document.getElementById('red-name').textContent = data.red_entry_id;
        document.getElementById('red-rating').textContent = Math.round(data.red_rating);

    } catch (e) {
        console.warn('Failed to fetch matchup:', e);
    }
}

// Poll every 5 seconds
setInterval(updateMatchupOverlay, 5000);
updateMatchupOverlay();

// Toggle with 'M' key
document.addEventListener('keydown', (e) => {
    if (e.key === 'm' || e.key === 'M') {
        document.getElementById('matchup-overlay').classList.toggle('hidden');
    }
});
```

**Step 3: Test manually**

Open viewer.html, press 'M' to toggle overlay.

**Step 4: Commit**

```bash
git add viewer.html
git commit -m "feat(viewer): add matchup overlay (toggle with M key)"
```

---

## Phase 6: Integration

### Task 12: Integrate stats collection into train_ppo.py

**Files:**
- Modify: `scripts/train_ppo.py`

**Step 1: Import and initialize collector**

```python
from echelon.training.stats_collector import MatchStatsCollector
from echelon.arena.history import MatchHistory

# In main():
stats_collector = MatchStatsCollector(num_envs=args.num_envs)
match_history = MatchHistory(Path(args.run_dir) / "matches")
```

**Step 2: Call collector on each step**

In the training loop, after `venv.step()`:
```python
# Collect stats from infos
for env_idx, info in enumerate(infos):
    stats_collector.on_step(env_idx, info)
```

**Step 3: Save match record on episode end**

When episode ends:
```python
if ep_over:
    record = stats_collector.on_episode_end(
        env_idx=env_idx,
        winner=winner,
        termination=termination,
        duration_steps=ep_len,
        blue_entry_id="contender",
        red_entry_id=current_opponent_id,
    )
    match_history.save(record)

    # Update league entry stats if arena mode
    if args.train_mode == "arena" and current_opponent_id in league.entries:
        league.entries[current_opponent_id].add_match_stats(record.red_stats)
```

**Step 4: Test end-to-end**

Run short training:
```bash
uv run python scripts/train_ppo.py --total-steps 10000 --num-envs 2
ls runs/train/matches/  # Should see JSON files
```

**Step 5: Commit**

```bash
git add scripts/train_ppo.py
git commit -m "feat(training): integrate stats collection into PPO training"
```

---

### Task 13: Wire up server with league and history

**Files:**
- Modify: `echelon/server/__init__.py`
- Modify: `echelon/server/__main__.py`

**Step 1: Update create_app factory**

```python
def create_app(
    league_path: Path | None = None,
    matches_path: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Echelon")

    # Initialize league routes if path provided
    if league_path and league_path.exists():
        from echelon.arena.league import League
        from echelon.arena.history import MatchHistory
        from .routes.league import init_league_routes, router as league_router
        from .routes.matches import init_match_routes, router as matches_router
        from .routes.live import router as live_router

        league = League.load(league_path)
        history = MatchHistory(matches_path or league_path.parent / "matches")

        init_league_routes(league, history)
        init_match_routes(history)

        app.include_router(league_router)
        app.include_router(matches_router)
        app.include_router(live_router)

    # Existing routes...
    return app
```

**Step 2: Add CLI args to __main__.py**

```python
parser.add_argument("--league", type=Path, default=Path("runs/arena/league.json"))
parser.add_argument("--matches", type=Path, default=None)
```

**Step 3: Test server**

```bash
uv run python -m echelon.server --league runs/arena/league.json
curl http://localhost:8090/api/league/standings
```

**Step 4: Commit**

```bash
git add echelon/server/__init__.py echelon/server/__main__.py
git commit -m "feat(server): wire up league and match history APIs"
```

---

## Summary

**Total Tasks:** 13
**Estimated Time:** 3-4 hours (with testing)

**Key Deliverables:**
1. `MatchRecord` and `TeamStats` data models
2. `MatchHistory` file-based storage
3. `MatchStatsCollector` for training integration
4. 8 API endpoints on existing server
5. `dashboard.html` leaderboard page
6. Viewer overlay with live matchup

**Not Included (deferred):**
- Head-to-head records
- Match replay linking
- Database backend
- Authentication
