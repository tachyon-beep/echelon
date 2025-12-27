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

    def save(self, record: MatchRecord) -> None:
        """Save a match record to disk."""
        path = self._path_for(record.match_id)
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

    def load(self, match_id: str) -> MatchRecord | None:
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
    ) -> list[MatchRecord]:
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
