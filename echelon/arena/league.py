from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib

from .glicko2 import GameResult, Glicko2Config, Glicko2Rating, rate


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_id(path: Path) -> str:
    # Use a stable id derived from the checkpoint filename + size/mtime for uniqueness without hashing.
    stat = path.stat()
    return f"{path.stem}:{stat.st_size}:{int(stat.st_mtime)}"


def _commander_name(seed: int) -> str:
    adjectives = [
        "Iron",
        "Crimson",
        "Silent",
        "Vigilant",
        "Rogue",
        "Solar",
        "Ashen",
        "Emerald",
        "Ivory",
        "Obsidian",
        "Kestrel",
        "Phantom",
        "Gilded",
        "Static",
        "Tempest",
    ]
    nouns = [
        "Warden",
        "Viper",
        "Lancer",
        "Marauder",
        "Pilgrim",
        "Sentinel",
        "Specter",
        "Bastion",
        "Harrier",
        "Orchid",
        "Anvil",
        "Fox",
        "Raptor",
        "Nova",
        "Cairn",
    ]
    adj = adjectives[seed % len(adjectives)]
    noun = nouns[(seed // len(adjectives)) % len(nouns)]
    tag = (seed * 2654435761) & 0xFFFF
    return f"{adj} {noun} #{tag:04x}"


@dataclass
class LeagueEntry:
    entry_id: str
    ckpt_path: str
    kind: str  # "candidate" | "commander"
    commander_name: str | None = None
    created_at: str = field(default_factory=_now_iso)
    rating: Glicko2Rating = field(default_factory=Glicko2Rating)
    games: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.entry_id),
            "ckpt_path": str(self.ckpt_path),
            "kind": str(self.kind),
            "commander_name": self.commander_name,
            "created_at": str(self.created_at),
            "rating": self.rating.as_dict(),
            "games": int(self.games),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, cfg: Glicko2Config) -> "LeagueEntry":
        rating = d.get("rating") or {}
        return cls(
            entry_id=str(d.get("id")),
            ckpt_path=str(d.get("ckpt_path")),
            kind=str(d.get("kind", "candidate")),
            commander_name=d.get("commander_name"),
            created_at=str(d.get("created_at", _now_iso())),
            rating=Glicko2Rating.from_dict(rating).with_defaults(cfg),
            games=int(d.get("games", 0)),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class League:
    cfg: Glicko2Config = field(default_factory=Glicko2Config)
    entries: dict[str, LeagueEntry] = field(default_factory=dict)
    # Optional: store an env config signature to avoid mixing incompatible policies.
    env_signature: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "glicko2": {
                "tau": float(self.cfg.tau),
                "epsilon": float(self.cfg.epsilon),
                "rating0": float(self.cfg.rating0),
                "rd0": float(self.cfg.rd0),
                "vol0": float(self.cfg.vol0),
            },
            "env_signature": self.env_signature,
            "entries": [e.as_dict() for e in self.entries.values()],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "League":
        g = d.get("glicko2") or {}
        cfg = Glicko2Config(
            tau=float(g.get("tau", Glicko2Config().tau)),
            epsilon=float(g.get("epsilon", Glicko2Config().epsilon)),
            rating0=float(g.get("rating0", Glicko2Config().rating0)),
            rd0=float(g.get("rd0", Glicko2Config().rd0)),
            vol0=float(g.get("vol0", Glicko2Config().vol0)),
        )
        league = cls(cfg=cfg, env_signature=d.get("env_signature"))
        for item in d.get("entries") or []:
            entry = LeagueEntry.from_dict(item, cfg=cfg)
            league.entries[entry.entry_id] = entry
        return league

    @classmethod
    def load(cls, path: Path) -> "League":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    def upsert_checkpoint(self, ckpt_path: Path, *, kind: str) -> LeagueEntry:
        ckpt_path = ckpt_path.resolve()
        entry_id = _stable_id(ckpt_path)
        existing = self.entries.get(entry_id)
        if existing is not None:
            # Never demote a commander back to candidate by accident.
            if not (existing.kind == "commander" and kind == "candidate"):
                existing.kind = kind
            existing.ckpt_path = str(ckpt_path)
            return existing

        e = LeagueEntry(
            entry_id=entry_id,
            ckpt_path=str(ckpt_path),
            kind=kind,
            rating=Glicko2Rating(
                rating=float(self.cfg.rating0),
                rd=float(self.cfg.rd0),
                vol=float(self.cfg.vol0),
            ),
        )
        self.entries[entry_id] = e
        return e

    def top_commanders(self, n: int) -> list[LeagueEntry]:
        commanders = [e for e in self.entries.values() if e.kind == "commander"]
        commanders.sort(key=lambda e: float(e.rating.rating), reverse=True)
        return commanders[: max(0, int(n))]

    def recent_candidates(self, n: int, *, exclude_id: str | None = None) -> list[LeagueEntry]:
        cands = [e for e in self.entries.values() if e.kind == "candidate" and e.entry_id != exclude_id]
        # ISO timestamps sort lexicographically.
        cands.sort(key=lambda e: str(e.created_at), reverse=True)
        return cands[: max(0, int(n))]

    def promote_if_topk(self, entry_id: str, *, top_k: int) -> bool:
        entry = self.entries.get(entry_id)
        if entry is None:
            raise KeyError(entry_id)

        # Decide rank among (existing commanders + this candidate) by conservative score.
        def conservative(e: LeagueEntry) -> float:
            return float(e.rating.rating) - 2.0 * float(e.rating.rd)

        pool = [e for e in self.entries.values() if e.kind == "commander"]
        pool.append(entry)
        pool.sort(key=conservative, reverse=True)
        in_top = entry in pool[: max(1, int(top_k))]
        if not in_top:
            return False

        if entry.kind != "commander":
            entry.kind = "commander"
        if entry.commander_name is None:
            h = hashlib.sha256(entry.entry_id.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "big", signed=False)
            entry.commander_name = _commander_name(seed)
        return True

    def apply_rating_period(self, results: dict[str, list[GameResult]]) -> None:
        # Update only the entries that participated in this rating period.
        for entry_id, games in results.items():
            entry = self.entries.get(entry_id)
            if entry is None:
                continue
            entry.rating = rate(entry.rating, games, cfg=self.cfg)
            entry.games += len(games)
