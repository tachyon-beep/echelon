from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .glicko2 import GameResult, Glicko2Config, Glicko2Rating, rate

if TYPE_CHECKING:
    from pathlib import Path


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
    def from_dict(cls, d: dict[str, Any], *, cfg: Glicko2Config) -> LeagueEntry:
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
    def from_dict(cls, d: dict[str, Any]) -> League:
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
    def load(cls, path: Path) -> League:
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

    def add_heuristic(self) -> LeagueEntry:
        """Add or get the permanent heuristic baseline entry.

        The heuristic ("Lieutenant Heuristic") is a special commander that:
        - Always exists in the pool
        - Never retires
        - Uses venv.get_heuristic_actions() instead of model inference
        - Has normal Glicko-2 rating that updates from matches

        Returns:
            The heuristic LeagueEntry (created or existing)
        """
        entry_id = "heuristic"
        existing = self.entries.get(entry_id)
        if existing is not None:
            return existing

        entry = LeagueEntry(
            entry_id=entry_id,
            ckpt_path="",
            kind="heuristic",
            commander_name="Lieutenant Heuristic",
            rating=Glicko2Rating(
                rating=float(self.cfg.rating0),
                rd=float(self.cfg.rd0),
                vol=float(self.cfg.vol0),
            ),
        )
        self.entries[entry_id] = entry
        return entry

    def top_commanders(self, n: int) -> list[LeagueEntry]:
        # Include both commanders and heuristic entries
        commanders = [e for e in self.entries.values() if e.kind in ("commander", "heuristic")]
        commanders.sort(key=lambda e: float(e.rating.rating), reverse=True)
        return commanders[: max(0, int(n))]

    def recent_candidates(self, n: int, *, exclude_id: str | None = None) -> list[LeagueEntry]:
        cands = [e for e in self.entries.values() if e.kind == "candidate" and e.entry_id != exclude_id]
        # ISO timestamps sort lexicographically.
        cands.sort(key=lambda e: str(e.created_at), reverse=True)
        return cands[: max(0, int(n))]

    def sample_pfsp_opponent(
        self,
        pool: list[LeagueEntry],
        candidate_rating: float,
        rng: random.Random | None = None,
        sigma: float = 200.0,
        candidate_games: int | None = None,
        warmup_games: int = 20,
        warmup_rating_range: float = 200.0,
    ) -> LeagueEntry:
        """Sample opponent using Prioritized Fictitious Self-Play (PFSP).

        PFSP weights opponents by skill match - opponents with similar ratings
        provide the best learning signal. Uses a Gaussian bell curve centered
        on the candidate's rating.

        Cold-start warmup: New policies (< warmup_games) are matched only against
        opponents within warmup_rating_range to prevent being crushed by
        established commanders while their rating stabilizes.

        Reference: Vinyals et al., "Grandmaster level in StarCraft II using
        multi-agent reinforcement learning", Nature 2019.

        Args:
            pool: List of potential opponents.
            candidate_rating: Current rating of the learning agent.
            rng: Random number generator (uses global random if None).
            sigma: Standard deviation for the Gaussian weighting (default 200).
                   Larger values = more uniform sampling.
            candidate_games: Number of games played by candidate (for warmup).
            warmup_games: Number of games before full pool is available (default 20).
            warmup_rating_range: Rating range for warmup matchmaking (default 200).

        Returns:
            Selected opponent entry.
        """
        if not pool:
            raise ValueError("Pool must not be empty")

        rng = rng or random.Random()

        # Cold-start warmup: restrict pool for new policies with progressive expansion
        # Start with tight rating range, expand if needed to find at least 1 opponent
        effective_pool = pool
        if candidate_games is not None and candidate_games < warmup_games:
            # Try progressively wider rating ranges until we find at least 1 opponent
            for range_mult in [1.0, 2.0, 4.0]:
                current_range = warmup_rating_range * range_mult
                nearby = [e for e in pool if abs(e.rating.rating - candidate_rating) <= current_range]
                if nearby:
                    effective_pool = nearby
                    break
            # If still empty after 4x range, use full pool (fallback)

        if len(effective_pool) == 1:
            return effective_pool[0]

        # Compute Gaussian weights: exp(-(rating_diff)^2 / (2*sigma^2))
        weights: list[float] = []
        for entry in effective_pool:
            diff = entry.rating.rating - candidate_rating
            weight = math.exp(-(diff * diff) / (2 * sigma * sigma))
            weights.append(weight)

        # Normalize weights
        total = sum(weights)
        if total < 1e-10:
            # Fallback to uniform if all weights are ~0 (shouldn't happen with Gaussian)
            return rng.choice(effective_pool)

        # Sample using weights
        r = rng.random() * total
        cumsum = 0.0
        for entry, weight in zip(effective_pool, weights, strict=True):
            cumsum += weight
            if r <= cumsum:
                return entry

        # Fallback (shouldn't reach here due to floating point)
        return effective_pool[-1]

    def promote_if_topk(self, entry_id: str, *, top_k: int) -> bool:
        entry = self.entries.get(entry_id)
        if entry is None:
            raise KeyError(entry_id)

        # Decide rank among (existing commanders + this candidate) by conservative score.
        pool = [e for e in self.entries.values() if e.kind == "commander"]
        pool.append(entry)
        pool.sort(key=lambda e: e.rating.conservative_rating, reverse=True)
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
        """
        Apply Glicko-2 rating updates using two-phase commit (CRIT-6).

        Phase 1: Compute all new ratings using snapshotted opponent ratings.
        Phase 2: Commit all updates atomically to avoid circular dependency issues.
        """
        # Phase 1: Compute new ratings (opponent ratings are already frozen in GameResult)
        updates: dict[str, tuple[Glicko2Rating, int]] = {}
        for entry_id, games in results.items():
            entry = self.entries.get(entry_id)
            if entry is None:
                continue
            new_rating = rate(entry.rating, games, cfg=self.cfg)
            updates[entry_id] = (new_rating, len(games))

        # Phase 2: Commit all updates atomically
        for entry_id, (new_rating, game_count) in updates.items():
            entry = self.entries.get(entry_id)
            if entry is not None:
                entry.rating = new_rating
                entry.games += game_count

    def retire_commanders(
        self,
        keep_top: int = 20,
        min_games: int = 20,
    ) -> list[LeagueEntry]:
        """Retire underperforming commanders to keep pool manageable.

        Commanders are ranked by conservative rating (rating - 2*RD).
        Only commanders with sufficient games can be retired (new commanders
        are protected until their rating stabilizes).

        Args:
            keep_top: Number of top commanders to retain
            min_games: Minimum games before a commander can be retired

        Returns:
            List of retired LeagueEntry objects (removed from entries)
        """
        # Only retire regular commanders - heuristic entries are permanent
        commanders = [e for e in self.entries.values() if e.kind == "commander"]

        if len(commanders) <= keep_top:
            return []

        # Split into retirable (established) - new commanders are protected
        retirable = [e for e in commanders if e.games >= min_games]

        # Sort retirable by conservative rating (worst first)
        retirable.sort(key=lambda e: e.rating.conservative_rating)

        # Calculate how many we need to retire
        # Keep at least keep_top total, but never retire protected
        current_total = len(commanders)
        num_to_retire = max(0, current_total - keep_top)

        # Can only retire from retirable pool
        num_to_retire = min(num_to_retire, len(retirable))

        # Retire the worst performers
        retired = retirable[:num_to_retire]

        for entry in retired:
            # Change kind to "retired" rather than deleting (preserves history)
            entry.kind = "retired"

        return retired
