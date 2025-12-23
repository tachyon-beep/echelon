# Improvement Roadmap

**Project:** Echelon
**Analysis Date:** 2025-12-24
**Analyst:** System Architect
**Status:** FINAL RECOMMENDATION

---

## Executive Summary

Echelon has **no critical security vulnerabilities**. Prioritization follows standard risk-based hierarchy with architecture/maintainability as the primary focus.

**Phase 1 focus:** Training infrastructure refactoring (TD-001) - this unblocks the majority of ROADMAP "Future Vision" items.

---

## Priority Hierarchy Applied

Since no security issues exist, priority order is:

1. ~~Critical Security~~ - None
2. ~~High Security~~ - None
3. ~~System Reliability~~ - None
4. **Architecture Debt** - TD-001 (train_ppo.py)
5. **Code Quality** - TD-002, TD-003, TD-004, TD-005
6. **Documentation** - TD-007
7. ~~Features~~ - Separate from debt remediation

---

## Phase 1: Training Infrastructure (BLOCKING)

**Duration:** 4-6 hours
**Items:** TD-001
**Goal:** Unblock ROADMAP extensibility

### TD-001: Extract Training Package

**Current State:**
```
scripts/train_ppo.py (1,442 LOC)
    - VectorEnv wrapper
    - PPO algorithm
    - Rollout buffer
    - Worker management
    - Checkpoint handling
    - W&B integration
```

**Target State:**
```
echelon/training/
├── __init__.py
├── vec_env.py          # VectorEnv wrapper (~200 LOC)
├── ppo.py              # PPO trainer class (~400 LOC)
├── rollout.py          # Rollout buffer (~150 LOC)
├── worker.py           # Multiprocessing (~200 LOC)
└── checkpoint.py       # Save/load (~100 LOC)

scripts/train_ppo.py    # Thin CLI wrapper (~200 LOC)
```

**Verification:**
- [ ] `PYTHONPATH=. uv run pytest tests/` passes
- [ ] `uv run python scripts/train_ppo.py --updates 10 --size 40` produces same results
- [ ] Each extracted module has docstrings
- [ ] W&B integration still works

**Unblocks:**
- Curriculum learning implementation
- Alternative algorithms (SAC, A2C)
- Experiment templates (ROADMAP)
- Unit testing of training components

---

## Phase 2: Testing Confidence

**Duration:** 3-4 hours
**Items:** TD-002, TD-004
**Goal:** Enable confident modifications to self-play

### TD-002: Arena Unit Tests

**Test Cases Required:**

```python
# tests/unit/test_glicko2.py
def test_rating_update_win():
    """Winner gains rating, loser loses."""

def test_rating_update_upset():
    """Low-rated beating high-rated causes larger swing."""

def test_rating_bounds():
    """Rating stays within valid range."""

def test_deviation_decreases_with_games():
    """More games = more certain rating."""

# tests/unit/test_league.py
def test_promotion_requires_top_k():
    """Only top-K candidates promote."""

def test_two_phase_commit():
    """Ratings update atomically."""

def test_opponent_sampling_excludes_self():
    """Never face yourself."""

def test_snapshot_rating_captures_at_match_time():
    """GameResult uses rating at match time, not current."""
```

**Verification:**
- [ ] `PYTHONPATH=. uv run pytest tests/unit/test_glicko2.py -v` passes
- [ ] `PYTHONPATH=. uv run pytest tests/unit/test_league.py -v` passes
- [ ] Coverage report shows arena/ > 80%

### TD-004: PyTorch weights_only Migration

**Changes:**
1. Register safe globals for model loading
2. Update `torch.load()` calls to use `weights_only=True`
3. Test model loading/saving roundtrip

**Verification:**
- [ ] Model save/load works with `weights_only=True`
- [ ] No deprecation warnings from PyTorch

---

## Phase 3: Code Quality

**Duration:** 4-5 hours
**Items:** TD-003, TD-005
**Goal:** Improve maintainability for balance tuning

### TD-003: Physics Constants

**Extract to `echelon/constants.py`:**

```python
# Physics
GRAVITY = 9.81  # m/s^2
DIAGONAL_FACTOR = 0.707  # 1/sqrt(2) for diagonal movement normalization

# Combat multipliers
REAR_ARMOR_MULTIPLIER = 1.5
CRITICAL_HIT_MULTIPLIER = 2.0
HEAT_DAMAGE_THRESHOLD = 0.8  # % of heat capacity

# Movement
MAX_STEP_HEIGHT = 2  # voxels
FOOTPRINT_RADIUS = 1  # voxels for erosion
```

**Verification:**
- [ ] All magic numbers in sim.py replaced with constants
- [ ] `uv run ruff check .` passes
- [ ] `PYTHONPATH=. uv run pytest tests/unit/test_mechanics.py` passes

### TD-005: Observation Builder Extraction

**Extract from `env.py` to `echelon/env/observation.py`:**

```python
@dataclass(frozen=True)
class ObservationConfig:
    contact_slots: int = 5
    contact_dims: int = 22
    pack_size: int = 10
    comm_dims: int = 8
    local_map_size: int = 11
    telemetry_size: int = 16
    self_dims: int = 40

class ObservationBuilder:
    def __init__(self, config: ObservationConfig): ...
    def build_contacts(self, mech: MechState, sim: Sim) -> np.ndarray: ...
    def build_pack_comm(self, mech: MechState, pack: list[MechState]) -> np.ndarray: ...
    def build_local_map(self, mech: MechState, world: VoxelWorld) -> np.ndarray: ...
    def build_telemetry(self, world: VoxelWorld) -> np.ndarray: ...
    def build_self_features(self, mech: MechState) -> np.ndarray: ...
    def build(self, mech: MechState, sim: Sim, pack: list[MechState], world: VoxelWorld) -> np.ndarray: ...
```

**Verification:**
- [ ] `env.py` LOC reduced by ~200
- [ ] `PYTHONPATH=. uv run pytest tests/integration/` passes
- [ ] Observation dimensions unchanged (607)

---

## Phase 4: Documentation & Polish

**Duration:** 2-3 hours
**Items:** TD-006 (partial), TD-007
**Goal:** Long-term maintainability

### TD-007: README Architecture Section

**Options:**
1. **Minimal:** Link to `docs/arch-analysis-2025-12-24-0627/04-final-report.md`
2. **Inline:** Add condensed architecture summary to README

**Recommended:** Option 1 (link) - keeps README concise, architecture docs are comprehensive.

### TD-006: Mypy Strict Mode (Incremental)

**Phase 4 scope:** Enable strict mode with per-module ignores

```toml
# pyproject.toml
[tool.mypy]
strict = true

[[tool.mypy.overrides]]
module = [
    "echelon.sim.sim",
    "echelon.env.env",
    "scripts.*",
]
ignore_errors = true
```

**Then:** Remove module ignores as each module is touched in normal development.

---

## Timeline Summary

| Phase | Items | Effort | Dependency |
|-------|-------|--------|------------|
| **Phase 1** | TD-001 | 4-6 hrs | None |
| **Phase 2** | TD-002, TD-004 | 3-4 hrs | None (parallel OK) |
| **Phase 3** | TD-003, TD-005 | 4-5 hrs | None (parallel OK) |
| **Phase 4** | TD-006, TD-007 | 2-3 hrs | None |

**Total:** ~15-18 hours

**Recommended Execution:**
- Phase 1 first (blocking)
- Phases 2 & 3 can run in parallel
- Phase 4 whenever convenient

---

## ROADMAP Alignment

| ROADMAP Item | Blocked By | Unblocked After |
|--------------|------------|-----------------|
| Experiment templates | TD-001 | Phase 1 |
| Curriculum learning | TD-001 | Phase 1 |
| Alternative algorithms | TD-001 | Phase 1 |
| Population-based training | TD-002 | Phase 2 |
| Interactive tuning | TD-001, TD-003 | Phases 1 + 3 |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Phase 1 refactor introduces bugs | Existing tests + determinism checks |
| Arena tests reveal Glicko-2 bugs | Fix bugs before proceeding (good!) |
| Observation builder extraction breaks dims | Property-based tests on observation shape |

---

## Decision Log

**Decision:** Phase 1 = Architecture (TD-001), not security

**Rationale:** No security vulnerabilities identified. Architecture debt (train_ppo.py) blocks the most ROADMAP items.

**Security Priority Maintained:** N/A (no security debt exists)

---

## Appendix: Quick Reference

### Phase 1 Commands
```bash
# After extraction, verify:
PYTHONPATH=. uv run pytest tests/
uv run python scripts/train_ppo.py --updates 10 --size 40
uv run ruff check .
uv run mypy echelon/training/
```

### Phase 2 Commands
```bash
# After arena tests:
PYTHONPATH=. uv run pytest tests/unit/test_glicko2.py tests/unit/test_league.py -v
PYTHONPATH=. uv run pytest --cov=echelon/arena
```

### Phase 3 Commands
```bash
# After constants extraction:
uv run ruff check echelon/constants.py
PYTHONPATH=. uv run pytest tests/unit/test_mechanics.py

# After observation builder:
PYTHONPATH=. uv run pytest tests/integration/
```

---

*Generated by System Architect on 2025-12-24*
*Security Priority: N/A (no security debt identified)*
