# Technical Debt Catalog

**Project:** Echelon
**Analysis Date:** 2025-12-24
**Analyst:** System Architect
**Coverage:** COMPLETE (all identified items analyzed)
**Confidence:** HIGH

---

## Executive Summary

Echelon has **minimal technical debt** for a codebase of its size (6,670 LOC core). No critical security issues. Primary debt is structural/maintainability-focused, concentrated in the training infrastructure.

| Priority | Count | Primary Impact |
|----------|-------|----------------|
| Critical | 0 | - |
| High | 2 | Blocks ROADMAP features, extensibility |
| Medium | 3 | Testing confidence, documentation |
| Low | 2 | Minor cleanup |

**Total Items:** 7

---

## Critical Priority (Immediate Action Required)

**None identified.**

No security vulnerabilities, data loss risks, or system failure modes detected.

---

## High Priority (Blocks ROADMAP / Extensibility)

### TD-001: Monolithic Training Script

**Evidence:** `scripts/train_ppo.py` (1,442 LOC)

**Impact:**
- **Business:** Blocks ROADMAP items: "Experiment templates", "Curriculum walkthroughs", "Interactive tuning"
- **Technical:** Cannot unit test PPO components; cannot add alternative algorithms (SAC, A2C); cannot implement curriculum learning without major refactoring

**Effort:** L (4-6 hours)

**Category:** Architecture

**Details:**
The training script mixes multiple concerns:
- VectorEnv wrapper (parallel environment execution)
- PPO algorithm implementation (GAE, clipping, optimization)
- Rollout buffer management
- Worker pool / multiprocessing orchestration
- Checkpoint management
- W&B integration
- CLI argument parsing

Each of these should be a separate, testable module.

**Recommendation:** Extract to `echelon/training/` package:
```
echelon/training/
├── __init__.py
├── vec_env.py          # VectorEnv wrapper
├── ppo.py              # PPO trainer class
├── rollout.py          # Rollout buffer
├── worker.py           # Multiprocessing workers
└── checkpoint.py       # Save/load logic
```

---

### TD-002: Arena Subsystem Has No Unit Tests

**Evidence:**
- `echelon/arena/league.py` (230 LOC) - no tests
- `echelon/arena/match.py` - no tests
- `echelon/arena/glicko2.py` - no tests
- `tests/unit/` - no test_league.py, test_match.py, or test_glicko2.py

**Impact:**
- **Business:** Cannot confidently modify self-play for population-based training (ROADMAP)
- **Technical:** Glicko-2 rating calculations, promotion logic, and two-phase commit are untested; bugs here corrupt training evaluation

**Effort:** M (2-3 hours)

**Category:** Code Quality / Testing

**Details:**
The arena subsystem handles:
- Glicko-2 rating updates (mathematical algorithm)
- Promotion thresholds (business logic)
- Two-phase commit for atomic rating updates (correctness critical)
- Opponent sampling distribution

All are testable in isolation but have no coverage.

**Recommendation:** Add `tests/unit/test_arena.py`:
```python
def test_glicko2_rating_update_win():
    """Winner gains rating, loser loses rating."""

def test_glicko2_rating_update_upset():
    """Low-rated beating high-rated causes larger swing."""

def test_promotion_requires_top_k():
    """Candidate only promotes if in conservative top-K."""

def test_two_phase_commit_atomicity():
    """All ratings update together or none do."""

def test_opponent_sampling_excludes_self():
    """Policy never faces itself."""
```

---

## Medium Priority (Maintainability / Confidence)

### TD-003: Magic Numbers in Physics Calculations

**Evidence:**
- `echelon/sim/sim.py` - scattered numeric literals
- `echelon/config.py` - some constants defined, others not

**Impact:**
- **Technical:** Balance tuning requires hunting through code for values; no single source of truth for physics parameters

**Effort:** S (1 hour)

**Category:** Code Quality

**Details:**
Examples found:
- Gravity: `9.81` used inline
- Diagonal movement factor: `0.707` (1/sqrt(2))
- Various damage multipliers hardcoded

**Recommendation:** Add to `echelon/constants.py`:
```python
# Physics
GRAVITY = 9.81  # m/s^2
DIAGONAL_FACTOR = 0.707  # 1/sqrt(2)

# Combat
REAR_ARMOR_MULTIPLIER = 1.5
CRITICAL_HIT_MULTIPLIER = 2.0
```

---

### TD-004: PyTorch weights_only Migration Pending

**Evidence:** `echelon/arena/match.py:43`
```python
# TODO: migrate to weights_only=True after adding safe globals
```

**Impact:**
- **Technical:** Future PyTorch versions may deprecate current behavior; security best practice for model loading

**Effort:** S (< 1 hour)

**Category:** Code Quality / Future-Proofing

**Details:**
PyTorch is moving toward `weights_only=True` as default for `torch.load()` for security. Current code uses permissive loading.

**Recommendation:**
1. Add safe globals registration
2. Switch to `weights_only=True`
3. Test model loading still works

---

### TD-005: env.py Observation Construction Complexity

**Evidence:** `echelon/env/env.py` - `_obs()` method and related helpers

**Impact:**
- **Technical:** Observation construction is complex (607 dims across 5 components); changes risk subtle bugs

**Effort:** M (2-3 hours)

**Category:** Code Quality

**Details:**
Observation space (607 dims) is constructed from:
- Contacts (110 dims)
- Pack Comm (80 dims)
- Local Map (121 dims)
- Telemetry (256 dims)
- Self Features (40 dims)

Logic is correct but spread across multiple methods. Could benefit from explicit ObservationBuilder class.

**Recommendation:** Extract to `echelon/env/observation.py`:
```python
class ObservationBuilder:
    def build_contacts(self, mech, sim) -> np.ndarray: ...
    def build_pack_comm(self, mech, pack) -> np.ndarray: ...
    def build_local_map(self, mech, world) -> np.ndarray: ...
    def build_telemetry(self, world) -> np.ndarray: ...
    def build_self_features(self, mech) -> np.ndarray: ...
    def build(self, mech, sim, pack, world) -> np.ndarray: ...
```

---

## Low Priority (Nice-to-Have)

### TD-006: Mypy Not in Strict Mode

**Evidence:** `pyproject.toml` mypy configuration

**Impact:**
- **Technical:** Some type errors may slip through; strict mode catches more bugs

**Effort:** L (8-12 hours incremental)

**Category:** Code Quality

**Details:**
Current mypy runs but doesn't use:
- `disallow_untyped_defs`
- `disallow_incomplete_defs`
- `check_untyped_defs`
- `no_implicit_optional`

**Recommendation:** Enable incrementally:
1. Add strict flags to `pyproject.toml`
2. Add per-module overrides for legacy code
3. Remove overrides as modules are touched

---

### TD-007: README Architecture Section Placeholder

**Evidence:** `README.md:62-63`
```markdown
## Architecture Overview

<!-- TODO: Pending architectural audit -->
```

**Impact:**
- **Business:** Users/contributors don't have architecture overview in README

**Effort:** S (30 minutes)

**Category:** Documentation

**Details:**
The archaeology analysis produced comprehensive architecture documentation in `docs/arch-analysis-2025-12-24-0627/`. README should link to it or include summary.

**Recommendation:** Either:
1. Add brief architecture summary to README
2. Link to `docs/arch-analysis-2025-12-24-0627/04-final-report.md`

---

## Dependency Map

```
TD-001 (train_ppo.py refactor)
    ↓
    └── Enables: curriculum learning, experiment templates, alternative algorithms

TD-002 (Arena tests)
    ↓
    └── Enables: confident self-play modifications, population-based training

TD-003 through TD-007: Independent, can be done in any order
```

---

## Summary

| ID | Item | Priority | Effort | Category |
|----|------|----------|--------|----------|
| TD-001 | Monolithic train_ppo.py | HIGH | L | Architecture |
| TD-002 | Arena unit tests missing | HIGH | M | Testing |
| TD-003 | Magic numbers in physics | MEDIUM | S | Code Quality |
| TD-004 | PyTorch weights_only TODO | MEDIUM | S | Code Quality |
| TD-005 | Observation construction complexity | MEDIUM | M | Code Quality |
| TD-006 | Mypy not strict | LOW | L | Code Quality |
| TD-007 | README architecture placeholder | LOW | S | Documentation |

**Total Effort:** ~20-30 hours for complete remediation

**Recommended Focus:** TD-001 and TD-002 (High priority items) provide the most value for ROADMAP progress.

---

*Generated by System Architect on 2025-12-24*
