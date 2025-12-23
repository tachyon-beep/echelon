# Gap Analysis: Echelon DRL Test Suite

**Status:** In Progress (Phase 1–2 largely implemented; Phase 3 partial; Phase 4 deferred until trained policies exist)
**Goal:** Transform the current test suite into a world-class verification system for DRL/Simulation.

## 1. Executive Summary

The current Echelon test suite (`tests/`) now includes unit coverage for core mechanics/determinism/fuzzing plus integration coverage for replay reproducibility and a minimal training-loop smoke test. Remaining work is primarily around long-run reliability (soak + wrappers), and **trained-policy regressions** (win-rate bands, learning-curve checks) once stable models exist.

This document outlines the gaps and proposes a roadmap to bridge them.

---

## 2. World-Class DRL Test Suite Requirements

Based on industry best practices (OpenAI Gym standards, RL reliability metrics), a robust DRL test suite must address the following layers:

### Layer 1: Mechanics & Physics (The "Game Engine")
*   **Current State:** Good. Unit tests cover key mechanics, projectiles/LOS, and several edge cases.
*   **Gap:** 
    *   **Determinism:** Verify that `sim.step(action)` is bit-for-bit identical given the same seed and action sequence across different hardware/OS (if possible) or at least strictly reproducible on the same machine.
    *   **Conservation Laws:** Ensure mass/momentum/energy (or game equivalents like "total heat") behave consistently.
    *   **Edge Cases:** Physics behavior at boundaries (map edges, max velocity, zero stability).

### Layer 2: Environment API (The "Gym Interface")
*   **Current State:** Direct fuzzing tests exist for NaN/Inf and OOB actions; basic determinism is covered.
*   **Gap:**
    *   **API Compliance:** Verify strict adherence to the environment contract (observation shape, action bounds, reward range).
    *   **Seeding:** Verify `env.reset(seed=X)` produces identical initial states.
    *   **Step Consistency:** Verify `env.step` returns valid types/shapes for *any* action (fuzzing).

### Layer 3: Learning & Regression (The "Intelligence")
*   **Current State:** Basic integration coverage exists (PPO loop smoke + trivial convergence), but no trained-policy regressions yet.
*   **Gap:**
    *   **Sanity Check (The "One-Step" Test):** Can an agent learn to maximize reward in a trivial, simplified version of the env (e.g., "Move to X")?
    *   **Regression Baselines:** Maintain a set of "Golden Policies" and win-rate checks **(once we have a trained model/policy snapshot to baseline)**.
    *   **Learning Curve:** Verify that a standard PPO implementation *improves* over a random baseline within N steps **(once we have a trained model/policy snapshot to baseline)**.

### Layer 4: Performance & Scale
*   **Current State:** Basic SPS and short-run memory stability checks exist, but are not a long soak.
*   **Gap:**
    *   **Throughput (SPS):** Track Steps Per Second. Fail if performance drops by >10%.
    *   **Memory Leaks:** Long-running episodes (10k steps) to ensure no RAM creep.
    *   **Vectorization:** Verify `AsyncVectorEnv` or equivalent wrappers work correctly.

---

## 3. Implementation Plan

### Phase 1: Hardening the Foundation (Immediate)
1.  **Determinism Test:** 
    *   Implemented (basic): `tests/unit/test_determinism.py` (candidate improvement: increase step count and expand asserted state).
2.  **API Fuzzing:** 
    *   Implemented: `tests/unit/test_api_fuzzing.py`.
3.  **Goldens:** 
    *   Implemented (replay rerun): `tests/integration/test_golden.py`.

### Phase 2: Learning Verification (High Value)
1.  **Integration Test:** 
    *   Implemented (smoke): `tests/integration/test_learning.py` (ensures env+model loop runs and backprop works).
2.  **Convergence Test (Trivial Task):** 
    *   Implemented (batch-optimization proxy): `tests/integration/test_convergence.py` (candidate improvement: assert a behavioral metric, not just loss).

### Phase 3: Performance & DevOps (Long Term)
1.  **Benchmark Suite:** Implemented as pytest perf tests: `tests/performance/test_benchmark.py` (candidate improvement: gate behind a marker/env var and add a `scripts/benchmark.py` runner for local profiling).
2.  **Mutation Testing:** Config exists (`pyproject.toml`), but needs regular execution/wiring (CI or a documented local workflow).

### Phase 4: Trained-Model Regression (Once We Have A Trained Model)
1.  **Golden Policy Baselines:** Add 1–3 stable policy snapshots (and opponent pools) and assert win-rate bands on fixed eval seeds.
2.  **Learning Curve Regression:** Track “improves vs random” within a small budget (N updates / N env-steps) to detect reward/obs/action regressions.
3.  **Policy Compatibility Checks:** Ensure old snapshots still load and can run inference after env/action/obs changes (or fail with a clear compatibility error).

---

## 4. Recommended Directory Structure

```text
tests/
├── unit/                   # Fast, localized tests (current test_sim_mechanics)
│   ├── test_mechanics.py
│   ├── test_obs_space.py
│   └── test_determinism.py
├── integration/            # Slower, full-loop tests
│   ├── test_learning.py    # Can PPO learn "Move To Goal"?
│   └── test_golden.py      # Replay verification
├── performance/
│   └── test_benchmark.py   # SPS checks
├── conftest.py
└── resources/              # Golden replays, config files
```

## 5. Next Actions for "World Class" Status

1.  **Harden determinism:** longer runs, more asserted state, and looser float tolerance where needed for portability.
2.  **Add API contract checks:** observation/action shapes, bounds, and return types across modes/configs.
3.  **Add long-run soak + wrappers:** 10k-step memory soak and vectorized-env compatibility.
4.  **(Once trained)** Add golden policy + win-rate regression tests.
