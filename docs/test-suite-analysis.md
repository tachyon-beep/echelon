# Gap Analysis: Echelon DRL Test Suite

**Status:** Planning
**Goal:** Transform the current test suite into a world-class verification system for DRL/Simulation.

## 1. Executive Summary

The current Echelon test suite (`tests/`) provides a solid foundation of **unit tests** (`test_sim_mechanics.py`) and basic **property-based tests** (`test_sim_properties.py`). However, to reach a "world-class" standard for a Deep Reinforcement Learning (DRL) environment, it requires significant expansion in **determinism verification**, **performance benchmarking**, **regression testing**, and **simulation validity**.

This document outlines the gaps and proposes a roadmap to bridge them.

---

## 2. World-Class DRL Test Suite Requirements

Based on industry best practices (OpenAI Gym standards, RL reliability metrics), a robust DRL test suite must address the following layers:

### Layer 1: Mechanics & Physics (The "Game Engine")
*   **Current State:** Good. We verify movement, damage, and basic interactions.
*   **Gap:** 
    *   **Determinism:** Verify that `sim.step(action)` is bit-for-bit identical given the same seed and action sequence across different hardware/OS (if possible) or at least strictly reproducible on the same machine.
    *   **Conservation Laws:** Ensure mass/momentum/energy (or game equivalents like "total heat") behave consistently.
    *   **Edge Cases:** Physics behavior at boundaries (map edges, max velocity, zero stability).

### Layer 2: Environment API (The "Gym Interface")
*   **Current State:** Implicitly tested via usage.
*   **Gap:**
    *   **API Compliance:** Verify strict adherence to the environment contract (observation shape, action bounds, reward range).
    *   **Seeding:** Verify `env.reset(seed=X)` produces identical initial states.
    *   **Step Consistency:** Verify `env.step` returns valid types/shapes for *any* action (fuzzing).

### Layer 3: Learning & Regression (The "Intelligence")
*   **Current State:** None.
*   **Gap:**
    *   **Sanity Check (The "One-Step" Test):** Can an agent learn to maximize reward in a trivial, simplified version of the env (e.g., "Move to X")?
    *   **Regression Baselines:** Maintain a set of "Golden Policies". Ensure new code doesn't break their win rate against a fixed opponent.
    *   **Learning Curve:** Verify that a standard PPO implementation *improves* over random baseline within N steps.

### Layer 4: Performance & Scale
*   **Current State:** None.
*   **Gap:**
    *   **Throughput (SPS):** Track Steps Per Second. Fail if performance drops by >10%.
    *   **Memory Leaks:** Long-running episodes (10k steps) to ensure no RAM creep.
    *   **Vectorization:** Verify `AsyncVectorEnv` or equivalent wrappers work correctly.

---

## 3. Implementation Plan

### Phase 1: Hardening the Foundation (Immediate)
1.  **Determinism Test:** 
    *   Action: Run two envs side-by-side with same seed. Feed random actions. Assert `obs1 == obs2` for 1000 steps.
2.  **API Fuzzing:** 
    *   Action: Feed `NaN`, `Inf`, and `OOB` values to `step()`. Ensure Env sanitizes them or raises clean errors, never crashing the Sim.
3.  **Goldens:** 
    *   Action: Record a "Golden Replay" (actions + expected outcomes) for complex scenarios. Test against it to catch subtle physics regressions.

### Phase 2: Learning Verification (High Value)
1.  **Integration Test:** 
    *   Action: Train a tiny PPO agent on `EchelonEnv` for 100 steps in CI. Assert it doesn't crash.
2.  **Convergence Test (Trivial Task):** 
    *   Action: Create `EchelonMini-v0` (1 mech, no enemy, move to goal). Assert PPO solves it in <5 mins. This proves the *environment* provides learnable signals.

### Phase 3: Performance & DevOps (Long Term)
1.  **Benchmark Suite:** `scripts/benchmark.py` running standard scenarios.
2.  **Mutation Testing:** Fix the `mutmut` configuration to ensure tests catch logic changes (e.g., flipping `<` to `<=`).

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

1.  **Move existing tests** to `tests/unit/`.
2.  **Implement `test_determinism.py`:** Use `SeedSequence` to prove reproducibility.
3.  **Implement `test_learning.py`:** A minimal "sanity check" RL loop.
