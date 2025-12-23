# Drone System Implementation Plan

## 1. Overview
The goal is to implement a "Drone Station" weapon system that allows mechs to deploy autonomous drones. These drones serve as force multipliers by extending sensor range ("spotting") and providing additional firepower or electronic warfare support.

## 1.1 Terminology / Intent

- **Hull**: the mech configuration that determines what drone variants it carries (e.g. a Heavy hull carries `SPECTRE` + `RAZOR`).
- **Drone variant**: the behavior package/spec for a drone (`SPECTRE`, `RAZOR`, `AEGIS`, `SENTRY`).
- **Drone station**: the mech-mounted “launcher” ability that consumes ammo and cooldown to deploy one drone from the hull’s loadout.
- **Design goal**: add *new controllable complexity* (vision sharing, EWAR auras, intercept/fire-support tradeoffs) while remaining deterministic and performant.

## 1.2 Complexity & Risk Assessment

- **Complexity:** Medium–High. New entity type + update loop + observation integration + action/schema versioning.
- **Engineering risk:** Medium–High. Biggest risks are action/obs shape churn, determinism drift from AI/update ordering, and performance blowups if spotting/LOS is implemented naïvely.
- **Gameplay/balance risk:** High (intended). Drones can easily become “map hack” or unfun harassment without strict caps and counterplay.
- **Mitigations:** phase rollout (sensor-only → defense → offense → gun-drone), stable `drone_id` ordering + RNG stream split, “spotted set” computed once per team per tick, feature flags/schema versioning, and replay-visible drone state for debugging.

## 2. Architecture Changes

### A. Configuration (`echelon/config.py`)
*   **Final Drone Types**:
    *   `SPECTRE`: Scout variant. High speed, large sensor radius, ECCM aura.
    *   `RAZOR`: Combat variant. Equipped with Micro-Missiles (stability/chip damage).
    *   `AEGIS`: Support variant. Orbits owner closely, provides secondary AMS (Anti-Missile System).
    *   `SENTRY`: Gun-drone variant. Flies/loiters like a normal drone, but is “plain”: a regular kinetic gun (no micro-missiles/AMS/EWAR).
*   **Weapon Definition**:
    *   Add `DRONE_STATION` to the weapon list.
    *   Set `DRONE_COOLDOWN` (e.g., 15s) and `DRONE_AMMO` (default 2).

*   **Loadouts (per hull)**:
    *   Add a per-hull `drone_loadout` (ordered list of drone variants) and optional per-hull ammo/cooldown overrides.
    *   Example loadouts:
        *   `heavy`: `[SPECTRE, RAZOR]` (first launch scouts, second launch strikes)
        *   `scout`/`light`: `[SPECTRE]`
        *   `medium` (support hull): `[AEGIS]`
        *   `medium` (gun-drone hull): `[SENTRY]`
    *   Note: the current codebase has only one `"medium"` class; if we want `AEGIS` vs `SENTRY` to be distinct hulls, we need either:
        *   two medium classes (e.g. `medium_warden`, `medium_sentinel`) in the roster, OR
        *   a separate “hull variant” field on mechs that selects the loadout.

### B. Mech State (`echelon/sim/mech.py`)
*   **New Fields**:
    *   `drone_cooldown`: Float, countdown timer.
    *   `drone_ammo`: Int, remaining charges.
    *   `drone_loadout_index`: Int (index into the hull’s `drone_loadout`, used to cycle variants deterministically).

### C. Drone Entity (`echelon/sim/drone.py`)
*   **New Entity Class**: `Drone`
    *   **State**: `pos`, `vel`, `owner_id`, `team`, `drone_type`, `hp`, `remaining_life`, `target_id`.
    *   **Behaviors**:
        *   `SPECTRE`: High-altitude orbit, shares vision, provides ECCM.
        *   `RAZOR`: Aggressive seek-and-strike (Micro-Missiles).
        *   `AEGIS`: Close-proximity orbit, intercept incoming missiles.
        *   `SENTRY`: Loiter near the owner (or a commanded point) and fire a standard kinetic weapon at valid targets.

### D. Simulation Engine (`echelon/sim/sim.py`)
*   **Storage**: Add `self.drones: list[Drone]` to the `Sim` class.
*   **Lifecycle**:
    *   `reset()`: Clear drone list.
    *   `step()`: Call `drone.update()` for all active drones; remove dead/expired ones.
*   **Interaction**:
    *   `spawn_drone(owner, drone_type)`: Factory method to create and register a drone.
    *   `drone_visibility(team, target_id|target_pos)`: Returns whether any friendly drone makes a target visible.
    *   `drone_ewar_levels(viewer)`: Optionally contributes `eccm_level` (SPECTRE) and other effects to sensor quality.

*   **Determinism requirements**:
    *   Drones must use a deterministic RNG stream (seed-split from the sim RNG).
    *   Drone iteration order must be stable (e.g., assign monotonically increasing `drone_id` and iterate sorted by id).

### E. Actions (`echelon/actions.py`)
*   **New Action**: `ActionIndex.FIRE_DRONE`.
    *   This will be a discrete trigger (value > 0).
    *   If `drone_ammo > 0` and `drone_cooldown <= 0`, a drone is launched.
    *   **Variant Logic**:
        *   Launches the next drone in the hull’s `drone_loadout` and increments `drone_loadout_index` (wraps around if desired).

*   **Action-space impact**:
    *   Adding this action increases `ACTION_DIM` and therefore changes policy checkpoints and training code. Plan should stage this behind a feature flag and/or bump an explicit model/action schema version.

### F. Environment & Observations (`echelon/env/env.py`)
*   **Action Handling**:
    *   Map the new action index to the simulation's `spawn_drone` logic.
*   **Observation Update (`_obs`)**:
    *   **Spotting**: Modify the contact discovery loop. A target is visible if:
        1.  It is within the mech's own radar/LOS (existing logic).
        2.  **OR** it is visible to any friendly `Drone` (new logic using `sim.drone_sight_check`).
    *   **Self Status**: Add `drone_ready` (normalized ammo + cooldown) to the self-observation vector.
    *   **Incoming**: (Optional) Add "drone_detected" warning to observations.

*   **Visibility semantics (must be explicit)**:
    *   Decide whether drone spotting is **team-wide**, **pack-scoped**, or **owner-scoped**.
    *   Decide whether drone spotting bypasses jamming/sensor quality, or feeds into sensor quality (recommended) to avoid “map hack”.

*   **Performance design**:
    *   Avoid per-(viewer,target) drone LOS checks inside `_obs`.
    *   Prefer computing a per-team “spotted set” once per `_obs()` (bounded by drones × agents) and then just checking membership while building contact lists.

## 3. Implementation Steps

### Recommended Phasing

1.  **Phase 1 (Sensor-only)**: Implement `SPECTRE` as vision-sharing + (optional) ECCM aura. No drone weapons yet.
2.  **Phase 2 (Defense)**: Implement `AEGIS` as a secondary AMS (hook into missile update/intercept logic).
3.  **Phase 3 (Offense)**: Implement `RAZOR` micro-missiles (chip + stability) with strict caps and clear counterplay.
4.  **Phase 4 (Gun drone)**: Implement `SENTRY` as a “plain” gun-drone (intentionally not a static emplacement; should be comparatively straightforward).

### Concrete Steps

1.  **Define config + loadouts**: Introduce `DroneType` + `DroneSpec` and per-hull `drone_loadout`.
2.  **Create entity**: Implement `Drone` with deterministic ids and RNG stream usage.
3.  **Update sim**: Add storage + update loop + helpers (`drone_visibility`, optional `drone_ewar_levels`).
4.  **Update mech state**: Add cooldown/ammo/loadout index; ensure it resets deterministically.
5.  **Wire action**: Add `FIRE_DRONE` and integrate into env step/sim step.
6.  **Wire observations**: Implement “spotted set” computation once per `_obs()` and apply chosen visibility semantics.
7.  **Replay/viewer**: Include drone states (and optionally spot/ewar debug info) so behavior is inspectable.
8.  **Tests**: Add determinism + visibility-scope tests + perf guardrails.

## 4. Risks & Considerations
*   **Performance**: Raycasting for every drone every step could be expensive. 
    *   *Mitigation*: Run drone sensor checks at a lower frequency (e.g., every 10 ticks) or limit drone count.
*   **Balance**: "Map hack" (spotting everything) is very powerful. 
    *   *Mitigation*: Give drones limited sensor radius and make them destructible (low HP).

*   **Determinism**: Drone AI and update ordering can easily introduce replay drift.
    *   *Mitigation*: stable drone ids/order; dedicated RNG stream; avoid dict iteration ordering pitfalls.

*   **Action-schema churn**: Adding a new action dimension breaks old checkpoints and some tests.
    *   *Mitigation*: feature flag + schema versioning; keep compatibility path for legacy models.

## 5. Test Plan (Minimum)

- **Visibility scope**: assert the chosen spotting scope (team/pack/owner) works and doesn’t leak.
- **EWAR interaction**: if SPECTRE provides ECCM, assert `sensor_quality` changes as expected under ECM.
- **Determinism**: same seed + same actions ⇒ identical drone spawns/states over N steps.
- **Perf**: `_obs()` cost with max drones stays within a budget on default map sizes.
