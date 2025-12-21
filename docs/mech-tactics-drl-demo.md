# Mech Tactics DRL Demo (WIP Notes)

Goal: a lightweight, long-running DRL “tactics” demo with thin visualization that people can check in on over days/weeks (including self-play leagues and “named” policies).

## Game concept

- Simple voxel arena for cover/terrain + objectives, but with **continuous real-time** movement (not 2D tile-step / turn-based).
- Pack vs pack (10 mechs per pack; start smaller in curriculum; scale up later).
- Fixed-timestep real-time simulation; no human in the loop.
- Intended to produce legible tactical behavior (take/hold cover, focus fire, flank, area denial).

### Deliberate departures (from the old env)

- Move away from a 2D grid + turn-based feel toward continuous positions/orientation and fixed-timestep real time.
- Keep voxelization as an implementation detail for collision/cover/LOS queries (not as the primary “move cell-to-cell” gameplay).

### Scale assumption (current)

- Treat `1 voxel` as roughly a `5m × 5m × 5m` cell (“chunky” but readable).
- Buildings/cover blocks can be 1 voxel (or stacked for height).
- Smaller vehicles can still occupy 1 voxel for simplicity; “smallness” shows up in stats (speed/signature/etc.), not collision footprint.
- With this scale, arena sizes map roughly to:
  - `10×10×10` ≈ `50m × 50m × 50m` (very brawly / debug box)
  - `20×20×20` ≈ `100m × 100m × 100m` (early prototype)
  - `50×50×20` ≈ `250m × 250m × 100m` (more “tactical” spacing)
  - (Rule of thumb: expand XY first; keep Z modest.)

### Mech collision volumes (current assumption)

- Light: `1×1×1` voxels.
- Medium: `1×1×1` voxels (stats differentiate; can revisit footprint later).
- Heavy: `2×2×2` voxels (physically larger; matters for cover, clearance, and choke points).

### Performance philosophy

- If/when we outgrow naive implementations, prefer **architectural scaling** (chunking, interest management, culling, batched queries) over micro-optimizing hot loops prematurely.

## Mechs & mechanics (target set)

Mech size/class impacts: turn speed, move speed, armor, weapon loadout, ability loadout.

- Heavy: missile launchers (indirect fire), lasers (hitscan energy), gauss (single-shot heavy kinetic).
- Medium: lasers, gatlings/autocannon (kinetic).
- Light: laser, target painter, jump jets, ECM/ECCM.

Shared mechanics:

- Simple heat model; overheat disables/turns off (“shutdown”).
- Very simple positional damage (directional armor / hit location).
- Sensors including “am I being targeted” / incoming-fire awareness.

## Simulation & decision timing

Recommended structure: keep simulation fidelity high while keeping learning sequences manageable.

- Run the world at a small simulation tick `dt_sim` (movement, projectiles, heat, cooldowns).
- Only choose actions every `dt_act = N * dt_sim` (“decision tick”).
- Hold the last chosen action for `N` micro-steps (action repeat).
- At each decision tick, all agents act simultaneously (“we go”); then the world advances `N` micro-steps.

### Adaptive decision frequency (“slow/med/fast mode”)

Feasible and potentially useful; implement as **variable-duration steps (SMDP)**:

- Keep `dt_sim` fixed; adapt `action_repeat` / `N` based on **agent-observable** signals (e.g., enemy detected, being targeted, objective contested).
- Add hysteresis / minimum time-in-mode to prevent threshold thrashing.
- Cooldowns/heat/projectiles always advance on **sim time**, not decision ticks (so more decisions never means more ability usage per second).
- Make learning time-consistent:
  - Accumulate reward over the `N` micro-steps.
  - Discount by elapsed time (e.g., `gamma^N` for fixed `dt_sim`, or `exp(-k * dt)`).
- Include `dt`/mode (and optionally `time_since_last_decision`) in observations so the recurrent core doesn’t have to infer step duration.

## RL approach (sketch)

- PPO with an LSTM (or similar recurrent core) to handle partial observability and non-determinism.
- Keep an LSTM hidden state **per mech**; reset on episode end; mask properly on `done`.
- Train with sequence minibatches (truncated BPTT); optionally feed `prev_action` (and sometimes `prev_reward`) into the recurrent core.

## Observation model (current)

The env emits a fixed-size vector per mech consisting of:

- **Top-K contact table**: up to 5 visible contacts, each with rel position/velocity, yaw, hp/heat/stability, class, relation (friendly/hostile/neutral), paint flag, and a `visible` bit.
  - Slot quotas (with repurposing): 3× friendly, 1× hostile, 1× neutral; unused slots are filled from other categories (hostiles first).
- **Pack comm board**: last messages from packmates only (`PACK_SIZE * comm_dim` floats, 1 decision-tick delayed).
- **Self/objective scalars**: threat + status + objective UI (weapon cooldowns, self velocity, zone vector/radius, zone control, zone score progress, time fraction).
  - Includes the current contact selector settings: sort mode one-hot (closest/biggest/most_damaged) + hostile-only filter flag.

Visibility rules in `observation_mode="partial"`:

- There is no pack/team “telemetry omniscience”: other mechs are visible only if **(LOS)** or within a short **radar range**; opponents can also become visible if **painted by your pack**.
- Otherwise their dynamic fields are zeroed and `visible=0` (and team/class are treated as unknown until visible).

### Self-play league / “nicknamed” policies

- Train a learner vs opponents sampled from an opponent pool; periodically promote snapshots into a hall-of-fame.
- Check-ins: periodic evaluation matches against fixed baselines + hall-of-famers on fixed seeds; save short replays and summary metrics.
- Optional: auto-label “styles” from metrics (e.g., Flanker, Turtle, Brawler, Sniper, Heat Miser/Spiker, Caller).

## Curriculum ladder (agreed direction)

Start simple to get early, legible competence; add complexity one mechanic at a time:

1. Kinematic movement + cover/LoS + 1–2 weapon types; fixed loadouts; small teams (1v1/2v2).
2. Add heat (resource management).
3. Add indirect missiles (area denial).
4. Add “being targeted” sensor (evasion / peeking).
5. Add ECM/ECCM + painter (team coordination).
6. Scale to 5v5 and richer loadouts.

## Open questions

- Observation model: per-mech partial observability only, or team-shared vision/targets?
- Action space: continuous control vs discrete “tactical” actions (move-to voxel / face / fire / ability).
- Target `dt_sim` and `dt_act` (and desired max action-repeat range for slow/med/fast modes).
- Objective rules (capture point? payload? hold zones?) and episode termination.

---

## Roadmap: The "Triad of Ballistics" & Physics

**Goal:** Deepen the tactical sandbox by introducing distinct weapon roles and a physics-based control layer (Stability).

### 1. New Weapon Mechanics
Create a distinct "Rock-Paper-Scissors" of damage types:

*   **Energy (Lasers):**
    *   **Role:** Precision & Control.
    *   **Effect:** Instant hitscan damage + **Heat Transfer** to target.
    *   **Tactical Use:** Hunting fast lights; overheating enemies to force shutdowns.
*   **Missiles (LRMs):**
    *   **Role:** Area Denial & Punishment.
    *   **Effect:** Indirect fire (Arcing); requires Lock (LOS or Paint). Splash damage.
    *   **Counter:** **AMS (Anti-Missile System)** point-defense turrets that shoot down projectiles.
*   **Kinetic (Ballistics):**
    *   **Role:** Impact & Suppression.
    *   **Gauss Rifle:** Heavy sniper/mortar. High velocity, gravity drop. Can be fired indirectly (lobbed) without lock using pure geometry.
    *   **Autocannon:** Rapid fire. Suppresses enemies and prevents stability regen.
    *   **Effect:** Deals **Stability Damage** (Impulse).

### 2. The Stability System ("The Gyro")
A second resource bar alongside Heat.

*   **Stability Meter:**
    *   Regenerates over time (slower when moving).
    *   Depleted by taking **Kinetic Damage** (Gauss = massive spike, Autocannon = constant suppression).
    *   Depleted by **Jump Jet Landings** (hard landings shock the gyro).
*   **Knockdown:**
    *   If Stability reaches 0, the mech falls (`FALLEN` state).
    *   **Effect:** Immobilized and unable to fire for ~3.0s. Vulnerable to called shots.
    *   **Airborne Knockdown:** Losing stability mid-jump results in a crash landing (bonus damage).

### 3. Leg Integrity & Mobility Kill
Separate health pool for legs (or routed splash damage).

*   **"Legged" State:**
    *   Max speed reduced by ~60% (Limping).
    *   Max Stability & Regen reduced by ~50%.
    *   **Tactical Result:** A legged mech becomes a "turret"—easy to knock over and unable to flank.

### 4. Implementation Priorities
1.  **Kinetic Projectiles:** Implement gravity-affected projectiles (Gauss) and rapid-fire projectiles (Autocannon).
2.  **Stability Logic:** Add stability attribute, regen logic, and `FALLEN` state handling.
3.  **AMS:** Implement point-defense logic to shoot down incoming `Projectile` entities.
4.  **Indirect Fire (Manual):** AI logic to "lob" Gauss rounds over walls without a lock.
