# Echelon Analysis: 10v10 DRL Demo & Future Roadmap

This document serves as a synthesis of technical analysis and the established project roadmap. It categorizes improvements based on implementation weight and provides commentary on their alignment with the long-term vision of hierarchical command and modular systems.

---

## Current Demo Snapshot (What We Have Now)

### Core loop
- **10v10 via packs**: `PACK_SIZE=10` (2 heavy, 5 medium, 3 light) per team; `num_packs=1` yields 10v10.
- **Multi-agent env**: PettingZoo-parallel-like API (`reset/step`) with `"full"` and `"partial"` observation modes.
- **Real-time sim**: fixed `dt_sim` micro-steps; actions held for `decision_repeat` substeps (SMDP-ish cadence).
- **Objective**: king-of-the-hill capture circle with team zone scoring + elimination/time-up fallbacks.

### Actions & observations
- **Action layout**: base 9-d control + target selection + ECM/ECCM toggles + observation-controls + optional pack comm tail.
- **Observation layout**: fixed top‑K contact table (5 slots), pack-local comm board, ego-centric local occupancy map, and self/objective scalars.
- **Partial observability**: radar-range visibility + LOS checks; pack-scoped paint provides indirect visibility/locks (now EWAR-gated).

### Mechs & mechanics
- **Classes**: light/medium/heavy with distinct size/speed/yaw/heat caps.
- **Heat**: weapon heat + venting; overheating triggers shutdown (no control, physics continues).
- **Stability**: stability damage can cause knockdown/stun; regen over time; legged reduces speed/accel and stability max/regen.
- **Leg damage**: `leg_hp` with an `is_legged` state.
- **Death/debris**: dead mechs stamp voxel debris into the world as new cover.

### Weapons (current)
- **Laser**: hitscan, arc + LOS gated; rear-crit multiplier; chance of leg hit; now transfers heat to target.
- **Missiles**: homing projectile with splash; lock via LOS or pack-scoped paint (paint lock now affected by EWAR).
- **Kinetics**: heavy gauss (ballistic drop) and medium autocannon (linear); swept collision to prevent tunneling.
- **Painter**: light-only target paint enabling indirect pack lock + bonus damage.
- **AMS**: medium/heavy point-defense chance to intercept incoming homing missiles.

### Terrain / voxels
- **World**: boolean `solid[z,y,x]` voxel grid; archetype generator (`citadel`, `urban_grid`, `highway`) + scatter.
- **Variants**: map reorientation transforms + randomized spawn corners.
- **Fightability**: spawn clears + capture-zone clearing + macro corridor carving + connectivity validator “dig A*” safety net.

### Training & ops
- **Trainer**: PPO + LSTM (`scripts/train_ppo.py`) vs heuristic or arena league opponents.
- **Arena**: Glicko‑2 league for self-play population management (`echelon/arena/*`).
- **Metrics**: `metrics.jsonl` now includes per-episode outcome stats (kills/assists/damage/knockdowns/etc.).

### Visualization
- **Replay recording**: env can serialize world + frames + events.
- **Viewer**: `viewer.html` renders voxels/mechs/objective and weapon FX; FastAPI server for listing/loading/pushing replays.

---

## Recently Added (Complexity & Risk)

> “Risk” here is **engineering/regression risk** against the current codebase (not gameplay balance risk alone).

| Addition | Primary touchpoints | Complexity | Risk | Notes |
|---|---|---:|---:|---|
| Contact-slot target selection (focus fire) | `echelon/env/env.py`, `echelon/sim/sim.py` | Medium | Medium | Adds new action dims and a dependency on last-step contact-slot identity; improves coordination but breaks old checkpoints. |
| ECM/ECCM toggles + sensor quality | `echelon/env/env.py`, `echelon/sim/sim.py`, `echelon/sim/mech.py` | Medium | Medium | Affects radar visibility + painted lock; adds continuous heat costs; balance-sensitive but localized. |
| Ego-centric local occupancy map | `echelon/env/env.py` | Medium | Low–Med | Bigger obs vector + extra per-step compute; simple footprint slice keeps cost contained. |
| Missile incoming-warning scalar | `echelon/env/env.py` | Low | Low | Purely observational; low regression risk. |
| Laser heat transfer | `echelon/sim/sim.py` | Low | Low–Med | Mechanically small; balance impact is the primary unknown. |
| Autocannon suppression (stability regen debuff) | `echelon/sim/sim.py`, `echelon/sim/mech.py` | Low–Med | Med | Interacts with knockdown/stability; watch for stun-lock edges. |
| AMS missile intercept | `echelon/sim/sim.py`, `echelon/sim/mech.py` | Medium | Med | Touches projectile update loop; stochastic defense can affect fairness/learnability. |
| Macro corridor carving + spawn jitter | `echelon/gen/corridors.py`, `echelon/env/env.py`, terrain scripts | Medium | Low–Med | Changes terrain distribution + hashes; should remain safe with validator as backstop. |
| Richer replay events + mech state | `echelon/env/env.py`, `echelon/sim/sim.py` | Low–Med | Low | Mostly additive fields; viewer updated to consume them. |
| Episode stats logged to training metrics | `echelon/env/env.py`, `scripts/train_ppo.py` | Low | Low | Additive logging; minimal behavioral impact. |

**Compatibility note:** action/obs dimensions changed. Existing `.pt` checkpoints trained on the old shapes will not load; retraining is expected.

## 5 Light-Duty Ideas
*Focus: Refining immediate mechanics without introducing "dead-end" logic.*

1. **Electronic Warfare (EW) Basics**
    * **Description:** Integrate `ecm_on` and `eccm_on` flags to modify radar detection radius.
    * **Status:** Implemented (light-only ECM/ECCM with sensor-quality effects and heat cost).
    * **Roadmap Alignment:** High. This provides the foundational sensor logic that "Commander" agents will later exploit for stealth maneuvers.

2. **Leg Damage & Physics Penalties**
    * **Description:** Add rotation penalties or stumbling chances when `is_legged`.
    * **Status:** Partially implemented (`leg_hp` + movement penalties via `is_legged`; no rotation/stumble yet).
    * **Roadmap Alignment:** High. Enhances the value of tactical targeting, making "crippling" an enemy a viable strategy for the upcoming hierarchical commander.

3. **Observation Augmentation (Self-State)**
    * **Description:** Explicitly add Stability and Leg HP to the agent's self-observation.
    * **Status:** Implemented and expanded (adds self stability/heat caps, suppression/AMS, EWAR status, missile warning, and a local occupancy map).
    * **Roadmap Alignment:** High. Essential for "Generalized Policies." If an agent is to control any modular mech, it must have a clear understanding of its own current physical integrity.

4. **Generalized Tactical Adaptation (Refinement of Action Masking)**
    * **Original Idea:** Masking irrelevant weapon actions for current classes.
    * **Developer Commentary:** Rejected. Action masking is a short-term crutch that hinders the goal of a universal policy.
    * **Refined Goal:** The agent should remain "class-agnostic," learning to probe its capabilities (e.g., trying to fire a gauss on a light mech should simply yield no result/heat, teaching the agent what it *is* through interaction).

5. **Emergent Suppressive Fire**
    * **Original Idea:** Explicit reward for stability damage.
    * **Developer Commentary:** Risky (MVRP violation). 
    * **Status:** Implemented as a mechanic (autocannon suppression reduces stability regen) without adding reward shaping.
    * **Refined Goal:** Avoid explicit rewards. Trust that as the simulation moves to "Commander vs. Commander" matches, suppressive behavior will emerge naturally as a dominant tactic to pinning enemies in cover.

---

## 5 Medium-Duty Ideas
*Focus: Tactical depth and dynamic environments.*

1. **ECM/ECCM Aura (The "Bubble")**
    * **Description:** Area-of-effect radar concealment provided by specific units.
    * **Status:** Implemented (radius-based jam/eccm → sensor quality; impacts radar/paint lock).
    * **Roadmap Alignment:** High. Directly supports the "Spotter" and "Scout" roles that Commanders will need to build.

2. **Destructible Voxel Fragments**
    * **Description:** Weapons like Gauss/Missiles can remove cover.
    * **Status:** Not implemented (death debris adds cover, but weapons do not carve/remove voxels yet).
    * **Roadmap Alignment:** Essential. In a voxel-based game, static terrain is a missed opportunity. Destruction forces the DRL agent to constantly re-evaluate the "safety" of a position.

3. **Weapon Hardpoint Variants (The Mid-Step)**
    * **Description:** Spawn mechs with varied loadouts (Brawler vs. Sniper).
    * **Status:** Not implemented.
    * **Roadmap Alignment:** This is the bridge to the Modular Assembly System. It forces the current policy to handle variance in "Pack Composition," shifting the focus from "more guns" to "tandem tactical solving."

4. **Verticality & Jump Jet Mastery**
    * **Description:** Layered heightmaps and platforms in the terrain generator.
    * **Status:** Not implemented (terrain is boolean solids; no multi-level pathing beyond basic Z clearance).
    * **Roadmap Alignment:** High. True tactical mastery requires 3D spatial awareness, preventing the DRL from collapsing into 2D "blob" tactics.

5. **Unit Specialization via Design (Refinement of Broadcast Roles)**
    * **Original Idea:** Static "Spotter" roles.
    * **Developer Commentary:** Overtaken by the Hierarchical Commander. 
    * **Refined Goal:** Instead of hard-coding roles, focus on the *capability* (e.g., enhanced sensors part). Commanders will then build "Scout Units" using the modular system.

---

## 5 Heavy-Duty Ideas
*Focus: Systemic architecture and the "Final Form" of Echelon.*

1. **The 3-Layer Command Hierarchy**
    * **Structure:** 
        * **Squad Leader:** Direct control of a 10-mech Pack.
        * **Platoon Leader:** Orchestrates 3 Squad Leaders.
        * **Company Commander:** Strategizes across 3 Platoon Leaders.
    * **Impact:** Solves the "Strategic Horizon" problem. Lower-level agents focus on micro-tactics (LOS, movement), while higher-level agents focus on the macro-war (flanking, resource allocation).

2. **Voxel-Based Acoustic Sensing**
    * **Description:** Simulate vibrations/sound through the voxel grid.
    * **Impact:** A unique "Echelon" feature that provides a non-visual sensor layer, critical for the "Information War" aspect of the simulation.

3. **Full-Spectrum Electronic Warfare (EW)**
    * **Description:** Jammers, decoys, and sensor spoofing.
    * **Impact:** Moves the DRL challenge from "perfect information" to "adversarial information," where agents must deduce the true state of the battlefield.

4. **Modular Mech Assembly System (Post-Hierarchy)**
    * **Description:** Budget-based assembly of mechs (Chassis, Reactor, Weapons).
    * **Roadmap Alignment:** This follows the Hierarchical Agent. Once the command structure is stable, the simulation becomes a test of "Strategic Engineering"—designing the right tools for the Commander to use.

5. **Multi-Objective Mission Dynamics**
    * **Description:** Escort, Search & Destroy, Extraction.
    * **Impact:** Prevents the policy from over-fitting to the "King of the Hill" circle, creating a truly robust tactical engine capable of handling any combat scenario.
