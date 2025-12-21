# Drone System Implementation Plan

## 1. Overview
The goal is to implement a "Drone Station" weapon system that allows mechs to deploy autonomous drones. These drones serve as force multipliers by extending sensor range ("spotting") and providing additional firepower or electronic warfare support.

## 2. Architecture Changes

### A. Configuration (`echelon/config.py`)
*   **New Enum**: `DroneVariant` (`SCOUT`, `COMBAT`).
*   **New Dataclass**: `DroneSpec`
    *   `hp`: Health points (e.g., 20.0).
    *   `speed`: Flight speed (e.g., 10.0 m/s).
    *   `lifetime`: Duration in seconds (e.g., 30s).
    *   `sensor_radius`: For spotting/EW (e.g., 30 voxels).
    *   `orbit_radius`: Distance to orbit target or owner.
    *   `weapon_spec`: (Optional) For combat drones.
*   **Weapon Definition**:
    *   Add `DRONE_STATION` to the weapon list.
    *   Set `DRONE_COOLDOWN` (e.g., 15s) and `DRONE_AMMO` (default 2).

### B. Mech State (`echelon/sim/mech.py`)
*   **New Fields**:
    *   `drone_cooldown`: Float, countdown timer.
    *   `drone_ammo`: Int, remaining charges.

### C. Drone Entity (`echelon/sim/drone.py`)
*   **New Entity Class**: `Drone`
    *   **State**: `pos`, `vel`, `owner_id`, `team`, `variant`, `hp`, `remaining_life`, `target_id`.
    *   **Logic**:
        *   `update(dt, sim)`: Handle movement, lifetime decay, and AI decision loop.
        *   `ai_scout()`: Orbit nearest enemy, apply "Painted" status, provide vision.
        *   `ai_combat()`: Orbit owner or enemy, fire micro-lasers.

### D. Simulation Engine (`echelon/sim/sim.py`)
*   **Storage**: Add `self.drones: list[Drone]` to the `Sim` class.
*   **Lifecycle**:
    *   `reset()`: Clear drone list.
    *   `step()`: Call `drone.update()` for all active drones; remove dead/expired ones.
*   **Interaction**:
    *   `spawn_drone(owner, variant)`: Factory method to create and register a drone.
    *   `drone_sight_check(team, target_pos)`: Helper to check if *any* active drone of a specific team has LOS to a position.

### E. Actions (`echelon/actions.py`)
*   **New Action**: `ActionIndex.FIRE_DRONE`.
    *   This will be a discrete trigger (value > 0).
    *   If `drone_ammo > 0` and `drone_cooldown <= 0`, a drone is launched.
    *   **Variant Logic**: To avoid action bloat, the drone variant is determined by the Mech Class (e.g., Scout/Light Mechs launch `SCOUT` drones, Medium/Heavy launch `COMBAT` drones).

### F. Environment & Observations (`echelon/env/env.py`)
*   **Action Handling**:
    *   Map the new action index to the simulation's `spawn_drone` logic.
*   **Observation Update (`_obs`)**:
    *   **Spotting**: Modify the contact discovery loop. A target is visible if:
        1.  It is within the mech's own radar/LOS (existing logic).
        2.  **OR** it is visible to any friendly `Drone` (new logic using `sim.drone_sight_check`).
    *   **Self Status**: Add `drone_ready` (normalized ammo + cooldown) to the self-observation vector.
    *   **Incoming**: (Optional) Add "drone_detected" warning to observations.

## 3. Implementation Steps

1.  **Define Configs**: Create the specs in `config.py`.
2.  **Create Entity**: Implement `Drone` class in `sim/drone.py`.
3.  **Update Sim**: Integrate drone list management and update loop in `sim.py`.
4.  **Update Mech**: Add state fields in `mech.py`.
5.  **Wire Actions**: Add `FIRE_DRONE` to `ActionIndex` and `EchelonEnv` step logic.
6.  **Wire Observations**: Update `_obs` to use drone vision for spotting enemies.
7.  **Viewer**: Update replay recording to include drone states.

## 4. Risks & Considerations
*   **Performance**: Raycasting for every drone every step could be expensive. 
    *   *Mitigation*: Run drone sensor checks at a lower frequency (e.g., every 10 ticks) or limit drone count.
*   **Balance**: "Map hack" (spotting everything) is very powerful. 
    *   *Mitigation*: Give drones limited sensor radius and make them destructible (low HP).
