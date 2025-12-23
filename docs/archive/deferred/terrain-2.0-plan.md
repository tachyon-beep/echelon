# Voxel Overhaul: The Living Grid

**Status:** Draft / Planning
**Target:** Future Roadmap (Major Architectural Upgrade)

## 1. Vision & Core Philosophy

The current Echelon terrain generation is effectively "2.5D" — while technically voxel-based, the generation logic primarily functions as a heightmap (pillars of solid ground). **Terrain 2.0** aims to unlock **True 3D** gameplay, but the **Voxel Overhaul** goes further: treating the grid not just as geometry, but as a simulation medium.

**The Goal:** A battlefield where the environment is an active participant. Explosions should ripple through tunnels, bridges should collapse under load, and the very air can become a weapon.

### Key Gameplay Shifts
*   **Verticality as a Weapon:** Knocking an enemy off a bridge is a valid tactic.
*   **Cover is Temporary:** Structural integrity means heavy weapons can reshape the map.
*   **The Grid Transmits Information:** Shockwaves, heat, and sound propagate through the voxel medium.

---

## 2. Technical Architecture: Breaking the Heightmap

### 2.1. From `set_box` to `Structure Modules`
Current generation relies on placing solid rectangular prisms. 2.0 will introduce a library of **Structural Voxels**:
*   **The Slab:** A horizontal platform with air above *and* below.
*   **The Arch/Tunnel:** Solid ceiling, air middle, solid floor.
*   **The Overhang:** A protrusion from a vertical face.

### 2.2. Connectivity Graph 2.0
The `ConnectivityValidator` (A*) must be upgraded to support 3D pathfinding.
*   **Problem:** A simple 2D nav-mesh no longer works if `(x,y)` can have multiple valid `z` positions (floor 1, floor 2).
*   **Solution:** The connectivity graph becomes a **Voxel Adjacency Graph**.
    *   Nodes are "Traversable Air Voxels" (adjacent to a surface).
    *   Edges connect adjacent air voxels.
    *   Special Edges: "Jump" connections (based on mech jump capability) connecting non-adjacent nodes.

---

## 3. Biome Concepts: Theaters of Future War

We will move beyond generic "Urban" or "Desert" to specific, narrative-rich combat environments that force distinct tactical adaptations.

### 3.1. The Arcology Spire (Vertical Urban)
*   **Theme:** Dense, brutalist mega-structures. A city built upwards, not outwards.
*   **Structure:**
    *   **The Void:** The map floor is lethal (fall = death/reset).
    *   **The Stacks:** multiple "floor" layers (e.g., z=10, z=20, z=30) connected by ramps and gravity lifts.
    *   **Skyways:** Narrow bridges connecting separate towers.
*   **Tactical Hook:** **Knockbacks**. Weapons with high impulse (Railguns, Missiles) become deadly not just for damage, but for map control. "Ring out" kills become possible.

### 3.2. The Orbital Refinery (Industrial/Zero-G)
*   **Theme:** A tangled mess of pipes, spherical tanks, and lattice-work scaffolding in space or low-g.
*   **Structure:**
    *   **Spiderweb Layout:** Non-linear connectivity. No clear "lanes," just a web of catwalks.
    *   **Micro-Gravity:** `gravity_multiplier = 0.3`. Jump jets are extremely potent; ballistic arcs are flat.
    *   **Venting Hazards:** Periodic jets of hot steam/plasma from pipe sections (timed environmental hazards).
*   **Tactical Hook:** **3D Flanking**. Attacks can come from literally any angle. "Up" is just another direction.

### 3.3. The Geo-Front (Subterranean)
*   **Theme:** A massive underground cavern, partially natural, partially reinforced.
*   **Structure:**
    *   **The Ceiling:** A solid voxel roof at max-Z.
    *   **Stalactites/Pillars:** Massive columns blocking LOS.
    *   **Lava Tubes:** Enclosed tunnels (fully roofed) that crisscross the map.
*   **Tactical Hook:** **Tunnel Fighting**. Indirect fire (Mortars/Missiles) is useless inside tubes. Combat forces close-range brawls. Heavy mechs dominate the tunnels; Lights dominate the open caverns.

### 3.4. The Glass Desert (Crystalline)
*   **Theme:** A blasted landscape of obsidian and crystal shards.
*   **Structure:**
    *   **Reflective Surfaces:** Specific voxel types that "bounce" laser shots (or scatter them).
    *   **Brittle Cover:** "Glass" walls that shatter (turn to air) after absorbing X damage.
    *   **Razor Terrain:** `Floor` voxels that deal minor damage to leg structure if walked on (requires Jump Jets to traverse safely).
*   **Tactical Hook:** **Attrition & Mobility**. You cannot stay still. Cover is temporary.

### 3.5. The Proving Grounds (Artificial)
*   **Theme:** A simulation-within-a-simulation. Clean white blocks, orange hazard lines, non-euclidean geometry.
*   **Structure:**
    *   **Dynamic Walls:** Walls that toggle on/off on a timer (phase shift).
    *   **Teleporter Nodes:** Instant travel between point A and B.
*   **Tactical Hook:** **Timing**. Agents must learn the rhythm of the map.

---

## 4. New Interactive Terrain Features

To support these biomes, the voxel engine needs "Special Block" support.

| Feature | Description | Interaction |
| :--- | :--- | :--- |
| **Destructible Structural** | A bridge tile or support pillar. | If HP -> 0, the voxel disappears. *Advanced:* A connectivity check runs; if a "floating island" is created, it falls/destroys. |
| **Vent / Jump Pad** | A floor tile with high vertical impulse. | Stepping on it launches the mech to Z+20. |
| **Phase Wall** | A wall that is SOLID for 10s, then AIR for 10s. | Provides rhythmic cover. Can trap greedy players. |
| **Hazard Voxel** | Magma, Acid, Radiation. | Apply DoT (Damage over Time) and Heat. |
| **Hardened Hardpoint** | A voxel that repairs adjacent mechs? | *Risk:* Encourages camping. Maybe better as a "Resupply" zone. |

---

## 5. Implementation Roadmap

### Phase 1: The "Overhang" Update
*   **Goal:** Modify `VoxelWorld.generate` to create a map with at least one bridge or tunnel.
*   **Task:** Create a `Tunneler` agent in the generator that "digs" horizontally through existing solids.
*   **Validation:** Ensure the camera and physics engine don't freak out when a unit is "under" something.

### Phase 2: The Physics Update
*   **Goal:** Implement Gravity Zones and Pushback.
*   **Task:** Add `impulse` vector to weapon hits. Add `global_gravity` to `WorldConfig`.
*   **Validation:** "Sumo" matches (try to push enemy off a platform).

### Phase 3: The Biome Pack
*   **Goal:** Implement the "Arcology" and "Geo-Front" archetypes.
*   **Task:** Write specific generator logic for these (not just generic noise).
*   **Validation:** Train agents on these maps. Do they learn to use the tunnels? Do they learn to knock enemies off ledges?

### Phase 4: Dynamic Voxels
*   **Goal:** Destructibility and Phase Walls.
*   **Task:** Update `Sim` loop to handle voxel state changes (HP, timers).
*   **Validation:** Performance check (re-baking nav/collision when world changes).

---

## 6. Voxel Physics Extensions (The "Overhaul" Layer)

By treating the grid as a simulation medium, we unlock "Systemic Gameplay" (where systems interact unexpectedly).

### 6.1. Concussion & Shockwave Propagation
*   **Concept:** Explosions in enclosed spaces are deadlier.
*   **Mechanism:** When an explosion occurs, trace rays *through the voxel air*.
    *   **Open Field:** Energy dissipates via inverse-square law.
    *   **Tunnel/Room:** Energy reflects off walls or is channeled down corridors.
*   **Gameplay:** Firing a rocket into a tunnel entrance deals massive stability damage to everyone inside, even if they aren't hit directly. This creates a "Breach and Clear" mechanic.

### 6.2. Structural Integrity (The "Jenga" Effect)
*   **Concept:** Things fall down.
*   **Mechanism:**
    *   Run a background connectivity check (Union-Find) on `SOLID` voxel clusters.
    *   If a cluster is no longer connected to `z=0` (ground), it becomes a "Falling Object" entity.
    *   Falling objects crush mechs below them (`Instant Death` or massive damage).
*   **Gameplay:** Destroying the supports of a sniper tower isn't just about removing their cover—it's about dropping a building on their head.

### 6.3. Fluid Dynamics (Lava/Water Flow)
*   **Concept:** Liquids shouldn't be static blocks.
*   **Mechanism:** Cellular Automata (like Minecraft/Dwarf Fortress).
    *   If a `LAVA` voxel has `AIR` below it, it moves down.
    *   If a `LAVA` voxel has `AIR` beside it, it spreads.
*   **Gameplay:**
    *   **Trap:** Shoot a "holding tank" to flood a corridor with magma.
    *   **Denial:** Use terraforming weapons to create moat barriers.

### 6.4. Thermal Conductivity
*   **Concept:** Heat isn't just internal; it's environmental.
*   **Mechanism:**
    *   `LAVA` voxels radiate heat to adjacent air voxels.
    *   Mechs standing in "Hot Air" gain heat even if not touching lava.
    *   Flamers heat up the *voxels* they hit, creating temporary "fire zones."
*   **Gameplay:** Area denial via temperature. You can block a path not with a wall, but with an inferno that forces shutdowns.

