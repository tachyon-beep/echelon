# Echelon Terrain System: Master Design Document

**Version:** 2.0 (The "Living Grid" Architecture)
**Status:** Implementation Blueprint

## 1. Core Philosophy

The Echelon terrain system is shifting from a static 2.5D heightmap generator to a **3D Voxel Simulation Medium**. The grid is an active participant in combat, enabling true verticality, destructibility, and systemic interactions (physics, fluid dynamics, connectivity).

**Key Architectural Pillars:**

1. **Fightability First:** Every map must guarantee playable lanes and connectivity, regardless of seed.
2. **True 3D Connectivity:** The fundamental unit of navigation is the "Walkable Node" (space + floor), not the 2D coordinate. Tunnels, bridges, and overhangs are first-class citizens.
3. **Determinism & Replayability:** A "Map Recipe" metadata structure completely decouples generation parameters from the random seed, ensuring debuggability.
4. **Composition over Noise:** Maps are built via a structured pipeline (Skeleton → Biomes → Details), not generic Perlin noise.

---

## 2. Generation Pipeline

The generation logic resides in `echelon/gen/` and follows a strictly ordered pass system. This ensures macro-scale tactical validity before micro-scale decoration.

### 2.1. The Pipeline Stages

1. **RNG Splitting:** The master seed spawns independent RNG streams for Layout, Biomes, and Details to prevent "butterfly effect" breakages.
2. **Skeleton Generation (The "Bones"):**

* Selects a **Macro Template** (e.g., "Cross & Ring", "Diagonal Arterial").
* Generates a `CorridorMask` (3D boolean volume) defining guaranteed open air for movement.
* *Constraint:* No biome module may overwrite the `CorridorMask`.

1. **Biome Assignment:**

* Partitions the map into 4 quadrants.
* Assigns **Biome Types** (e.g., Industrial, Arcology, Geo-Front) via permutation (no fixed corners).

1. **Module Placement:**

* Iterates through quadrants, executing Biome-specific "Brushes":
* **Pillar Brush:** Standard 2.5D buildings.
* **Plank Brush:** Horizontal bridges (Air above/below).
* **Worm Brush:** Tunnels (carving Air inside Solid).

1. **Spawn & Objective Clearing:**

* Identifies spawn corners (permutation) and central objective.
* Force-clears "Safe Zones" around these points.

1. **Global Transform:**

* Applies `FlipX`, `FlipY`, or `FlipXY` to the entire voxel array and metadata.

1. **3D Validation & Fixups:**

* Builds the adjacency graph.
* Runs A* to verify connectivity between Spawns and Objective.
* If disconnected, runs **"Staircase Digging"** to force a path.

---

## 3. Data Structures

### 3.1. The Map Recipe (`world.meta`)

This metadata is saved with replays to allow perfect reconstruction and debugging.

```python
@dataclass
class MapRecipe:
    seed: int
    generator_version: str = "v2_living_grid"
    
    # Structural Choices
    template_name: str         # e.g., "cross_and_ring"
    transform: str             # "identity", "flip_x", "flip_y", "flip_xy"
    
    # Assignments
    biome_layout: Dict[str, str]  # {"TL": "arcology", "BR": "geo_front"...}
    spawn_corners: Dict[str, str] # {"blue": "TL", "red": "BR"}
    
    # Validation Logs
    fixups: List[str]          # e.g., ["blue_to_center_dig_staircase_12"]

```

### 3.2. The Voxel World

The world is a 3D boolean array (initially), expandable to an ID array later.

* **Coordinate System:** `(z, y, x)` where Z is vertical (0=Floor, H=Ceiling).
* **Scale:** 1 Voxel ≈ 5m × 5m × 5m.
* **Grid Size:** Standard `50 × 50 × 20`.

---

## 4. Biome Catalog (The "Theaters")

Biomes dictate which "Brushes" are used during the Module Placement phase.

| Biome | Theme | Signature Structure | Tactical Hook |
| --- | --- | --- | --- |
| **Urban Residential** | Dense City | **Courtyards:** Hollow 2.5D blocks. | Close-quarters, standard lanes. |
| **Industrial Refinery** | Pipes & Tanks | **Spiderweb:** Messy catwalks & tanks. | Complex LOS, jumping required. |
| **Arcology Spire** | Brutalist Vert | **The Stacks:** Platforms at Z=5, Z=15. | **Knockback kills** (void floor). |
| **Geo-Front** | Subterranean | **Lava Tubes:** Enclosed tunnels. | No indirect fire inside; brawling. |
| **Glass Desert** | Crystalline | **Mirrors:** Reflective low cover. | Long sightlines, beam weapons buffer. |

---

## 5. The "Living Grid" Features (Future Proofing)

The architecture supports these systemic interactions for future phases:

* **Destructibility:** `SOLID` voxels can become `AIR` if HP reaches 0. Connectivity graph updates dynamically.
* **Falling Objects:** If a voxel cluster loses connection to `Z=0`, it falls, crushing units below.
* **Fluid Dynamics:** `LAVA` voxels flow down into empty `AIR` voxels.
* **Concussion:** Explosions trace rays through `AIR` voxels; confined spaces amplify damage.

---

## 6. Connectivity Validation: The Safety Net

To support 3D complexity without breaking the game, we use a **3D Adjacency Graph**.

### 6.1. Graph Definition

* **Node:** A voxel `(z,y,x)` is a node if it is `AIR` and has a `SOLID` floor at `z-1`.
* **Edge:** Connects adjacent nodes.
* *Flat:* `(z, y, x) ↔ (z, y+1, x)`
* *Ramp:* `(z, y, x) ↔ (z+1, y+1, x)` (Simulates climbing/stairs)
* *Drop:* `(z, y, x) → (z-N, y+1, x)` (One-way drop allowed)

### 6.2. The "Staircase Digger"

If `Spawn A` cannot reach `Objective`, the validator runs a weighted A* (High cost to break walls) to find a path.

* **Fixup:** Unlike 2D carving, 3D carving creates a "staircase" tunnel.
* *Algorithm:* If path moves from `(z, y, x)` to `(z+1, y+1, x)` through a solid wall, it clears `AIR` at the target *and* ensures a `SOLID` floor exists at `target_z - 1`.

---

## 7. Implementation Roadmap

### Phase 1: The Core (Immediate)

* [ ] Refactor `VoxelWorld` to accept `MapRecipe` metadata.
* [ ] Implement `VoxelWorldGenerator` pipeline (Skeleton → Fill → Transform).
* [ ] Implement `ConnectivityValidator3D` (Graph builder + basic check).
* [ ] **Milestone:** Generates guaranteed-playable 2.5D maps with metadata logs.

### Phase 2: True 3D (Next)

* [ ] Implement `StaircaseDigger` in the validator.
* [ ] Implement "Plank" and "Worm" brushes.
* [ ] Create "Arcology" biome using Plank brushes.
* [ ] **Milestone:** Agents can navigate multi-level maps; Replays show 3D structures.

### Phase 3: The Simulation (Future)

* [ ] Add Material IDs (Glass, Lava, Reinforced).
* [ ] Implement Destructibility & Falling logic.
* [ ] **Milestone:** "Battlefield" style destruction.
