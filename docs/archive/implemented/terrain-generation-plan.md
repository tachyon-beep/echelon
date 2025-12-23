# Terrain Generation Plan (Model Biomes + Fightable Corridors)

This document proposes a “model terrain” system for Echelon: maps with **four obvious biome quarters** that remain reliably **fightable** thanks to **guaranteed artificial corridors/roads**, while maximizing replayability via:
- **Biome↔quadrant permutation** (no biome fixed to a corner)
- **Spawn permutation/rotation** (no team fixed to BL/TR)
- Cheap whole-map **reorientation transforms**: `identity | flip_x | flip_y | flip_xy`

The goal is to produce terrain that is:
- Immediately readable in replays (clear biome identity and landmark silhouettes).
- Tactically interesting (cover cadence, lane choices, chokepoints, flank routes, mixed LOS ranges).
- Deterministic and debuggable (seed + metadata fully reconstructs layout decisions).
- Hard to “solve” (no fixed corner meaning; many seeds/variants).

Current integration points:
- `echelon/sim/world.py` → `VoxelWorld.generate()` (terrain entry point)
- `echelon/env/env.py` → spawn clearing and replay serialization (includes `world.meta`)
- `viewer.html` → renders walls and displays world metadata

---

## Design principles

1) **Fightability first**: macro traversal lanes are guaranteed; biomes decorate without breaking play.
2) **Silhouette over noise**: each biome has signature shapes and 1–3 large landmarks; avoid “random box soup”.
3) **Replayability via variants**: flips and permutations multiply variety without extra generator complexity.
4) **Determinism + diagnostics**: store a “map recipe” in `world.meta` and validate connectivity.
5) **Manual-tweak friendly**: stable seed streams and explicit modules make iteration predictable.

---

## v1 constraints (fits current engine)

- World representation is currently `solid[z, y, x]` (bool voxels). Biomes must be readable via **shape**, not materials.
- Carving is done by clearing solids in boxes/lines (roads/corridors).
- Mech footprints are small (heavy ~2×2 voxels). Corridors must be wide enough to pass and fight.

Future (v2+):
- Material IDs (soft cover, slow terrain, foliage blocks LOS but not movement, etc.).
- Heightmaps/slopes, destructibility, dynamic props.

---

## High-level pipeline (“canonical map + variants”)

Generate a single **canonical** map and then apply variant operations:
- `biome_assignment` (permutation)
- `spawn_assignment` (rotation/permutation)
- `transform` (flip X/Y/both)

This ensures:
- No corner becomes “the dense biome corner” permanently.
- Map orientation cannot be exploited by fixed strategies.
- Debugging remains easy (the canonical recipe is logged).

### 0) Seed splitting (determinism without accidental coupling)
Use independent RNG streams so a tweak in one module doesn’t reshuffle everything:
- `rng_layout`: macro corridors + biome borders
- `rng_biomes`: fillers per biome
- `rng_landmarks`: set pieces
- `rng_variants`: biome/spawn assignment + transforms

Implementation hint: `np.random.SeedSequence(seed).spawn(k)` → `np.random.default_rng(child)`.

### 1) Choose biome palette + assign to quarters (per seed)
- Pick 4 biomes from a catalog (weighted or uniform).
- Assign them to quarter regions by a random permutation.

Important: quarters are a **structural partition**, not a semantic meaning.

### 2) Build macro fightability skeleton (corridors/roads)
Create and record a corridor plan that guarantees playability:
- At least **2 independent routes** from each spawn region to the center.
- At least **1 cross-map arterial** that crosses multiple biomes.
- At least **1 ring-ish route** near the objective to enable flanks.

Corridor carving is enforced twice:
1) Before biome fill (biomes should avoid filling corridor masks).
2) After biome fill (final “clear pass”).

### 3) Place large landmarks (biome signature anchors)
Each biome places 1–3 “readable from orbit” set pieces. These shapes make the biome obvious even with uniform wall colors.

### 4) Fill biomes (mid-scale patterns + micro cover)
Biome modules place their characteristic patterns while respecting:
- corridor masks
- center objective region (kept readable and contestable)
- spawn clear regions (applied later, after spawn assignment)

### 5) Blend borders (transition bands)
Biomes should not have perfect straight seams. Use a transition band to:
- jitter boundaries (low-frequency noise)
- mix a small amount of neighboring biome props
- optionally place intentional boundary features (canal/rail/highway/greenbelt)

### 6) Assign spawns (unfixed) + clear spawn regions
To avoid “solving the test”:
- Choose `blue_corner` uniformly among corners.
- Set `red_corner = opposite(blue_corner)` for fairness.
- Clear spawn regions in those chosen corners (after generation or in canonical then transform).

### 7) Apply map transform (4 variants for free)
Choose `transform ∈ {identity, flip_x, flip_y, flip_xy}`.

Transform applies to:
- `solid[z,y,x]`
- biome masks / region metadata
- landmark bboxes
- corridor polylines
- spawn corners

### 8) Validate + auto-fix
Run connectivity/density checks; if failing:
- carve a new corridor along the cheapest blocked path (A* on footprint)
- widen corridors below minimum width
- thin blockers inside corridor masks

Record any fix-ups in metadata.

---

## Macro corridor templates (fightability skeleton)

Each map chooses one macro template (plus small random perturbations):

### Template A: Cross + Ring
- Two major boulevards: N–S and E–W.
- Inner ring (or partial ring) around the objective.
- 1–2 diagonals as optional connectors.

Good for: predictable flanks, symmetric engagement options.

### Template B: Diagonal arterial + spurs
- One major diagonal route crossing all biomes.
- Two spurs that connect the “off diagonal” corners.
- Small plazas at intersections.

Good for: strong “main front” plus flank routes.

### Template C: Looped S arterial
- A curving “S” route with 3–5 connectors.
- Objective sits near a junction rather than dead center.

Good for: varied sightlines and less “mirror match” feel.

### Corridor width guidance (voxels)
- Target width: **4–6** voxels (comfortably passes heavy units and allows fights).
- Minimum width: **3** voxels (hard floor; below this feels bad).
- Open plazas: radii that scale with map size (e.g., 6–10 voxels).

---

## Implementation notes / triage (pragmatic guidance)

These are concrete implementation patterns that tend to keep a procedural system “production-grade” (deterministic, debuggable, and hard to exploit).

### Canonical + variants should be a first-class contract
Treat generation as two explicit stages:
1) **Canonical generation** (build all geometry + masks + structured artifacts).
2) **Variant application** (biome permutation, spawn assignment, transform).

To avoid “we flipped solids but forgot to flip spawns/landmarks”, define a canonical structure that includes *everything that has coordinates*:
- `solid[z,y,x]`
- biome mask(s) `biome_id[y,x]`
- masks: `corridor_mask`, `objective_mask`, `border_band_mask`, `do_not_fill_mask`
- corridor plan as polylines (list of `{points, width}`)
- landmarks as structured records (bbox + tags)
- spawn slots (corner boxes/regions)

### Seed splitting: stable streams, recorded
Use `np.random.SeedSequence(seed).spawn(k)` to create stable named streams, e.g.:
- `layout`, `biomes`, `landmarks`, `variants`, `fixups`

Record the child seed integers in metadata so “why did this seed change after refactor?” is diagnosable.

### Corridors as polylines → rasterized masks
Represent macro corridors as polylines + width, then rasterize into a `corridor_mask[y,x]` via:
- Bresenham line rasterization per segment
- “square brush” expansion to achieve width (fast and good enough for voxel grids)

Enforce corridors twice:
- Pre-fill: biomes/landmarks don’t place solids in the mask.
- Post-fill: final clear pass (guarantee).

### Quarter partition jitter (no Perlin required)
To keep “four quarters” obvious but avoid ruler-straight seams:
- Start from `mid_x`, `mid_y`
- Create `x_split[y]` and `y_split[x]` using low-frequency jitter:
  - sample offsets at sparse control points (every 8–12 cells)
  - linearly interpolate offsets between control points

This yields organic borders that are still quarter-like at macro scale.

### Validation + fixups: weighted “digging” A*
For connectivity validation:
- build a **2D nav grid** from `solid` by OR-ing across a clearance Z band
- inflate obstacles to account for mech footprint (configuration space)
- run a **weighted A*** where “blocked” cells are high cost (digging)
- carve a corridor tube along the resulting path if digging occurs

For “two routes”, compute path 1, then build a penalty tube around it and compute path 2; record an overlap ratio.

This “safety net” is implemented as `ConnectivityValidator` in `echelon/gen/validator.py` (no SciPy dependency) and wired into `EchelonEnv.reset()` behind `WorldConfig.ensure_connectivity`.

### Corridor width enforcement (optional, later)
Even with corridor masks, it’s easy for biomes to pinch lanes. A pragmatic post-pass is:
- sample along corridor polylines
- measure local free width (scan perpendicular)
- widen if below minimum and record fixup

---

## Biome catalog (broad mix of generic areas + set pieces)

The map always uses 4 biomes at a time, but the catalog should be large so seeds feel different over time.

Each biome definition includes:
- **Signature geometry**: must-have shapes.
- **Lane style**: how corridors appear in that biome.
- **Landmarks**: 1–3 set pieces + supporting props.
- **Tactical profile**: what this biome is “good for”.

### 1) Urban Residential (Dense blocks)
- Signature: courtyard apartment blocks (hollow rectangles), alley grid, occasional towers.
- Lane style: 1 boulevard + multiple alleys; short connectors at boundaries.
- Landmarks: high-rise trio, park square, “metro entrance”.
- Tactical: high cover, short fights, strong flanks; risky chokepoints if overfilled.

### 2) Urban Commercial (Downtown)
- Signature: tall blocks, wider streets, billboard canyons, parking structures.
- Lane style: 2–3 boulevards with cover islands at intersections.
- Landmarks: mall block, skyscraper cluster, plaza with low walls.
- Tactical: long linear lanes; punishes open crossings; needs cover ribbons.

### 3) Suburban Sprawl
- Signature: smaller building footprints, fences/hedges, cul-de-sac patterns, strip mall row.
- Lane style: curvy roads, roundabouts, many approach angles.
- Landmarks: school campus, water tower, shopping strip.
- Tactical: mixed cover and angles; can become “samey” without strong landmarks.

### 4) Civic / Political District (Government)
- Signature: symmetrical large buildings, plazas, colonnade-like low walls.
- Lane style: ceremonial axes (wide) + guard walls shaping approach.
- Landmarks: capitol dome, fortified gate, comms tower.
- Tactical: readable lanes and long LOS; requires deliberate low cover to avoid sniper dominance.

### 5) Military Base / Facility
- Signature: perimeter walls, controlled gates, hangars, bunkers.
- Lane style: perimeter road + internal grid; ≥2 entrances required.
- Landmarks: hangar bay, bunker complex, helipad.
- Tactical: strong defensive holds; must guarantee multiple breach paths.

### 6) Industrial Refinery / Factory
- Signature: large sheds, tank farms, pipe corridors (thin walls), height variation.
- Lane style: truck routes (wide) + maintenance corridors (narrow).
- Landmarks: cooling towers, refinery stacks, substation yard.
- Tactical: mid-range cover labyrinth; validation must prevent dead-end mazes.

### 7) Logistics: Rail Yard / Container Port
- Signature: parallel “track” corridors, container stacks, crane silhouettes.
- Lane style: long straight corridors with intermittent cover islands; contested crossings.
- Landmarks: crane tower, container canyon, station platform.
- Tactical: strong lane fights and kiting; needs multiple crossings to avoid single-lane dominance.

### 8) Rural Farmland / Irrigation
- Signature: large fields, hedgerows/ditches, barns/silos clusters.
- Lane style: dirt road network (T/H junctions) + canal crossings.
- Landmarks: grain silos, barn cluster pocket, windmill line.
- Tactical: directional cover via hedges; mixes mid/long fights; can be too open if under-hedged.

### 9) Forest / Woods
- Signature: tree trunk columns in clusters, clearings, rock outcrops.
- Lane style: logging road + firebreak corridors (clear lanes through forest).
- Landmarks: ranger station, clearcut patch, ridge outcrop.
- Tactical: broken LOS; ambush angles; needs density caps to keep movement pleasant.

### 10) Mining / Quarry
- Signature: open pit “bowl” (v1 via stepped walls/terraces), spoil piles, crusher site.
- Lane style: rim road + stepped switchbacks.
- Landmarks: crusher plant, conveyor corridor, spoil ridge.
- Tactical: dramatic landmark fights; needs careful stepped traversal without real slopes.

### 11) Scrapyard / Dump (Wasteland)
- Signature: irregular piles, container stacks, broken walls, wreck canyon.
- Lane style: 1 truck road + multiple narrow passages; avoid dead ends.
- Landmarks: crane, compactor yard, wreck canyon.
- Tactical: brawly cover density; can become frustrating if not validated.

### 12) Research / Government Facility (Black site)
- Signature: clean geometric perimeter, internal courtyards, sensor towers.
- Lane style: controlled corridors, “security checkpoints” as choke + flank vents.
- Landmarks: main lab block, satellite dish, perimeter gatehouse.
- Tactical: deliberate chokepoints; must add alternate breach routes for fun.

---

## Set piece library (reusable modules)

Reusable landmarks keep biomes readable and speed iteration. Examples:
- Stadium / arena (ring/rectangle with interior courtyard)
- Hospital / campus cluster
- Power plant (cooling towers + substation yard)
- Bridge + canal (boundary feature with limited crossings)
- Train station (platform + track corridors)
- Airport runway strip (huge clear lane + hangars + cover islands)
- Radar dish / comms tower (tall silhouette + low-wall yard)
- Dam/spillway (wall line + canal + crossing points)

Each set piece should be parameterized and recorded in metadata:
- footprint bbox, height profile, orientation, and any connected corridors.

---

## Border blending (“blended but intentional”)

Biome borders should feel authored:
- Prefer placing an intentional separator (also a playable lane):
  - canal/ditch + bridges
  - rail line + crossings
  - highway + underpasses
  - park belt + paths
- Add a narrow transition band:
  - jitter boundary line
  - mix props at low rate (e.g., trees near farms, suburbs near civic district)

Goal: you should *feel* the biome change, but not see a perfect straight seam.

---

## Validation (fightability + fairness)

### Connectivity checks (hard requirement)
Create a 2D footprint passability grid and enforce:
- `spawn(blue) → center` has ≥2 distinct paths
- `spawn(red) → center` has ≥2 distinct paths
- `spawn(blue) ↔ spawn(red)` connected
- each biome connects to at least 2 neighbors (avoid isolated pockets)

Fix-up strategy:
- A* on footprint to find a minimal carve path.
- Carve/widen corridor along that path.
- Re-check until passing or until a bounded number of fix-ups.

### Corridor width enforcement
Given heavy units are the largest footprint, enforce minimum widths:
- target: 4–6 voxels
- minimum: 3 voxels

### Density targets (per biome)
Track density in 2D footprint and clamp via thinning/adding cover ribbons:
- Dense urban: high density, but must maintain alley network.
- Civic/government: lower density, but must include low-cover ribbons in open areas.
- Farmland: low density overall, but high count of long hedgerow segments.
- Forest: medium density; prefer many thin columns rather than big blocks.
- Industrial/scrap: high density, but must maintain multiple lanes.

### “Biome readability” heuristics
Each biome should satisfy at least one signature count, e.g.:
- Residential: ≥N courtyard blocks
- Farmland: ≥N hedgerow segments + ≥1 farm cluster
- Forest: ≥N trunk columns + ≥2 clearings
- Government: ≥2 large buildings + ≥1 plaza + ≥1 low-wall ribbon

Record these stats in metadata so debugging is fast.

---

## Replayability controls (avoid “solving the test”)

### Biome permutation
- Never bind a biome to a named quadrant.
- Record `biomes_by_quadrant` in metadata.

### Spawn rotation/permutation
- Choose spawn corners per seed, not fixed BL/TR.
- Prefer opposite corners for fairness; record in metadata.

### Whole-map transforms
Pick a transform uniformly:
- `identity`
- `flip_x`: mirror across vertical axis
- `flip_y`: mirror across horizontal axis
- `flip_xy`: flip both

Definition (for coordinate `(x, y)`):
- `flip_x`: `x' = (size_x - 1) - x`, `y' = y`
- `flip_y`: `x' = x`, `y' = (size_y - 1) - y`
- `flip_xy`: both

Transforms must apply consistently to:
- voxel solids
- corridor polylines
- landmark bboxes
- biome masks
- spawn corners

---

## Metadata (“map recipe”) for debugging + manual tuning

Store a compact, human-readable recipe in `world.meta`, included in replay:

Example shape:
- `generator`: name/version string (e.g., `model_biomes_v1`)
- `seed`
- `variant`: `{ "transform": "flip_x", "biome_perm": [...], "spawn": {"blue":"TR","red":"BL"} }`
- `biomes`: `{ "BL":"industrial_refinery", "BR":"forest", "TL":"civic", "TR":"farmland" }`
- `corridors`: list of `{type, points, width}`
- `landmarks`: list of `{id, biome, bbox, notes}`
- `stats`: density + signature counts + connectivity metrics
- `fixups`: list of any auto-carves/widenings performed

This makes “why does this seed look like this?” answerable from a replay alone.

---

## Implementation roadmap (suggested phases)

### Phase 1: Variants + metadata plumbing (low risk, high value)
- Spawn permutation (opposite corners) and record it.
- Transform system (`identity/flip_x/flip_y/flip_xy`) and record it.
- Ensure replay viewer shows key metadata (biomes/spawns/transform).

### Phase 2: Macro corridor skeleton
- Implement corridor templates + carving utilities.
- Central objective “nexus” region with consistent readability.
- Connectivity validation and corridor widening (prototype implemented as `ConnectivityValidator` in `echelon/gen/validator.py` and wired into `EchelonEnv.reset()`).

### Phase 3: Biome modules (start with 4–6 strong biomes)
- Implement 4 foundational biomes:
  - urban_residential, civic_government, farmland, forest
- Add 2 more:
  - industrial_refinery, rail_yard

### Phase 4: Border blending
- Transition band jitter + prop mixing.
- Add 1 boundary feature (canal/rail/highway) as a first-class option.

### Phase 5: Expand biome + set piece library
- Mining/quarry, military base, scrapyard, commercial downtown, research facility, etc.
- Add more landmark modules and keep them parameterized.

---

## Manual tuning workflow (recommended)

1) Pick a seed and run a short recorded replay.
2) In the viewer, read:
   - seed, transform, biome assignment, spawn corners, landmark list.
3) Adjust a small parameter set (density, corridor widths, border band width, landmark sizes).
4) Re-run the same seed to compare.

The combination of:
- stable sub-RNG streams
- explicit `world.meta` recipe
- deterministic validation/fixups
…makes iteration predictable and fast.

---

## v2 extensions (optional but high impact)

1) **Material IDs** instead of bool solids:
   - foliage blocks LOS but not movement (soft cover)
   - mud increases movement cost
   - water/canal requires bridges
2) **Height variation rules** (true multi-level play):
   - ramps/terraces for mining pit
   - overpasses for highways
3) **Destructible cover**:
   - thin walls break under sustained fire, creating evolving lanes

These should only be added after v1 geometry consistently produces fun fights.
