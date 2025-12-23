# Glossary

Domain terms, acronyms, and jargon used in Echelon.

## Acronyms

| Term | Expansion | Description |
|------|-----------|-------------|
| **AC** | Autocannon | Rapid-fire kinetic weapon; deals stability damage through sustained fire |
| **AMS** | Anti-Missile System | Point defense that shoots down incoming missiles |
| **DDA** | Digital Differential Analyzer | Algorithm for voxel raycasting; traverses grid cells along a ray |
| **DRL** | Deep Reinforcement Learning | Machine learning where neural networks learn from reward signals |
| **ECM** | Electronic Countermeasures | Jams enemy targeting; reduces lock-on effectiveness |
| **ECCM** | Electronic Counter-Countermeasures | Counters ECM; improves targeting through jamming |
| **EWAR** | Electronic Warfare | ECM/ECCM systems collectively |
| **GAE** | Generalized Advantage Estimation | Technique for computing advantage in PPO; balances bias/variance |
| **HP** | Hit Points | Health/damage capacity |
| **LOS** | Line of Sight | Whether one position can see another; computed via raycasting |
| **LRM** | Long Range Missile | Indirect fire weapon; arcs over obstacles, requires lock |
| **LSTM** | Long Short-Term Memory | Recurrent neural network architecture; handles temporal dependencies |
| **PPO** | Proximal Policy Optimization | RL algorithm used for training; stable, sample-efficient |
| **RL** | Reinforcement Learning | Learning paradigm based on reward maximization |
| **SPS** | Steps Per Second | Training throughput metric |
| **SRM** | Short Range Missile | Direct fire missile; faster, shorter range than LRM |

## Domain Terms

| Term | Description |
|------|-------------|
| **Biome** | Procedural terrain style (urban, forest, industrial, etc.); painted into quadrants |
| **Contact** | Detected entity in observation space; limited to top-K visible units |
| **Decision Repeat** | Number of simulation steps per RL decision; controls action frequency |
| **Glicko-2** | Rating system for self-play league; tracks skill with uncertainty |
| **Hall of Fame** | Archive of promoted policy snapshots; opponents sampled from here |
| **Heat** | Resource that accumulates from firing weapons; overheat causes shutdown |
| **Knockdown** | State when stability reaches zero; mech falls, temporarily immobilized |
| **NavGraph** | Navigation graph built from walkable voxel surfaces; used for pathfinding |
| **Pack** | Group of 10 mechs (1 Heavy, 5 Medium, 3 Light, 1 Scout); coordination unit |
| **Paint / Target Lock** | Marking an enemy for pack-wide targeting bonus; enables missile lock |
| **Quadrant** | Quarter of the map; each gets a biome assignment |
| **Recipe** | Deterministic map specification; seed + config → reproducible terrain |
| **Stability** | Resource depleted by kinetic damage; zero triggers knockdown |
| **Voxel** | Volumetric pixel; 3D grid cell with material type and HP |
| **Walkable Air** | AIR voxel with SOLID floor below; basis for navigation graph |

## Mech Classes

| Class | Role | Pack Count | Signature Weapons |
|-------|------|------------|-------------------|
| **Heavy** | Anchor, fire support | 1 | Laser, Missile (LRM), Gauss |
| **Medium** | Versatile line unit | 5 | Laser, Autocannon |
| **Light** | Flanker, harasser | 3 | Flamer, Laser, Paint |
| **Scout** | Recon, electronic warfare | 1 | Paint, ECM/ECCM |

## Voxel Materials

| Material | Properties |
|----------|------------|
| **AIR** | Empty space; passable |
| **SOLID** | Standard terrain; destructible |
| **REINFORCED** | High HP; harder to destroy |
| **GLASS** | Transparent; blocks movement, allows LOS |
| **FOLIAGE** | Blocks LOS; passable |
| **WATER** | Passable; may affect heat dissipation |
| **LAVA** | Hazard; damages mechs |
| **DEBRIS** | Destroyed terrain remnants |

## Action Indices

| Index | Name | Description |
|-------|------|-------------|
| 0 | FORWARD | Forward/backward movement throttle |
| 1 | STRAFE | Left/right movement throttle |
| 2 | VERTICAL | Jump jet / vertical movement |
| 3 | YAW_RATE | Turn rate |
| 4 | PRIMARY | Main weapon (class-dependent) |
| 5 | VENT | Heat dump |
| 6 | SECONDARY | Secondary weapon / EWAR toggle |
| 7 | TERTIARY | Tertiary weapon / paint |
| 8 | SPECIAL | Smoke deployment |
