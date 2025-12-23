# C4 Architecture Diagrams

**Project:** Echelon
**Analysis Date:** 2025-12-24

---

## 1. Level 1: System Context Diagram

```mermaid
graph TB
    subgraph Users["External Actors"]
        Trainer["Training Script<br/>(train_ppo.py)"]
        Evaluator["Evaluator<br/>(eval_policy.py)"]
        Visualizer["Visualization User<br/>(viewer.html)"]
    end

    subgraph External["External Systems"]
        WB["W&B<br/>(Experiment Tracking)"]
        Gym["Gymnasium<br/>(Optional Interface)"]
    end

    subgraph Core["Echelon System"]
        Echelon["Echelon<br/>Deep RL Mech Combat<br/>Environment"]
    end

    subgraph Supporting["Supporting Services"]
        Server["Replay Server<br/>(FastAPI/SSE)<br/>Port 8090"]
        Arena["Arena/Self-Play<br/>(arena.py)"]
    end

    Trainer -->|step/reset| Echelon
    Evaluator -->|evaluate policy| Echelon
    Echelon -->|metrics| WB
    Gym -->|optional| Echelon
    Visualizer -->|SSE stream| Server
    Arena -->|league execution| Echelon
```

**Context:** Echelon acts as a central multi-agent RL environment, receiving training commands from PPO scripts and exposing observations/rewards through a Gymnasium-compatible interface.

---

## 2. Level 2: Container Diagram

```mermaid
graph TB
    subgraph Scripts["Scripts"]
        TrainPPO["train_ppo.py"]
        EvalPolicy["eval_policy.py"]
        ArenaScript["arena.py"]
    end

    subgraph Environment["Environment Layer"]
        Env["EchelonEnv<br/>reset() / step()"]
    end

    subgraph Generation["Map Generation"]
        Layout["Layout Generator"]
        Biomes["Biome Painter"]
        Validator["Connectivity Validator"]
    end

    subgraph Simulation["Simulation Core"]
        Sim["Sim<br/>(Physics & Combat)"]
        World["VoxelWorld<br/>(3D Grid)"]
        MechState["MechState[]<br/>(20 Mechs)"]
    end

    subgraph Navigation["Navigation"]
        NavGraph["NavGraph"]
        Planner["Planner<br/>(A*)"]
    end

    subgraph RL["RL Components"]
        Model["ActorCriticLSTM"]
        League["League<br/>(Self-Play)"]
    end

    TrainPPO -->|step/reset| Env
    Env -->|generate map| Layout
    Layout --> Biomes --> Validator --> NavGraph
    Env -->|manage episode| Sim
    Sim -->|query/modify| World
    Sim -->|update| MechState
    TrainPPO -->|load/evaluate| Model
    League -->|execute| Env
```

---

## 3. Level 3: Simulation Core Components

```mermaid
graph TB
    subgraph Input["Input"]
        Actions["Actions[9]<br/>(per agent)"]
    end

    subgraph Physics["Physics & State"]
        PhysicsIntegration["Physics Integration"]
        Stability["Stability System"]
        HeatModel["Heat Management"]
    end

    subgraph Combat["Combat & Targeting"]
        Los["LOS<br/>(Numba DDA)"]
        Targeting["Weapon Targeting"]
        Projectiles["Projectile Manager"]
    end

    subgraph State["World State"]
        VoxelWorld["VoxelWorld<br/>[z,y,x] grid"]
        MechStates["MechState[20]"]
        SpatialGrid["SpatialGrid<br/>(O(1) queries)"]
    end

    subgraph Output["Output"]
        Events["Combat Events"]
    end

    subgraph Main["Orchestrator"]
        Sim["Sim.step()"]
    end

    Actions --> Sim
    Sim --> PhysicsIntegration
    Sim --> HeatModel
    Sim --> Los
    Sim --> Targeting
    Sim --> Projectiles
    Sim --> VoxelWorld
    Sim --> MechStates
    Sim --> SpatialGrid
    Sim --> Events
```

---

## 4. Level 3: Environment Components

```mermaid
graph TB
    subgraph Episode["Episode Management"]
        Reset["reset()"]
        Step["step(actions)"]
        Terminate["Termination Check"]
    end

    subgraph Observation["Observation (607 dims)"]
        Contacts["Contacts<br/>110 dims"]
        PackComm["Pack Comm<br/>80 dims"]
        LocalMap["Local Map<br/>121 dims"]
        Telemetry["Telemetry<br/>256 dims"]
        SelfFeatures["Self Features<br/>40 dims"]
    end

    subgraph Rewards["Reward Shaping"]
        ZoneTick["Zone Control (0.10)"]
        Approach["Approach (0.25)"]
        Damage["Damage (0.005)"]
        Kill["Kill (1.0)"]
        Terminal["Terminal (±5.0)"]
    end

    Reset --> Contacts
    Reset --> LocalMap
    Reset --> Telemetry

    Step --> Contacts
    Step --> PackComm
    Step --> SelfFeatures
    Step --> ZoneTick
    Step --> Approach
    Step --> Damage
    Step --> Kill

    Terminate --> Terminal
```

---

## 5. Level 3: Procedural Generation Pipeline

```mermaid
graph TB
    subgraph Input["Input"]
        Seed["Seed + WorldConfig"]
    end

    subgraph Pipeline["Generation Pipeline"]
        LayoutGen["generate_layout()<br/>(quadrant split)"]
        BiomeAssign["Assign Biomes<br/>(16 types)"]
        BiomePaint["BiomeBrush Paint"]
        MacroCorridor["carve_macro_corridors()"]
    end

    subgraph Validation["Validation & Fixup"]
        NavGraphTest["NavGraph Build"]
        Fallback["2D A* Fallback"]
        Staircase["Staircase Carving"]
    end

    subgraph Output["Output"]
        VoxelWorld["VoxelWorld"]
        NavGraph["NavGraph"]
        Recipe["Recipe Hash"]
    end

    Seed --> LayoutGen
    LayoutGen --> BiomeAssign
    BiomeAssign --> BiomePaint
    BiomePaint --> MacroCorridor
    MacroCorridor --> NavGraphTest
    NavGraphTest -->|success| Output
    NavGraphTest -->|fail| Fallback
    Fallback --> Staircase
    Staircase --> Output
```

---

## 6. Data Flow: Training Loop

```mermaid
graph LR
    subgraph Training["Training"]
        Script["train_ppo.py"]
        Batch["Batch Collection<br/>(N=2048)"]
    end

    subgraph Execution["Execution"]
        Reset["env.reset()"]
        Step["env.step()"]
        SimStep["sim.step()"]
    end

    subgraph Observation["Observation"]
        BuildObs["Build 607-dim obs"]
        Reward["Compute reward"]
    end

    subgraph Policy["Policy Update"]
        GAE["Compute GAE"]
        Loss["PPO Loss"]
        Optim["Optimizer.step()"]
    end

    Script --> Batch
    Batch --> Reset
    Batch --> Step
    Step --> SimStep
    Step --> BuildObs
    Step --> Reward
    Batch --> GAE
    GAE --> Loss
    Loss --> Optim
    Optim --> Script
```

---

## 7. Data Flow: Self-Play Loop

```mermaid
graph TB
    subgraph League["League"]
        Init["Initialize Pool"]
        Sample["Sample Opponent"]
    end

    subgraph Match["Match"]
        Load["Load Policies"]
        Play["play_match()"]
    end

    subgraph Result["Resolution"]
        Winner["Determine Winner"]
        GameResult["Create GameResult"]
    end

    subgraph Rating["Glicko-2"]
        Snapshot["Snapshot Ratings"]
        Compute["rate()"]
        Commit["Two-Phase Commit"]
    end

    subgraph Promotion["Promotion"]
        Check["In top-K?"]
        Promote["Promote to Commander"]
    end

    Init --> Sample
    Sample --> Load
    Load --> Play
    Play --> Winner
    Winner --> GameResult
    GameResult --> Snapshot
    Snapshot --> Compute
    Compute --> Commit
    Commit --> Check
    Check -->|yes| Promote
    Check -->|no| Sample
    Promote --> Sample
```

---

## Summary

| Diagram | Purpose |
|---------|---------|
| Level 1: System Context | External actors and systems |
| Level 2: Container | Major subsystem containers and data flow |
| Level 3: Simulation | Physics, combat, targeting components |
| Level 3: Environment | Observation and reward construction |
| Level 3: Generation | Procedural map synthesis pipeline |
| Data Flow: Training | PPO training loop |
| Data Flow: Self-Play | League and Glicko-2 rating flow |

---

*Generated on 2025-12-24*
