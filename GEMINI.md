# Echelon

## Project Overview

**Echelon** is a rapid-prototype Deep Reinforcement Learning (DRL) environment for mech tactics. It simulates 10v10 combat where teams are composed of "packs" (coordinated units of 10 mechs: 1 Heavy, 5 Medium, 3 Light, 1 Scout). The simulation takes place in a voxel-based world with physics-driven interactions like heat management, stability/knockdown, and line-of-sight targeting.

**Key Technologies:**
*   **Language:** Python 3.13
*   **ML Framework:** PyTorch (PPO implementation with LSTM)
*   **Environment:** Custom voxel engine with Gym-like API (`reset`/`step`)
*   **Backend/Viz:** FastAPI, Uvicorn, WebSockets
*   **Experiment Tracking:** Weights & Biases (WandB)
*   **Dependency Management:** `uv`

**Architecture:**
*   `echelon/env`: RL environment logic (observations, actions, rewards).
*   `echelon/sim`: Core simulation engine (physics, mechs, projectiles, world).
*   `echelon/gen`: Procedural content generation (terrain, objectives).
*   `echelon/rl`: Neural network models (Actor-Critic LSTM).
*   `echelon/arena`: League management (Glicko-2) for self-play.
*   `scripts/`: Entry points for training, evaluation, and utilities.

## Building and Running

The project is managed by `uv`. Ensure you have `uv` installed.

### Setup
```bash
uv sync
```

### Training (PPO)
Run the main training loop with PPO and LSTM. WandB is integrated for tracking.
```bash
# Basic training
uv run python scripts/train_ppo.py --packs-per-team 1 --size 100 --mode full

# With W&B tracking (recommended)
uv run python scripts/train_ppo.py \
  --wandb \
  --wandb-run-name "experiment-01" \
  --updates 500

# (Alternative) Enable via explicit project
uv run python scripts/train_ppo.py --wandb-project echelon --wandb-run-name "experiment-01"

# (Optional) Offline mode (no network / no login required)
uv run python scripts/train_ppo.py --wandb-mode offline --wandb-run-name "offline-01"
```

### Visualization
Start the replay server and open the viewer.
```bash
uv run server.py
# Then open viewer.html in a browser (e.g., http://localhost:8090/viewer.html usually implies opening the file directly or via the server if configured)
```
*Note: The server listens on port 8090 by default.*

### Arena Mode (Self-Play)
Train against a league of past agents.
```bash
# Initialize league
uv run python scripts/arena.py --league runs/arena/league.json init

# Add a commander
uv run python scripts/arena.py --league runs/arena/league.json add runs/train/best.pt --kind commander --name "Base Policy"

# Train against the league
uv run python scripts/train_ppo.py --train-mode arena --arena-league runs/arena/league.json --arena-submit
```

### Testing
```bash
# Run unit and integration tests
uv run pytest

# Run a quick smoke test of the env loop
uv run python scripts/smoke.py --episodes 1
```

## Development Conventions

*   **Code Style:** Adhere to `ruff` formatting and linting rules. Line length is set to 110.
    ```bash
    uv run ruff check .
    ```
*   **Type Hinting:** The codebase uses type hints extensively.
*   **Mutation Testing:** Critical simulation logic (`echelon/sim/sim.py`) is subject to mutation testing via `mutmut` to ensure robustness.
    ```bash
    uv run mutmut run
    ```
*   **Environment Design:**
    *   **Observations:** Partial or Full observability. Includes voxel maps, contact lists, and mech status.
    *   **Actions:** Discrete/Continuous hybrid (movement, targeting, EWAR toggles).
    *   **Sim Loop:** Fixed time step (`dt_sim`) with action repetition (`decision_repeat`).
*   **Dependencies:** All dependencies are managed in `pyproject.toml` and locked in `uv.lock`. Always use `uv sync` to update your environment.
