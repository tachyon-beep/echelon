# Echelon (Agent Notes)

## Commands

- Use `uv` for setup and execution (prefer `uv run ...` over calling system `python` directly).
- Python: `3.13` (see `pyproject.toml` / `.python-version`).
- If sandboxed CI blocks `~/.cache/uv`, set `UV_CACHE_DIR=.uv_cache` (repo-local).

### Setup

- `uv python install 3.13`
- `uv sync -p 3.13`

### Tests

- Run all tests: `PYTHONPATH=. uv run -p 3.13 pytest tests`
- Fast unit tests: `PYTHONPATH=. uv run -p 3.13 pytest tests/unit`
- Integration tests: `PYTHONPATH=. uv run -p 3.13 pytest tests/integration`
- Performance tests (slow / hardware-sensitive): `PYTHONPATH=. uv run -p 3.13 pytest tests/performance`

### Smoke / Training

- Smoke episode: `uv run -p 3.13 python scripts/smoke.py --episodes 1 --packs-per-team 1 --size 40 --mode full`
- Train: `uv run -p 3.13 python scripts/train_ppo.py --packs-per-team 1 --size 100 --mode full`

### Lint

- `uv run -p 3.13 ruff check .`
