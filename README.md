# echelon

Rapid-prototype DRL mech tactics demo.

## Setup (uv)

Requires `uv` and Python 3.13.

```bash
uv python install 3.13
uv sync -p 3.13
```

## Smoke test

```bash
uv run -p 3.13 python scripts/smoke.py --episodes 1 --num-per-team 3 --size 20 --mode full
```

## Train (PPO + LSTM)

```bash
uv run python scripts/train_ppo.py --num-per-team 2 --size 20 --mode full
# device selection: --device auto (default) | cpu | cuda | cuda:0 ...
```

Outputs:
- `runs/<run>/metrics.jsonl` (one JSON object per update)
- `runs/<run>/best.pt` + `runs/<run>/best.json` (best eval snapshot)

Resume:
```bash
uv run python scripts/train_ppo.py --run-dir runs/train --resume latest --updates 200
```
