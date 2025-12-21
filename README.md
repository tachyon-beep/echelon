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
uv run -p 3.13 python scripts/smoke.py --episodes 1 --packs-per-team 1 --size 40 --mode full
```

## Train (PPO + LSTM)

```bash
uv run python scripts/train_ppo.py --packs-per-team 1 --size 100 --mode full
# device selection: --device auto (default) | cpu | cuda | cuda:0 ...
```

Arena training (red team sampled from a Glicko-2 league):
```bash
uv run python scripts/arena.py --league runs/arena/league.json init
uv run python scripts/arena.py --league runs/arena/league.json add runs/lots_of_packs/best.pt --kind commander --name "Founding Commander"
uv run python scripts/train_ppo.py --train-mode arena --arena-league runs/arena/league.json --packs-per-team 1 --size 50 --mode full --arena-submit
```
Note: `--train-mode arena` expects your env flags (`--packs-per-team`, `--size`, etc.) to match the league’s `env_signature`.
`--arena-submit` saves `runs/<run>/arena_candidate.pt`, plays `--arena-submit-matches` games vs the league pool, updates ratings, and promotes the candidate if eligible.

Outputs:
- `runs/<run>/metrics.jsonl` (one JSON object per update)
- `runs/<run>/best.pt` + `runs/<run>/best.json` (best eval snapshot)

Resume:
```bash
uv run python scripts/train_ppo.py --run-dir runs/train --resume latest --updates 200
```

## Teams / Packs

Teams spawn in “packs” to preserve combined-arms synergies.

- `--packs-per-team N` spawns `N` identical packs on each team.
- Each pack is 10 mechs: 2 Heavy, 5 Medium, 3 Light.
- Packs are the default coordination unit; comms/coordination features are scoped to pack membership.
- Painted-target locks/bonuses are pack-scoped (not shared across packs).
