# 0003 — raw-state DQN from clean starts (1-1)

Status: **planned**

This manifest records the current DQN defaults for the raw-state clean-start
path. It is the config-driven replacement for the old `02_dqn.py extract` /
`train` / `eval` command sequence.

## Data

```bash
.venv/bin/python tools/extract_route_starts.py --config experiments/0003_dqn_clean_start_1_1
```

Writes:

```text
runs/dqn_clean_start_1_1/spawns/1-1_spawn_pool.json
runs/dqn_clean_start_1_1/spawns/1-1_start_pool.json
```

The config points `[data].train_pool` at the flat start list.

## Train

```bash
.venv/bin/python scripts/reinforcement/02_dqn.py \
  --config experiments/0003_dqn_clean_start_1_1
```

For a smoke:

```bash
.venv/bin/python scripts/reinforcement/02_dqn.py \
  --config experiments/0003_dqn_clean_start_1_1 \
  --timesteps 10000 \
  --output-dir runs/dqn_clean_start_1_1_smoke
```

## Evaluate

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0003_dqn_clean_start_1_1 \
  --model runs/dqn_clean_start_1_1/dqn/dqn_final.zip
```

Canonical and the configured evaluation set run in one persistent Godot
process each. Results are written under `runs/dqn_clean_start_1_1/eval/<tag>/`.
