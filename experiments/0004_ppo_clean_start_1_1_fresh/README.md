# 0004 — Fresh PPO clean-start run on 1-1

Status: planned  
Started: 2026-10-28

## Question

How does a fresh lean PPO policy behave when training starts from a clean
model, using the latest 1-1 NEAT routes and the corrected standability-validated
start pool?

This run intentionally does **not** reuse any previous PPO checkpoint,
preprocessor, or output directory.

## Setup

```text
level:            1-1
observation:      lean 188 -> Python one-hot 954
model:            954 -> 512 -> 512 -> 2, Tanh
environments:      8
training starts:  canonical 0.70 / state 0.30
budget:           500,000 timesteps
PPO:              lr=1e-4, target_kl=0.02, ent_coef=0.02
checkpoints:      every 25,000
```

The new preprocessor is built from the current game tile vocabulary. The start
pool is rebuilt from `runs/neat_clean_450` and contains 12 train / 8 holdout
starts after rejecting non-standable cells.

## Output

```text
runs/ppo_clean_start_1_1_fresh/
```

Evaluation is separate:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0004_ppo_clean_start_1_1_fresh \
  --model runs/ppo_clean_start_1_1_fresh/checkpoints/ppo-400000.zip
```

## Result

Pending.
