# 0001 — PPO clean-start 1-1

Status: complete; 225k checkpoint selected
Started: 2026-10-28
Completed: 2026-10-28

> **Harness note (2026-10-28).** This record was produced with the old
> flag-based CLI. The current reproduction is config-driven:
>
> ```bash
> .venv/bin/python scripts/reinforcement/01_ppo.py \
>   --config experiments/0001_ppo_clean_start_1_1
> .venv/bin/python tools/evaluate.py \
>   --config experiments/0001_ppo_clean_start_1_1 \
>   --model runs/ppo_clean_start_1_1/ppo_pilot.zip
> ```
>
> The manifest owns the 512×512 architecture, lean preprocessing, start pool,
> and PPO hyperparameters. It records the completed run without
> `[run].resume_from`, so the command above starts a fresh run. To reproduce
> the continuation, add
> `--resume runs/ppo_clean_start_1_1_failed_1m/checkpoints/ppo-200000.zip`.

## Question

Can a simple lean-observation PPO policy learn to execute 1-1 from the clean
NEAT state starts, while remaining competitive from the canonical start?

## Setup

- Level: `1-1`
- Observation: game lean 188
- Preprocessing: Python one-hot → 954 inputs
- Model: 512 × 512 Tanh MLP
- Algorithm: PPO, 8 processes, 1,000,000 environment timesteps
- Starts: canonical 0.70 / 13 train states 0.30
- Holdout: 10 route-cluster states, evaluation only

## Provenance

```text
PLaiGROUND:      neat @ 707b745 (dirty during restructuring)
Game:            prepare-for-expo @ adb9278 (dirty)
Game dirty files: docs/project/decisions.md, docs/status.md,
                  docs/technical/training.md,
                  godot/tests/ai/test_ai_training_reward.gd,
                  godot/reports/report_20/ (untracked)
```

The run directory (`runs/ppo_clean_start_1_1/`) contains the saved lean
preprocessor, checkpoints, training log, TensorBoard logs, and evaluation
outputs.

## Data

```text
NEAT archive:       runs/neat_clean_450/archive/routes.json
Filtered routes:    runs/neat_clean_450/filtered/
Train route file:   runs/neat_clean_450/filtered/train_clean_routes.json
Holdout route file: runs/neat_clean_450/filtered/holdout_clean_routes.json
Train start pool:   runs/neat_clean_450/pool/train_start_pool.json
Holdout start pool: runs/neat_clean_450/pool/holdout_start_pool.json
```

The route split is cluster-level. All 10 holdout state keys also occur on train
routes; this is **not** a state-unseen holdout.

## Historical command (old flag CLI)

The current config-driven reproduction is in the harness note above. This block
records the exact old command shape.

```bash
.venv/bin/python scripts/reinforcement/01_ppo.py \
  --level 1-1 \
  --n-envs 8 \
  --timesteps 1000000 \
  --obs-encoding lean \
  --preprocessing lean \
  --start-pool runs/neat_clean_450/pool/train_start_pool.json \
  --action-repeat 4 \
  --max-decisions 150 \
  --n-steps 150 \
  --batch-size 120 \
  --epochs 10 \
  --net-arch 512 512 \
  --speedup 8 \
  --save-freq 100000 \
  --eval-every 0 \
  --patience 0 \
  --eval-start-pool runs/neat_clean_450/pool/train_start_pool.json \
  --eval-start-pool runs/neat_clean_450/pool/holdout_start_pool.json \
  --output-dir runs/ppo_clean_start_1_1
```

## Result

The original 1M run collapsed at ~296k and is preserved in
`runs/ppo_clean_start_1_1_failed_1m/`. The stable continuation resumed from
`ppo-200000.zip` and selected `ppo-225008.zip`:

```text
canonical:    100/100, median 2.3167 s
train pool:   14/14, median 3.5583 s, best 2.1500 s
holdout pool: 10/10, median 5.2250 s, best 3.4167 s
```

The selected model is `runs/ppo_clean_start_1_1/best_model.zip`. It is not
installed in the game. See `results.md` for the full history.
