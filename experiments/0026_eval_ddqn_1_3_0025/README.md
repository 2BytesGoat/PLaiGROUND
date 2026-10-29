# 0026 — 1-3 evaluation for DDQN 0025

Status: planned

## Purpose

Evaluate the 0025 DDQN checkpoints on 1-3 without the multi-level start-pool
limitation of `evaluate_run`.

## Sets

```text
train:    runs/neat_frequent_rebalance_1_3/pool/train_start_pool.json   45 starts
holdout:  runs/neat_frequent_rebalance_1_3/pool/holdout_start_pool.json 23 starts
canonical 1-3
```

## Command

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0026_eval_ddqn_1_3_0025 \
  --model runs/dqn_multilevel_1_1_1_2_1_3_ddqn_300k/dqn/checkpoints/<checkpoint>.zip \
  --tag <checkpoint-name>
```

Use the selected 0025 checkpoint after training; do not replace the 1-1/1-2
reference until its canonical and holdout gates also pass.
