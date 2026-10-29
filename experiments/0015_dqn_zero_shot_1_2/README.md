# 0015 — DQN zero-shot 1-2

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

How does the selected 1-1 DQN behave on 1-2 before any 1-2 fine-tuning?

## Setup

```text
model:          runs/dqn_clean_start_1_1_8x1_buffer_finetune/dqn/dqn_final.zip
source level:   1-1
target level:   1-2
observation:    lean 188 → 954, fitted 1-1 preprocessor
start pool:     runs/neat_clean_450_1_2_transfer/pool_j07_cap1
```

Preview with the game window:

```bash
.venv/bin/python tools/watch.py \
  --config experiments/0015_dqn_zero_shot_1_2 \
  --model runs/dqn_clean_start_1_1_8x1_buffer_finetune/dqn/dqn_final.zip \
  --speedup 1 \
  --tag zero_shot_1_2_canonical
```

## Result

Canonical preview:

```text
finished:   yes
time:       9.1167 s
steps:      135
reward:     545.08
```

Full deterministic 1-2 evaluation:

```text
canonical:    9.1167 s
train pool:   17/31 finishes, median 5.5000 s, best 3.1833 s
holdout pool: 15/20 finishes, median 5.9500 s, best 3.1833 s
```

The policy transfers well enough to finish 1-2, but it is slow and misses
14/31 train starts and 5/20 holdout starts. This is the baseline for the
multi-level fine-tune in experiment 0016.
