# 0012 — DQN 8×1, 1M, update-count + n-step fixes

Status: complete  
Started: 2026-10-29

## Question

Does DQN learn the 1-1 policy when the update-to-data ratio is fixed and
multi-step returns are enabled, while keeping the same lean inputs, starts,
reward, and 1M budget as the fair PPO comparison?

## Changes from the failed 8×8 DQN

```text
topology:       8×1 (8 processes × 1 agent)
gradient_steps: 4 instead of 1
n_steps:        3-step returns instead of 1-step
buffer_size:    200k instead of 100k
learning_starts: 10k instead of 2k
exploration:    50% of training instead of 25%
final epsilon:  0.02 instead of 0.05
```

At 1M timesteps:

```text
8×1 / train_freq=4 / gradient_steps=4
≈ 125,000 gradient updates
```

The failed 8×8 run had only ~3,900 updates.

## Setup

```text
observation:  lean 188 -> 954, same PPO preprocessor
model:        954 -> 512 -> 512 -> 2, ReLU
environments: 8 processes × 1 agent
timesteps:    1,000,000
starts:       validated 12 train / 8 holdout
reward:       current 5/s time, continuous under-gold,
              -100 death, -100 timeout
```

## Output

```text
runs/dqn_clean_start_1_1_8x1_1m/
```

Evaluate with:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0012_dqn_clean_start_1_1_8x1_1m \
  --model runs/dqn_clean_start_1_1_8x1_1m/dqn/dqn_final.zip
```

## Result

Complete at 1M timesteps (~123k gradient updates). Deterministic evaluation:

```text
canonical:    1/1, 1.8667 s
train pool:   12/12, median 3.2833 s
holdout pool: 8/8, median 3.8583 s
```

This was the fastest deterministic 1-1 result at the time and beat the 600k PPO
baseline on canonical time. The 200k replay buffer was saved for a continuation:

```text
runs/dqn_clean_start_1_1_8x1_1m/dqn/replay_buffer_200k.pkl
```

The buffer contains 41 canonical **1.6333 s** episodes from 5,716 canonical
collection episodes. Experiment 0013 used it for a continuation and selected
the **1.85M** checkpoint at **1.6167 s** canonical, so this 1M model is now the
source checkpoint rather than the final selected policy.
