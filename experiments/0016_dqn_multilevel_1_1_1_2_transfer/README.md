# 0016 — DQN multi-level 1-1 → 1-2 transfer

Status: pilot complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

Can the selected 1-1 DQN learn 1-2 while retaining its 1-1 policy? This is a
continual-learning fine-tune, not a 1-2-only run.

## Baseline

The 1.85M 1-1 DQN was evaluated zero-shot on the 1-2 diverse start pool:

```text
canonical:    9.1167 s
train pool:   17/31 finishes, median 5.5000 s, best 3.1833 s
holdout pool: 15/20 finishes, median 5.9500 s, best 3.1833 s
```

It could finish 1-2 but was slow and missed many starts.

## Setup

```text
levels:            1-1, 1-2
processes:         8 total: 4 × 1-1 + 4 × 1-2
source model:      runs/dqn_clean_start_1_1_8x1_buffer_finetune/dqn/dqn_final.zip
preprocessing:     runs/ppo_clean_start_1_1_fresh/preprocessing.pkl
budget:            100,000 additional timesteps
final timesteps:   1,950,000
n_steps / grads:   3 / 4
buffer:            200,000
epsilon:           0.02
train starts:      12 1-1 entries + 31 1-2 entries
```

The 4/4 mix is the retention mechanism: every rollout batch contains both
levels, so 1-1 transitions stay in the replay buffer.

## Training monitor

```text
1-1:  2,163 episodes, 2,160 finishes, 3 timeouts
      canonical median 1.6167 s, state median 3.3333 s

1-2:  718 episodes, 631 finishes, 87 timeouts
      canonical median 3.7833 s, state median 4.9167 s
      best 2.9667 s
```

## Retention gate

```text
1-1 canonical: 1.6167 s  (limit 1.6500)  PASS
1-1 train:     3.2500 s  (limit 3.3500)  PASS
1-1 holdout:   3.8000 s  (limit 3.8900)  PASS
```

The 1-1 policy is unchanged at the measured precision: **1.6167 s canonical,
12/12 train, 8/8 holdout**.

## 1-2 result

Deterministic evaluation:

```text
canonical:    3.7500 s   (was 9.1167 s)
train pool:   31/31 finishes, median 4.3833 s, best 3.0500 s
holdout pool: 20/20 finishes, median 4.0500 s, best 3.0500 s
```

This fixes the zero-shot coverage completely: all 31 train and all 20 holdout
starts finish, with no timeouts or deaths in deterministic evaluation. The
canonical time is still slower than the old NEAT transfer's 3.3667 s best, but
the model now completes every evaluated start and reaches 3.0500 s from states.

## Output

```text
runs/dqn_multilevel_1_1_1_2/
  dqn/checkpoints/
  dqn/dqn_final.zip
  dqn/monitor.csv

runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/multilevel_pilot_1_1/summary.json
runs/dqn_zero_shot_1_2/eval/multilevel_pilot_1_2/summary.json
```

## Next decision

The pilot passed. The next choice is whether to:

1. continue the 4/4 mix for another 400k to target <3.0 s on 1-2,
2. switch to the 4+2+2 rehearsal schedule and add 1-3, or
3. promote this as the first level-agnostic 1-1/1-2 candidate and keep the
   selected 1-1 checkpoint as the retention reference.

Not promoted yet; the selected policy remains the 1-1 model.
