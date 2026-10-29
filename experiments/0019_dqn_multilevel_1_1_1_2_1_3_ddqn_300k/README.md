# 0019 — DDQN multi-level 1-1/1-2/1-3, 300k

Status: stopped (canonical-only fallback)  
Started: 2026-10-29  
Stopped: 2026-10-29 at 2.274M timesteps

## Question

Can the selected 1-1/1-2 Double-DQN model learn 1-3 using the 4+2+2 rehearsal
mix while preserving 1-1 and 1-2?

## Setup

```text
source:          ddqn_2150000_steps.zip (selected 1-1/1-2)
levels:          4 × 1-3 + 2 × 1-2 + 2 × 1-1
budget:          300,000 additional steps
algorithm:       Double DQN
n_steps/grads:   3 / 4
buffer:          200,000
epsilon:         0.02
learning rate:   0.0001
1-3 starts:      canonical only
```

Both the 1-1 and evolved 1-2 NEAT populations produced **0 finishers** on 1-3,
so there is no 1-3 start pool yet. The first 1-3 phase starts from canonical and
keeps the existing 1-1/1-2 state pools as anchors.

## Baseline

```text
1-1: 1.6167 s canonical; 12/12 train; 8/8 holdout
1-2: 2.9667 s canonical; 31/31 train; 20/20 holdout
1-3: no NEAT finishers yet (canonical-only start)
```

## Output

```text
runs/dqn_multilevel_1_1_1_2_1_3/
```

## Result

Pending.
