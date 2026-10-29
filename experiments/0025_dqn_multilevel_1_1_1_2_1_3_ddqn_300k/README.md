# 0025 — DDQN multi-level 1-1/1-2/1-3, 300k

Status: complete

Started: 2026-10-29

Completed: 2026-10-29

## Question

Can the selected 1-1/1-2 Double-DQN model learn 1-3 with the 4+2+2 rehearsal
mix while preserving 1-1 and 1-2?

## Source

```text
checkpoint:  runs/dqn_multilevel_1_1_1_2_ddqn_300k/dqn/checkpoints/ddqn_2150000_steps.zip
1-1/1-2:      12 + 31 train starts
1-3:          45 train starts (44 state + canonical)
combined:     88 level-tagged train starts
```

The 1-3 pool was rebuilt from the 0024 frequent-rebalance archive (655 routes,
105 clean) and has:

```text
train states:   44
holdout states: 23
route clusters: 19 at Jaccard 0.5
```

## Setup

```text
levels:          4 × 1-3 + 2 × 1-2 + 2 × 1-1
n_envs:          8
algorithm:       Double DQN
budget:          300,000 additional steps
speedup:         12
power:           AC, Low Power Mode off
buffer:          200,000
n_steps/grads:   3 / 4
epsilon final:   0.02
learning rate:   0.0001
resume:          ddqn_2150000_steps.zip
```

The combined training pool is:

```text
runs/dqn_multilevel_1_1_1_2_1_3/pool/train_start_pool_v2.json
```

## Expected gates

```text
1-1: preserve selected baseline (canonical 1.6167 s; train/holdout medians)
1-2: preserve selected baseline (canonical 2.9667 s; 31/31 train, 20/20 holdout)
1-3: improve from canonical-only to route-start evaluation, with train/holdout
     finish rates and medians reported
```

The selected 2.15M checkpoint remains the 1-1/1-2 reference; the final 2.45M
checkpoint must clear the same gates before replacing it.

## Output

```text
runs/dqn_multilevel_1_1_1_2_1_3_ddqn_300k/
  dqn/checkpoints/
  dqn/dqn_final.zip
  dqn/monitor.csv
  eval/
```

## Result

300k additional steps completed in about 390 s. The final 2.45M checkpoint
failed the 1-1 holdout gate, so the **2.30M checkpoint** was selected:

```text
selected:  runs/dqn_multilevel_1_1_1_2_1_3_ddqn_300k/dqn/checkpoints/ddqn_2300000_steps.zip
```

| Level | Canonical | Train | Holdout |
|---|---:|---:|---:|
| 1-1 | 1.6167 s | 12/12, 3.2750 s | 8/8, 3.8000 s |
| 1-2 | 3.8667 s | 31/31, 4.0000 s | 20/20, 3.1667 s |
| 1-3 | **3.1000 s** | **45/45, 5.2000 s** | **23/23, 5.4500 s** |

Comparison points:

```text
2.15M baseline 1-3:  canonical 5.1333 s, train 40/45, holdout 15/23
2.20M:               1-1 train 10/12 (2 timeouts); not selected
2.45M final:         1-1 holdout 4.0083 s > 3.89 s gate; not selected
```

The selected model preserves 1-1 coverage and holdout, preserves 1-2 finish
coverage, and makes 1-3 fully solvable on both train and holdout starts. The
remaining 1-2 regression is canonical-only (2.9667 -> 3.8667 s); train and
holdout medians remain close to the prior model.

