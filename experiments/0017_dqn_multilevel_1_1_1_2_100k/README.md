# 0017 — DQN multi-level 1-1/1-2, +100k

Status: running  
Started: 2026-10-29

## Question

Does another 100k of the 4×1-1 + 4×1-2 mix improve 1-2 without moving 1-1?

## Setup

```text
source:          runs/dqn_multilevel_1_1_1_2/dqn/dqn_final.zip (1.95M)
levels:          1-1, 1-2
processes:       4 × 1-1 + 4 × 1-2
budget:          100,000 additional steps (2.05M total)
n_steps/grads:   3 / 4
buffer:          200,000
epsilon:         0.02
learning rate:   0.0001
```

Start pools are the same combined 1-1 (12) + 1-2 diverse (31) pool.

## Baseline

Pilot 0016 at 1.95M:

```text
1-1 canonical: 1.6167 s
1-2 canonical: 3.7500 s
1-2 train:     31/31, median 4.3833 s, best 3.0500 s
1-2 holdout:   20/20, median 4.0500 s, best 3.0500 s
```

## Output

```text
runs/dqn_multilevel_1_1_1_2_100k/
```

## Result

Deterministic evaluation at 2.05M:

```text
1-1:
  canonical:    1.6167 s
  train:        12/12, median 3.2667 s
  holdout:      8/8, median 3.8000 s

1-2:
  canonical:    3.7833 s
  train:        31/31, median 4.0167 s, best 3.1000 s
  holdout:      20/20, median 3.3500 s, best 3.0833 s
```

No timeouts or deaths in deterministic evaluation.

Compared with the 1.95M pilot, 1-2 improved on the start pools (train median
4.3833 → 4.0167, holdout 4.0500 → 3.3500) while canonical stayed flat
(3.7500 → 3.7833). The 1-1 gate still passes; train moved from 3.2500 to
3.2667, inside the 3.3500 limit.

Evaluation summaries:

```text
runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/multilevel_100k_1_1/summary.json
runs/dqn_zero_shot_1_2/eval/multilevel_100k_1_2/summary.json
```

## Game preview

```text
canonical:   finished in 3.7833 s, 55 steps
state [6,20]: finished in 3.2167 s, 44 steps
```

## Action/Q diagnostic

Deterministic rollouts with `tools/diagnose_dqn.py`:

| Rollout | Decisions | Action flips | Flip rate | Mean Q margin | Min Q margin | Time |
|---|---:|---:|---:|---:|---:|---:|
| 1-2 canonical | 55 | 17 | 0.315 | 19.12 | 0.165 | 3.7833 s |
| 1-2 state [16,19] | 27 | 6 | 0.231 | 31.30 | 0.933 | 3.1000 s |
| 1-1 canonical | 23 | 5 | 0.227 | 38.91 | 0.435 | 1.6167 s |

The canonical 1-2 policy is **confident but jittery**: Q margins are usually
large, while short jump/no-jump runs alternate frequently. The fast 1-2 state
has a similar flip rate to the clean 1-1 policy and much higher Q margins.
This points more to a learned inefficient route/action policy than to
epsilon noise or Q-value ties.

Artifacts:

```text
runs/dqn_zero_shot_1_2/eval/diagnostics/multilevel_100k_canonical_q/
runs/dqn_zero_shot_1_2/eval/diagnostics/multilevel_100k_best_state_q/
runs/dqn_clean_start_1_1_8x1_buffer_finetune/eval/diagnostics/multilevel_100k_1_1_canonical_q/
```

## Recommendation

Double DQN is not the first fix indicated: the policy’s Q margins are mostly
large, so tie-breaking is not the main problem. The better first move is a
longer 1-2 fine-tune with higher exploration (around 0.05) and the same 4/4
mix. If canonical stays jittery after that, add an action-history input or a
very small action-change penalty. A separate 1-3 run should wait until 1-2 has
a smoother deterministic canonical route.
