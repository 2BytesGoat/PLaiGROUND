# 0060 — fresh lean_stick 191, stage 2 (1-2)

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

Can the selected fresh 191 1-1 policy learn 1-2 while retaining 1-1, using a
1-1 replay buffer collected from the 1.5667 model?

## Setup

```text
source:            0058 1M dqn_final.zip
1-1 buffer:        200k transitions collected from 0058
observation:       lean_stick 191 -> 957
model:             512 x 512 DDQN
levels:            4 x 1-2 + 4 x 1-1
starts:            43 tagged starts (12 x 1-1 + 31 x 1-2)
additional steps:  300,000
reset timesteps:   false
```

The 1-1 buffer is collected with the training environment, so it includes the
same `reassert_held_jump=true` behavior that produced the 1.5667 policy.

## Gate

```text
1-2 canonical:  near 2.9667 s
1-2 train:      31/31
1-2 holdout:    20/20
1-1 canonical:  1.5667 s
1-1 train:      12/12
1-1 holdout:    8/8
```

Do not start 1-3 until these are reviewed.

## Output

```text
runs/ddqn_fresh_1_1_1_2_lean_stick/
```

## Result

The **1.3M checkpoint** is selected:

```text
runs/ddqn_fresh_1_1_1_2_lean_stick/dqn/checkpoints/ddqn_1300000_steps.zip
```

```text
1-2 canonical:  3.0333 s
1-2 train:      31/31, median 4.4333 s, best 3.0333 s
1-2 holdout:    20/20, median 3.2333 s, best 3.1000 s
1-1 canonical:  1.5667 s
1-1 train:      12/12, median 3.2833 s
1-1 holdout:    8/8, median 3.8000 s
```

The post-training `dqn_final.zip` is slightly worse on 1-2 holdout (median
4.0000 s), so use the 1.3M checkpoint.

Checkpoint sweep on 1-2:

```text
1.1M: 15/31 train, 12/20 holdout; medians 3.18 / 3.38
1.2M: 30/31 train, 19/20 holdout; medians 4.38 / 4.32
1.3M: 31/31 train, 20/20 holdout; medians 4.43 / 3.23
```

The remaining weakness is 1-2 **state-start speed**, not coverage. Training
monitor state median is 4.32 s, so the policy is genuinely slower there; it is
not an eval-semantics artifact. More 1-2 state-focused training is the next
lever if this needs tightening.
