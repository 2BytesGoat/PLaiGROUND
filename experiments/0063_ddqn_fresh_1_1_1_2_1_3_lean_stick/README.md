# 0063 — fresh lean_stick 191, stage 3 (1-3)

Status: running  
Started: 2026-10-29

## Question

Can the selected fresh 191 1-1/1-2 policy learn 1-3 while retaining the fresh
1-1 and 1-2 behavior, using a fresh 1-1/1-2 anchor from the 1.3M model?

## Setup

```text
source:            0060 1.3M dqn checkpoint
1-1/1-2 anchor:    200k transitions collected from 0060
observation:       lean_stick 191 -> 957
model:             512 x 512 DDQN
levels:            4 x 1-3 + 2 x 1-2 + 2 x 1-1
starts:            88 tagged starts (12 x 1-1 + 31 x 1-2 + 45 x 1-3)
additional steps:  300,000
reset timesteps:   false
protected replay:  400k online + 200k anchor, 25% anchor sampling
```

The 1-3 route pool and holdout are the validated
`runs/neat_frequent_rebalance_1_3/` artifacts. The 1-1/1-2 anchor is collected
from the selected 0060 model so the 1-2 stage cannot silently regress while the
1-3 level is learned.

## Gate

```text
1-3 canonical:  near 3.25 s (0031 protected reference)
1-3 train:      45/45
1-3 holdout:    23/23
1-1 canonical:  1.5667 s
1-1 train:      12/12
1-1 holdout:    8/8
1-2 canonical:  3.0333 s
1-2 train:      31/31
1-2 holdout:    20/20
```

Do not promote a checkpoint that loses 1-1 or 1-2 coverage.

## Output

```text
runs/ddqn_fresh_1_1_1_2_1_3_lean_stick/
```

## Reference

The old 188/954 0031 protected run selected 2.225M:

```text
1-3 canonical: 3.2500 s
1-3 train:     45/45, median 5.2667 s
1-3 holdout:   23/23, median 5.3333 s
```

The 1-3 clean-route archive has a 4.7667 s best clean route and 105 clean
routes; full coverage is the priority for this stage.
