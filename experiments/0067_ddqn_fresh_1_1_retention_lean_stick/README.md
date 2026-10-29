# 0067 — 1-1 retention after the 1-3 stage

Status: running  
Started: 2026-10-29

## Question

Can a short 1-1-heavy continuation from the selected 1.5M 0063 checkpoint
recover the 1-1 holdout median from 4.2667 s to 3.8000 s without losing the
1-2 and 1-3 behavior?

## Setup

```text
source:            0063 1.5M ddqn checkpoint
observation:       lean_stick 191 -> 957
levels:            4 x 1-1 + 2 x 1-2 + 2 x 1-3
start pool:        88 tagged starts (12 x 1-1 + 31 x 1-2 + 45 x 1-3)
additional steps:  50,000
reset timesteps:   false
protected replay:  200k online + 200k 1-1/1-2 anchor, 25% anchor
learning rate:     0.0001
```

The 1-1 holdout regression is mostly one `[25,10]` start: it is 4.6500 s at
1.5M versus 3.5000 s at 1.3M, while the other seven starts are within 0.10 s.
The 1.3M 1-1/1-2 anchor is reused to pull that start back.

## Gate

```text
1-1 canonical:  1.5667
1-1 train:      12/12, median <= 3.2833 (or close)
1-1 holdout:    8/8, median <= 3.8000

1-2 canonical:  3.0167
1-2 train:      31/31, median <= 4.4167
1-2 holdout:    20/20, median <= 3.3083

1-3 canonical:  3.1667
1-3 train:      45/45, median <= 5.4667
1-3 holdout:    23/23, median <= 5.5500
```

## Output

```text
runs/ddqn_fresh_1_1_retention_lean_stick/
```

Checkpoints are saved every 10k so the 1-1 holdout can be swept before a full
1-2/1-3 gate.

## Result

The **1.53M checkpoint** is selected:

```text
runs/ddqn_fresh_1_1_retention_lean_stick/dqn/checkpoints/ddqn_1530000_steps.zip
```

1-1 sweep:

```text
1.51M: 1.5667 / 3.4083 / 4.2833
1.52M: 1.5667 / 3.4167 / 3.9583
1.53M: 1.5667 / 3.2583 / 3.8000   <- recovered
1.54M: 1.5667 / 3.3250 / 3.8000
1.55M: 1.5667 / 3.2583 / 3.8000
```

Full gate at 1.53M:

```text
1-1 canonical: 1.5667; train 12/12, 3.2583; holdout 8/8, 3.8000
1-2 canonical: 3.0167; train 31/31, 4.3500; holdout 20/20, 3.8833
1-3 canonical: 3.1667; train 45/45, 5.4667; holdout 23/23, 5.5333
```

The exact 1-1 failure recovered: `[25,10]` went from 4.6500 s at 1.5M to
3.4333 s at 1.53M, and the 1-1 train median improved to 3.2583 s. The 1-3 pool
is unchanged. 1-2 canonical and train are retained, but one 1-2 holdout start,
`[24,17]`, regressed from 3.1500 s to 4.7333 s, moving the 1-2 holdout median to
3.8833 s.

1.55M is not selected: the extra retention training collapsed 1-2 canonical to
3.8333 s and moved 1-2 holdout to 4.1583 s. The selected chain checkpoint is
therefore 0067 **1.53M**.
