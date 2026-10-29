# 0031 — Protected replay DDQN 1-1/1-2/1-3

Status: complete

Started: 2026-10-29

Completed: 2026-10-29

## Question

Can a fixed 1-1/1-2 anchor replay prevent the canonical 1-2 regression while
keeping the 1-3 gains?

## Anchor

```text
source model:  ddqn_2150000_steps.zip
levels:        1-1 + 1-2
starts:        existing 12 + 31 train starts (canonical + longest clean routes)
transitions:   200,000
epsilon:       0.02
buffer:        RareReplayBuffer with current-state buckets
priority:      positive (return-to-go - Q_old) gap above 0.90 quantile,
               combined with rare-state frequency
```

## Training

```text
levels:          4 × 1-3 + 2 × 1-2 + 2 × 1-1
online buffer:   400k transitions
anchor buffer:   200k transitions
sampling:        75% online / 25% anchor
anchor sampling: inverse state-bucket frequency × positive Q-gap weight
source:          2.15M selected model
budget:          300k
```

## Comparison protocol

This is a controlled A/B against the selected 0025 checkpoint:

```text
treatment:  0031, 300k continuation from 2.15M
control:    0025 ddqn_2300000_steps.zip, re-evaluated with 0032-0034
held fixed: game commit, pools, 4+2+2 mix, checkpoint cadence, evaluator
changed:    protected state/Q-gap replay only
```

The anchor takes 25% of each training batch, so the online 1-3 share falls from
50% to 37.5%. If 1-3 regresses, the first follow-up is a lower anchor fraction
or a 1-3 anchor partition; do not increase the old-level anchor blindly.

## Selection

Evaluate at least every 25k checkpoint on:

```text
1-1 canonical + train + holdout
1-2 canonical + train + holdout
1-3 canonical + train + holdout
```

Select the Pareto checkpoint; do not automatically select the final one.

## Success criteria

```text
1-3 train/holdout remain fully covered
1-1 train/holdout remain fully covered
1-2 canonical recovers toward 2.9667 s
1-2 train/holdout medians stay near the 2.15M reference
```

## Run

```bash
caffeinate -dimsu .venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0031_ddqn_protected_1_1_1_2_1_3
```

## Result

300k additional steps completed in about 7 minutes at ~720 fps after caching the
fixed anchor sampling distribution. The final 2.45M checkpoint was not selected.
The selected checkpoint is **2.225M**:

```text
runs/ddqn_protected_1_1_1_2_1_3/dqn/checkpoints/ddqn_2225000_steps.zip
```

| Level | Canonical | Train | Holdout |
|---|---:|---:|---:|
| 1-1 | 1.6167 s | 12/12, 3.2667 s | 8/8, 3.7917 s |
| 1-2 | **2.9667 s** | 31/31, 4.3500 s | 20/20, 3.1333 s |
| 1-3 | 3.2500 s | 45/45, 5.2667 s | 23/23, 5.3333 s |

Control, old 0025 2.30M:

```text
1-1: 1.6167 / 12/12 3.2750 / 8/8 3.8000
1-2: 3.8667 / 31/31 4.0000 / 20/20 3.1667
1-3: 3.1000 / 45/45 5.2000 / 23/23 5.4500
```

The protected replay fixed the 1-2 canonical regression: **3.8667 → 2.9667 s**,
exactly the 2.15M baseline, with full coverage. 1-1 canonical is unchanged and
its train/holdout medians are marginally faster. 1-3 stays fully covered; it is
**0.15 s slower canonically** and 0.07 s slower on train median, with a 0.12 s
faster holdout median. This is a retention win, not a 1-3 speed win.

The 25% anchor leaves 37.5% of samples from 1-3 instead of 50%. If the next goal
is faster 1-3 rather than regression recovery, try a 10–15% anchor fraction or a
1-3 high-gap anchor partition.

Evaluation configs: 0032 (1-3), 0033 (1-1), 0034 (1-2). The old 2.30M control
was re-evaluated under the same current game state and reproduced its recorded
numbers exactly.
