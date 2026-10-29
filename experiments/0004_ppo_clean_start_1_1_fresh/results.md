# Results — 0004

Status: complete; 500k selected  
Completed: 2026-10-28

## Outcome

The fresh 500k run produced a usable policy. Selected checkpoint:

```text
runs/ppo_clean_start_1_1_fresh/checkpoints/ppo_500000_steps.zip
```

## Checkpoint comparison

| Checkpoint | Canonical | Train pool | Holdout pool |
|---|---:|---:|---:|
| 25k | 0/1 timeout | 6/12, median 2.39 s | 5/8, median 5.23 s |
| 400k | **1/1, 2.15 s** | 12/12, median 3.45 s | 8/8, median 3.97 s |
| 475k | 0/1 timeout | 6/12, median 2.39 s | 5/8, median 5.23 s |
| 500k | **1/1, 2.15 s** | **12/12, median 3.33 s** | **8/8, median 3.86 s** |

All evaluated pool runs reported `spawn_ok`:

```text
train:   12/12
holdout:  8/8
```

## Selection

The 400k and 500k checkpoints both clear canonical at **2.15 s**. The 500k
checkpoint has the better train and holdout medians, so it is selected:

```text
canonical:    1/1, 2.1500 s
train:       12/12, median 3.3333 s, best 2.1500 s
holdout:      8/8, median 3.8583 s, best 3.4167 s
```

The 475k checkpoint is a real regression mode: canonical timeout, 6/12 train,
5/8 holdout. This confirms the policy still has brittle regions, but the 500k
checkpoint recovered and is a complete result.

## Final training stats

```text
timesteps:          500,400
ep_rew_mean:        521
ep_len_mean:        27.8
explained_variance: 0.996
value_loss:         4.35
approx_kl:          0.037
entropy_loss:      -0.050
```

## Decision

Keep `ppo_500000_steps.zip` as the fresh clean-start candidate. It is not
installed in the game. The 475k regression should be considered in any future
training/selection work.
