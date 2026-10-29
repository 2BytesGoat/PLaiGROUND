# Results — 0010

Status: complete; no speed gain, 600k retained  
Completed: 2026-10-29

## Outcome

The 8×8 continuous-reward run trained successfully but did not retain the fast
route. The 600k stable model remains selected.

## Deterministic evaluation

| Model | Canonical | Train pool | Holdout pool |
|---|---:|---:|---:|
| 600k stable selected | **2.1500 s** | **3.3250 s** | **3.8583 s** |
| 8×1 continuous 325k | 2.1500 s | 3.5583 s | 4.9500 s |
| 8×8 continuous 330k | 2.1500 s | 3.4167 s | 4.2750 s |
| DQN 200k | timeout | 8/12, 3.3333 s | 5/8, 5.2333 s |

All 8×8 pool runs finished and all spawns were applied.

## Training monitor

```text
episodes:          3,617
median:            2.283 s
best:              1.833 s
≤2.2 s:            31.3%
timeouts:          0.11%
episodes <1.75:    0
```

The best sample appeared in the first 25k-bin; it never improved. The median
drifted from 2.217 s to 2.283 s.

## Why 8×8 did not help enough

```text
8×8 rollout:  64 × 150 = 9,600
100k / 9,600 ≈ 10 PPO updates
```

The larger batch reduced update count relative to 8×1/600/480. A single 1.833 s
sample could not be reinforced in ~10 updates, especially with entropy 0.02.

## Decision

Keep the 600k stable model selected. The 8×8 topology and continuous reward are
implemented and validated, but another 8×8 run should use either a longer
budget or a smaller rollout / higher entropy to get more updates and
exploration.
