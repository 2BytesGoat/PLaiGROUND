# Results — 0007

Status: complete; 600k selected  
Completed: 2026-10-28

## Outcome

The 100k continuation with the winning **8×1 / `n_steps=600` / `batch_size=480`**
config completed at **600,800 timesteps**. It fixes the stability regression
without lowering the deterministic canonical time.

## Deterministic comparison

| Model | Canonical | Train pool | Holdout pool |
|---|---:|---:|---:|
| 500k baseline | 1/1, 2.1500 s | 12/12, median 3.3333 s | 8/8, median 3.8583 s |
| **600k selected** | **1/1, 2.1500 s** | **12/12, median 3.3250 s** | **8/8, median 3.8583 s** |
| 1M continuation | 1/1, 2.1500 s | 12/12, median 3.4167 s | 8/8, median 4.2750 s |

All 600k pool runs finished and all spawns were applied:

```text
train:   12/12 finishes, spawn_ok 12/12
holdout:  8/8 finishes, spawn_ok 8/8
```

The 600k model is selected because it matches the best canonical and holdout
results, slightly improves the 500k train median, and is far more stable in
training. The improvement is a robustness win, not a new fastest route.

## Stability

Training-monitor canonical episodes over the 100k continuation:

```text
episodes:       2,653
median:         2.150 s
best:           2.133 s
≤2.2 s:         97.7%
timeouts:       0.04%
episodes <1.9 s: 0
episodes <1.75 s: 0
```

Per-25k bins stayed healthy throughout:

```text
500–525k: 98.3% ≤2.2 s, 0.2% timeouts
525–550k: 97.3% ≤2.2 s, 0.0% timeouts
550–575k: 96.7% ≤2.2 s, 0.0% timeouts
575–600k: 98.5% ≤2.2 s, 0.0% timeouts
```

This is the opposite of the old 8×1/150/120 control, which degraded to a
4.9167 s canonical finish after 50k.

## Shortest-path plateau

The 600k run confirms that the stability fix does not change the speed
objective. Across the earlier 0–1M training history:

- only 17 / 19,853 canonical training episodes were under 1.75 s;
- the fastest observed episode was 1.6167 s around 80k–100k;
- after 300k, there were no episodes below 1.9 s;
- fast episodes had lower mean reward (~522.5) than the 2.15 s mode (~524.0).

PPO is converging to the higher-return stable route, not the shortest clock
route. The 600k model is the stable endpoint of that process, not a 1.75 s
policy.

## Decision

Select `runs/ppo_clean_start_1_1_best_100k/ppo_pilot_600k.zip` as the stable
1-1 policy. The 500k checkpoint remains the shorter-horizon baseline, and the
1M continuation remains a diagnostic of the late-training instability.

A future shortest-path attempt needs a different objective or a fast-route
warm start; more PPO at the current reward is not expected to break 2.13 s.
