# Results — 0005

Status: complete; 1M evaluated, 500k retained  
Completed: 2026-10-28

## Outcome

The continuation reached **1,000,400 timesteps**. The final 1M model is
deterministic-clean on canonical but did not improve the start-pool medians over
the 500k baseline.

## Comparison

| Model | Canonical | Train pool | Holdout pool |
|---|---:|---:|---:|
| 500k | **1/1, 2.1500 s** | **12/12, median 3.3333 s** | **8/8, median 3.8583 s** |
| 1M | 1/1, 2.1500 s | 12/12, median 3.4167 s | 8/8, median 4.2750 s |

All 1M pool runs finished and all spawns were applied:

```text
train:   12/12 finishes, spawn_ok 12/12
holdout:  8/8 finishes, spawn_ok 8/8
```

## Training-monitor trend

Canonical training time is a stochastic episode metric, not the deterministic
checkpoint evaluation.

| Training steps | Median | ≤2.2 s | Timeouts |
|---:|---:|---:|---:|
| 0–100k | 4.4–6.3 s | 4–7% | 11–28% |
| 200–500k | 2.150 s | 52–88% | 0.3–9.6% |
| 850–950k | 2.133 s | 93.5–96.2% | 0.3–1.6% |
| 950k–1M | 2.133 s | 62.6% | 21.0% |
| 990k–1M | 2.133 s | 88.7% | 5.4% |

The strongest training window was around **850k–950k**. A late collapse at
**970k–980k** produced 100% canonical timeouts in that bin, followed by a
recovery in the final 10k.

## Decision

Keep `runs/ppo_clean_start_1_1_fresh/checkpoints/ppo_500000_steps.zip` as the
selected 1-1 policy. The 1M checkpoint is preserved for diagnostics, but its
train and holdout medians are worse.

The next experiment is a controlled rollout/batch stability sweep from the
500k checkpoint; see
[`../0006_ppo_clean_start_1_1_stability_sweep/`](../0006_ppo_clean_start_1_1_stability_sweep/).
