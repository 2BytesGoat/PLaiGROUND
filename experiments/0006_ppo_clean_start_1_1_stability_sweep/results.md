# Results — 0006

Status: complete; `8x1_n600_b480` selected for 100k confirmation  
Completed: 2026-10-28

## Question

Which rollout/batch shape is most stable when continuing the selected 500k
clean-start checkpoint for 50,000 timesteps?

## Deterministic evaluation

All variants resumed the same 500k checkpoint and ran 50k new timesteps. Each
550k model was evaluated on canonical, train, and holdout starts.

| Variant | Envs | `n_steps` | Batch | Rollout | Canonical | Train pool | Holdout pool | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `8x1_n150_b120` | 8 | 150 | 120 | 1,200 | 4.9167 s | 8/12, median 2.9500 s | 5/8, median 5.2333 s | 580 |
| `8x1_n300_b240` | 8 | 300 | 240 | 2,400 | **2.1500 s** | **12/12, median 3.3000 s** | **8/8, median 3.8583 s** | 577 |
| `8x1_n600_b480` | 8 | 600 | 480 | 4,800 | **2.1500 s** | 12/12, median 3.3250 s | **8/8, median 3.8583 s** | 649 |
| `16x1_n150_b240` | 16 | 150 | 240 | 2,400 | timeout | 6/12, median 2.3917 s | 5/8, median 5.2333 s | 640 |
| `16x1_n300_b480` | 16 | 300 | 480 | 4,800 | **2.1500 s** | 12/12, median 3.4500 s | 8/8, median 3.9667 s | **820** |

All pool spawns were applied; every listed pool run had `spawn_ok` equal to the
number of starts.

## Training-monitor signal

Canonical training episodes, not deterministic evaluation:

| Variant | ≤2.2 s | Timeouts |
|---|---:|---:|
| `8x1_n150_b120` | 78.8% | 1.3% |
| `8x1_n300_b240` | 94.3% | 0.3% |
| `8x1_n600_b480` | **97.9%** | **0.1%** |
| `16x1_n150_b240` | 69.4% | 25.7% |
| `16x1_n300_b480` | 96.4% | 0.0% |

## Findings

1. **The 500k checkpoint is brittle under the old config.** The
   `8x1_n150_b120` control degraded to a 4.9167 s canonical finish, 8/12 train,
   and 5/8 holdout after only 50k more steps.
2. **A larger rollout fixes the training instability.** At 8×1, increasing
   `n_steps`/batch improved canonical training consistency from 78.8% to 94.3%
   to 97.9% at ≤2.2 s, with timeout rate falling from 1.3% to 0.1%.
3. **More processes are not automatically better.** `16x1_n150_b240` was the
   worst variant: canonical timeout, 6/12 train, and 5/8 holdout. Increasing
   its rollout to 4,800 decisions fixed canonical and pool finishes, but its
   deterministic train/holdout medians were still worse than 8×1.
4. **Throughput and stability trade differently.** `16x1_n300_b480` was the
   fastest (~820 fps), but it was not the best deterministic policy. The
   8×1/600/480 run was ~649 fps and had the best combined stability/efficiency
   result.

## Selection

Selected for the 100k confirmation:

```text
8 environments × 1 agent
n_steps = 600
batch_size = 480
rollout = 4,800 decisions
```

It matches the best canonical and holdout results, is within noise of the best
train median, and has the strongest training-monitor stability. The 16×1/300
variant remains the throughput reference, not the selected policy.

The 100k confirmation resumed the same 500k checkpoint and ran to 600k. It did
**not** continue from the sweep's 550k checkpoint, so the comparison stayed
anchored to the selected 500k baseline. The 600k model stayed stable and is now
the selected policy; see
[`../0007_ppo_clean_start_1_1_stability_100k/results.md`](../0007_ppo_clean_start_1_1_stability_100k/results.md).
