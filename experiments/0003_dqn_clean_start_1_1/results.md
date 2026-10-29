# Results — 0003

Status: complete; negative at 1M  
Completed: 2026-10-29

## Outcome

The fair lean-input DQN (188 → 954, same preprocessing/reward/starts as PPO)
trained for 1M timesteps with 8 processes × 8 agents. It still does not learn
the canonical finish.

## Deterministic evaluation

```text
canonical:    timeout (0/1)
train pool:   8/12 finishes, median 2.9750 s, best 2.2333 s
holdout pool: 7/8 finishes, median 5.2333 s, best 3.4167 s
```

All pool spawns were applied.

## Training

```text
timesteps:    1,000,000
environments: 64 (8 × 8)
fps:          ~1,700
n_updates:    ~3,900
final ep_rew: ~330
final ep_len: ~54
```

The 8×1 lean run had ~31k gradient updates at the same experience scale. The
64-env run collected samples much faster but updated the Q-network ~8× less
often, so it underfit despite the larger replay stream.

## Decision

Do not promote DQN. The 600k PPO model remains selected.

If DQN is tried again, the next changes should target update count rather than
more environments:

- 8×1 or 16×1 topology
- smaller `train_freq` / more gradient steps
- prioritized replay or n-step returns
- lower epsilon fine-tuning after exploration
