# Results — 0001

Status: complete
Completed: 2026-10-28

## Selected checkpoint

The 225k checkpoint was selected by deterministic canonical evaluation:

```text
runs/ppo_clean_start_1_1/best_model.zip
```

The failed original 1M run is preserved at:

```text
runs/ppo_clean_start_1_1_failed_1m/
```

## Final evaluation

Deterministic, 100 canonical agents:

```text
level:         1-1
finishes:      100 / 100
deaths:        0
timeouts:      0
best time:     2.3167 s
median time:   2.3167 s
mean reward:   524.28
efficiency:    0.8305
flow:          0.9231
```

Start pools:

```text
train pool:    14 / 14 finishes, spawn_ok 14 / 14
               median 3.5583 s, best 2.1500 s

holdout pool:  10 / 10 finishes, spawn_ok 10 / 10
               median 5.2250 s, best 3.4167 s
```

The holdout is route-cluster-level only. All 10 holdout state keys also occur on
train routes, so this is not a state-unseen generalization result.

## Training history

The original 1M run collapsed around 296k steps:

```text
value loss spike      → 11,116
entropy loss          → ~0
canonical finish rate → 0
```

The failed artifacts remain in `runs/ppo_clean_start_1_1_failed_1m/`.

The continuation resumed from `ppo-200000.zip` with:

```text
learning rate: 1e-4
target KL:     0.02
entropy coef:  0.02
checkpoint:    every 25k
evaluation:    every 25k
patience:      3 evaluations
```

It stopped at 300k after three evaluations without improvement. The selected
225k checkpoint was healthy:

```text
200k baseline: 2.7667 s
225k:          2.3167 s  ← selected
250k:          2.3167 s
275k:          2.3167 s
300k:          2.9000 s
```

## Decision

Keep the 225k policy as the clean-start PPO baseline candidate. It is not
installed in the game. Compact deployment/distillation options are recorded in
[`../../docs/experiments/compact-policy-deployment.md`](../../docs/experiments/compact-policy-deployment.md).
