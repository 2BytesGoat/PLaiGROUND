# 0007 — 100k stability confirmation with 8×1/600/480

Status: complete; 600k selected  
Started: 2026-10-28
Completed: 2026-10-28

## Question

Does the winning 4,800-decision rollout shape remain stable for 100,000
timesteps from the selected 500k checkpoint?

## Setup

```text
level:            1-1
observation:      lean 188 -> one-hot 954
model:            954 -> 512 -> 512 -> 2, Tanh
environments:      8 processes × 1 agent
rollout:          n_steps=600, batch_size=480
rollout size:     4,800 decisions per update
target:           600,000 total timesteps
resume:           runs/ppo_clean_start_1_1_fresh/checkpoints/ppo_500000_steps.zip
preprocessor:     runs/ppo_clean_start_1_1_fresh/preprocessing.pkl
PPO:              lr=1e-4, target_kl=0.02, ent_coef=0.02
checkpoints:      every 25,000
```

This is the 100k confirmation of the config selected in
[`../0006_ppo_clean_start_1_1_stability_sweep/`](../0006_ppo_clean_start_1_1_stability_sweep/).
It starts from 500k rather than the sweep's 550k checkpoint so the comparison
stays anchored to the selected baseline.

## Output

```text
runs/ppo_clean_start_1_1_best_100k/
```

Train with:

```bash
.venv/bin/python scripts/reinforcement/01_ppo.py \
  --config experiments/0007_ppo_clean_start_1_1_stability_100k
```

Evaluate with:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0007_ppo_clean_start_1_1_stability_100k \
  --model runs/ppo_clean_start_1_1_best_100k/ppo_pilot_600k.zip
```

## Result

Training completed at **600,800 timesteps**. The 600k deterministic evaluation:

```text
canonical:    1/1, 2.1500 s
train pool:   12/12, median 3.3250 s
holdout pool: 8/8, median 3.8583 s
```

The stability fix worked:

```text
canonical training ≤2.2 s:  97.7%
canonical training timeouts: 0.04%
best training time:          2.133 s
```

The 600k model is selected as the stable 1-1 policy. It does not lower the
deterministic canonical time — there are **no canonical training episodes below
1.9 s** in the 100k run — but it fixes the timeout/regression mode and matches
or slightly improves the 500k train median.

See [`results.md`](results.md) for the full comparison.
