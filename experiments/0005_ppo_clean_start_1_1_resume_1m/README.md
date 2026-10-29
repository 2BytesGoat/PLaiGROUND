# 0005 — 1-1 PPO clean-start continuation to 1M

Status: complete; 1M evaluated, 500k retained
Started: 2026-10-28
Completed: 2026-10-28

## Question

Does the fresh 1-1 PPO policy keep improving past the selected 500k checkpoint
when training continues to 1M with the same data, observation pipeline, and
PPO settings?

## Setup

This is a continuation, not a new model family:

```text
level:            1-1
observation:      lean 188 -> Python one-hot 954
model:            954 -> 512 -> 512 -> 2, Tanh
environments:      8
training starts:  canonical 0.70 / state 0.30
target:           1,000,000 total timesteps
resume:           runs/ppo_clean_start_1_1_fresh/checkpoints/ppo_500000_steps.zip
preprocessor:     runs/ppo_clean_start_1_1_fresh/preprocessing.pkl
PPO:              lr=1e-4, target_kl=0.02, ent_coef=0.02
checkpoints:      every 25,000
```

No new preprocessor is fitted and no start pool is rebuilt. The 500k run in
`experiments/0004_ppo_clean_start_1_1_fresh/` remains the completed baseline.

## Output

```text
runs/ppo_clean_start_1_1_resume_1m/
```

Train with:

```bash
.venv/bin/python scripts/reinforcement/01_ppo.py \
  --config experiments/0005_ppo_clean_start_1_1_resume_1m
```

Evaluate with:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0005_ppo_clean_start_1_1_resume_1m \
  --model runs/ppo_clean_start_1_1_resume_1m/ppo_pilot_1m.zip
```

## Result

Training completed at **1,000,400 timesteps**. The final 1M model was evaluated
deterministically:

```text
canonical:    1/1, 2.1500 s
train pool:   12/12, median 3.4167 s, best 2.1500 s
holdout pool: 8/8, median 4.2750 s, best 3.3833 s
```

The 1M continuation is slightly worse than the 500k checkpoint on both pools
(500k: train 3.3333 s, holdout 3.8583 s). The 500k checkpoint remains selected.

Training-monitor detail: canonical time reached a strong stretch around
850k–950k (2.133 s median, mostly <1% timeouts), then collapsed at 970k–980k
and recovered near 990k. The final 1M deterministic result was healthy but did
not improve the pool medians.

See [`results.md`](results.md) for the comparison.
