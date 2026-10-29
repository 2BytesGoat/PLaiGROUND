# 0009 — Continuous under-gold time-trial from 225k

Status: complete; reward ranking fixed, route not learned  
Started: 2026-10-28
Completed: 2026-10-28

## Question

Does a continuous under-gold finish bonus make the 1.6–1.75 s canonical route
preferred over the stable 2.13–2.15 s route?

This is the direct A/B against experiment 0008. It uses the same 225k source
checkpoint, same 100k budget, same canonical-only starts, and same 8×1/600/480
topology. Only the under-gold reward changes from flat to continuous.

## Reward

```text
time coefficient:       5.0 / s
finish:                 500 base
under-gold bonus:       min(50, 25 * max(0, gold_time - run_time))
death:                  -100
timeout:                -100
```

With gold ≈ 2.15:

```text
2.13 s: +0.5
1.75 s: +10.0
1.63 s: +13.0
```

The fast route now gets a materially larger terminal bonus than the 2.13 s
route. The cap prevents extreme rewards for implausibly short runs.

## Setup

```text
level:            1-1
observation:      lean 188 -> one-hot 954
model:            954 -> 512 -> 512 -> 2, Tanh
environments:      8 processes × 1 agent
rollout:          n_steps=600, batch_size=480
starts:           canonical only
target:           325,000 total timesteps
resume:           runs/ppo_clean_start_1_1_fresh/checkpoints/ppo_225000_steps.zip
preprocessor:     runs/ppo_clean_start_1_1_fresh/preprocessing.pkl
```

## Output

```text
runs/ppo_time_trial_continuous_1_1_225k_100k/
```

Evaluate with:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0009_ppo_time_trial_continuous_1_1_225k_100k \
  --model runs/ppo_time_trial_continuous_1_1_225k_100k/ppo_pilot_325k.zip
```

## Result

Training reached **325,800 timesteps**. The 325k deterministic evaluation:

```text
canonical:    1/1, 2.1500 s
train pool:   12/12, median 3.5583 s
holdout pool: 8/8, median 4.9500 s
```

The continuous bonus fixed the reward ordering:

```text
1.633 s episode reward: 556.8
2.133 s episode reward: 546.4
```

But the fast route appeared only once in 3,114 canonical episodes. The policy
still defaults to 2.15 s, and the train/holdout medians regressed versus the
600k stable model. The 600k model remains selected.

See [`results.md`](results.md).
