# 0008 — 1-1 time-trial from the still-exploring 225k checkpoint

Status: complete; no canonical speed gain, stable 600k retained  
Started: 2026-10-28
Completed: 2026-10-28

## Question

Can the faster 1.6–1.75 s canonical route be reinforced by a stronger terminal
time incentive, starting from a checkpoint that still samples it?

The 500k/600k policies no longer sample below 1.9 s. The fresh 0004 monitor
still shows fast canonical episodes around 224.5k (`1.70 s`) and 254.4k
(`1.75 s`), so this run starts from the 225k checkpoint.

## Reward

Changed in the game training controller:

```text
time coefficient:       5.0 / s   (was 3.75 / s)
finish bonus:           500       (unchanged)
under-gold finish:      +50 flat  (strictly faster than gold)
death:                  -100      (was -50)
timeout:                -100      (training wrapper)
```

The under-gold bonus is terminal and flat. It rewards crossing the gold
threshold but does not distinguish 1.75 s from 1.60 s; the dense 5/s time term
provides the within-threshold gradient.

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
runs/ppo_time_trial_1_1_225k_100k/
```

Train with:

```bash
.venv/bin/python scripts/reinforcement/01_ppo.py \
  --config experiments/0008_ppo_time_trial_1_1_225k_100k
```

Evaluate with:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0008_ppo_time_trial_1_1_225k_100k \
  --model runs/ppo_time_trial_1_1_225k_100k/ppo_pilot_325k.zip
```

## Result

Training reached **325,800 timesteps**. The 325k model was evaluated:

```text
canonical:    1/1, 2.1500 s
train pool:   12/12, median 3.2917 s, best 2.1500 s
holdout pool: 8/8, median 3.8583 s, best 3.3833 s
```

This is **not** a canonical speed gain: the deterministic policy still takes
the 2.15 s route. It does edge the 600k model's train median by 0.033 s while
matching canonical and holdout.

Training-monitor canonical behavior:

```text
episodes:          3,035
median:            2.150 s
best:              1.633 s
≤2.2 s:            69.8%
timeouts:          0.3%
episodes <1.75 s:  1
```

The flat `+50` under-gold bonus did not make the fast route the default.
2.13 s is also under gold, so both routes receive the same terminal bonus; the
2.13 s route still wins on progress/efficiency. The 1.633 s episode scored
575.9 versus 576.0–578.1 for the stable route.

The stable 600k model remains selected. A future shortest-path attempt should
use a continuous under-gold bonus or a stronger time-trial objective.

See [`results.md`](results.md).
