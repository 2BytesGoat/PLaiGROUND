# 0010 — 8×8 continuous-reward time-trial from 225k

Status: complete; no speed gain, 600k retained  
Started: 2026-10-28
Completed: 2026-10-29

## Question

Does a 64-environment, 9,600-decision rollout produce enough exploration to
learn the 1.6–1.75 s route that the continuous reward now ranks above the
stable 2.13 s route?

This is the topology follow-up to experiment 0009. The reward, source
checkpoint, and canonical-only start distribution are unchanged; the change is
8 processes × 8 agents and a proportionally larger batch.

## Setup

```text
level:            1-1
observation:      lean 188 -> one-hot 954
model:            954 -> 512 -> 512 -> 2, Tanh
processes:        8 Godot processes
agents/process:   8
virtual envs:     64
n_steps:          150 per env
rollout:          9,600 decisions per update
batch_size:       960
epochs:           10
starts:           canonical only
target:           325,000 total timesteps
resume:           runs/ppo_clean_start_1_1_fresh/checkpoints/ppo_225000_steps.zip
preprocessor:     runs/ppo_clean_start_1_1_fresh/preprocessing.pkl
```

Reward:

```text
time:             5.0/s
finish:           500 base
under-gold:       min(50, 25 * max(0, gold_time - run_time))
death:            -100
timeout:          -100
```

## Output

```text
runs/ppo_time_trial_continuous_8x8_225k_100k/
```

Evaluate with:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0010_ppo_time_trial_continuous_8x8_225k_100k \
  --model runs/ppo_time_trial_continuous_8x8_225k_100k/ppo_pilot_325k.zip
```

## Result

Training reached **330,600 timesteps**. The 325k evaluation:

```text
canonical:    1/1, 2.1500 s
train pool:   12/12, median 3.4167 s
holdout pool: 8/8, median 4.2750 s
```

The 8×8 run found a **1.833 s** training sample but did not retain it:

```text
median:            2.283 s
≤2.2 s:            31.3%
timeouts:          0.11%
episodes <1.75 s:  0
```

The 9,600-decision rollout also means only ~10 PPO updates in 100k steps. The
larger batch reduced update frequency and did not produce a deterministic speed
gain. The 600k stable model remains selected.

See [`results.md`](results.md).
