# 0003 — lean-state DQN from clean starts (1-1)

Status: **complete; negative at 1M**  
Started: 2026-10-29
Completed: 2026-10-29

This is the fair DQN comparison against PPO: same lean 188-value game packet,
same fitted 188 → 954 preprocessing, same 12-start train pool / 8-start
holdout pool, same reward, and the same canonical/train/holdout evaluation.

## Inputs

```text
observation:      lean 188
preprocessing:    same fitted pipeline as PPO
model input:      954
model:            954 -> 512 -> 512 -> 2, ReLU
environments:     8 processes × 1 agent
timesteps:        1,000,000
starts:           validated 12-start train / 8-start holdout
reward:           current game reward (5/s time, continuous under-gold,
                  -100 death, -100 timeout)
```

The preprocessor is loaded from:

```text
runs/ppo_clean_start_1_1_fresh/preprocessing.pkl
```

## Train

```bash
.venv/bin/python scripts/reinforcement/02_dqn.py \
  --config experiments/0003_dqn_clean_start_1_1
```

## Evaluate

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0003_dqn_clean_start_1_1 \
  --model runs/dqn_clean_start_1_1/dqn/dqn_final.zip
```

## Result

The lean 8×8 DQN trained for 1M timesteps and was evaluated:

```text
canonical:    timeout (0/1)
train pool:   8/12, median 2.9750 s, best 2.2333 s
holdout pool: 7/8, median 5.2333 s, best 3.4167 s
```

All pool spawns were applied.

The 64-environment run collected data quickly but only produced ~3,900 DQN
gradient updates in 1M timesteps, versus ~31,000 in the 8×1 lean run. It did not
learn the canonical finish. The selected 600k PPO model remains far ahead.
