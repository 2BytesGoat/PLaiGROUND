# 0058 — fresh lean_stick 191 DDQN, stage 1 (1-1)

Status: running  
Started: 2026-10-29

## Question

Can a fresh `lean_stick` 191 backbone learn the selected 1-1 clean-start policy
without inheriting the 188-input weights or the 1-4 fine-tune failure mode?

This is stage 1 of a new level-by-level chain. The 188 models remain frozen; no
188 weights are loaded. The new model sees all 191 inputs from its first update.
The three hold flags are zero on 1-1; the point of this stage is the fresh 191
backbone and the 1-1 gate.

## Setup

```text
observation:  lean_stick 191 -> 957 model inputs
model:        957 -> 512 -> 512 -> 2, ReLU
algorithm:    Double DQN, n_steps=3
levels:       1-1 only
envs:         8 processes x 1 agent
timesteps:    1,000,000
starts:       runs/neat_clean_450/pool/train_start_pool.json (12)
holdout:      runs/neat_clean_450/pool/holdout_start_pool.json (8)
archive:      runs/neat_clean_450/filtered/train_clean_routes.json
checkpoints:  every 25,000 steps
reward:       current default contract
```

## Gate

```text
1-1 canonical: 1.6167 s
1-1 train:     12/12
1-1 holdout:   8/8
```

Do not start 1-2 until this gate passes and the result is reviewed.

[0059](../0059_ddqn_fresh_1_1_lean_stick_continue/README.md) is available as a
planned 0013-equivalent **1.85M** continuation, but it is not queued or running.
Evaluate the 0058 1M checkpoint first.

## Output

```text
runs/ddqn_fresh_1_1_lean_stick/
```

Evaluate with:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0058_ddqn_fresh_1_1_lean_stick \
  --model runs/ddqn_fresh_1_1_lean_stick/dqn/dqn_final.zip
```

## Result

Complete at 1M. Selected model:

```text
runs/ddqn_fresh_1_1_lean_stick/dqn/dqn_final.zip
```

Corrected deterministic evaluation (`reassert_held_jump` enabled in eval, matching
training):

```text
canonical:    1/1, 1.5667 s
train pool:   12/12, median 3.2583 s, best 1.5667 s
holdout pool: 8/8, median 3.8000 s, best 3.3167 s
```

The first eval pass omitted `reassert_held_jump` and reported 2.7000 s
canonical. The 1M gate passes; 0059 is not needed.
