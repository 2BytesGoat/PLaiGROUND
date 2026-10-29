# 0059 — continue fresh lean_stick 191 on 1-1

Status: not needed — 0058 passed the 1M gate  
Started: 2026-10-29

## Question

Does the fresh 191 backbone need the same longer 1-1 budget as the earlier
fresh 188 DQN? Experiment 0012 reached full coverage at 1M but only 1.8667 s
canonical; 0013 continued to 1.85M and selected 1.6167 s with 12/12 train and
8/8 holdout.

This is the 0058 continuation from the 1M final model to 1.85M. It does not
start 1-2.

## Setup

```text
observation:  lean_stick 191 -> 957 model inputs
model:        fresh 191 backbone from 0058
levels:       1-1 only
envs:         8 processes x 1 agent
resume:       0058 dqn_final.zip
budget:       +850,000 steps (1.85M total)
reset steps:  false
reward:       default, unchanged
```

## Gate

```text
1-1 canonical: 1.6167 s
1-1 train:     12/12
1-1 holdout:   8/8
```

## Output

Same run directory as 0058:

```text
runs/ddqn_fresh_1_1_lean_stick/
```

The 1M checkpoint remains at
`dqn/checkpoints/ddqn_1000000_steps.zip`; the continuation writes later
checkpoints and the final model.

## Result

Not run. The corrected 0058 1M evaluation is already **1.5667 s canonical,
12/12 train, 8/8 holdout**, so the extra 850k is unnecessary.
