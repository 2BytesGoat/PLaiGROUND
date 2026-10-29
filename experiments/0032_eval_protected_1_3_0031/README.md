# 0032 — Evaluate protected DDQN on 1-3

Status: complete

Evaluate checkpoints from experiment 0031 on the canonical start plus the 1-3
train and holdout pools. The old 0025 2.30M checkpoint is evaluated with the same
config as the control.

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0032_eval_protected_1_3_0031 \
  --model runs/ddqn_protected_1_1_1_2_1_3/dqn/checkpoints/ddqn_2300000_steps.zip \
  --tag protected_2300000_1_3
```
