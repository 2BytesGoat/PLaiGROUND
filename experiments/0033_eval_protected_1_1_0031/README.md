# 0033 — Evaluate protected DDQN on 1-1

Status: complete

Regression gate for experiment 0031. Evaluates canonical, train, and holdout
starts at 1-1. The old 0025 2.30M checkpoint is the control.

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0033_eval_protected_1_1_0031 \
  --model runs/ddqn_protected_1_1_1_2_1_3/dqn/checkpoints/ddqn_2300000_steps.zip \
  --tag protected_2300000_1_1
```
