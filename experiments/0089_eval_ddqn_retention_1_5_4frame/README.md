# 0089 — evaluate the 0088 retention checkpoints

Status: complete  
Started: 2026-10-29

Configuration-only eval experiment. It runs the deterministic canonical trace
at 4-frame control for 1-5, 1-4, 1-3, 1-2, and 1-1 against an 0088 checkpoint.

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0089_eval_ddqn_retention_1_5_4frame \
  --model runs/ddqn_retention_1_5_4frame/dqn/checkpoints/ddqn_1667500_steps.zip \
  --tag eval_1667500
```
