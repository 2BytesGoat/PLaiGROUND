# 0096 — evaluate the 1-6-only retrain

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

Configuration-only eval for the 0095 checkpoints. It runs the deterministic
canonical 1-6 trace at 4-frame control.

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0096_eval_ddqn_1_6_retrain_1680 \
  --model runs/ddqn_1_6_retrain/dqn/dqn_final.zip \
  --tag eval_final
```

Result: final **6.3833 s**; intermediate checkpoints timed out, finished at
8.5000 s, and died. The 4.07 s expert route was not made deterministic.
