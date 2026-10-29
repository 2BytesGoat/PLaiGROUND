# 0123 — evaluate the 1-9 single-route expert

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

Evaluates every 0122 checkpoint on canonical 1-9 only. All eight checkpoints
finished at **2.9500 s**, 0 deaths/timeouts. Selected:
`runs/ddqn_1_9_single_route/dqn/checkpoints/ddqn_2060360_steps.zip`.

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0123_eval_ddqn_1_9_single_route \
  --model runs/ddqn_1_9_single_route/dqn/checkpoints/<checkpoint>.zip \
  --tag <checkpoint>
```
