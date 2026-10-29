# 0043 — Evaluate 0039 DDQN on 1-3

Status: planned

Evaluate selected 0039 checkpoints on 1-3 canonical plus the clean train
and holdout start pools. Use the same held-input action semantics as 0039.

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0043_eval_ddqn_1_3_0039 \
  --model runs/ddqn_protected_1_1_1_2_1_3_1_4/dqn/checkpoints/<checkpoint>.zip \
  --tag 13_<checkpoint>
```

Select by the 1-3 coverage/timing gate, not by the final training step.
