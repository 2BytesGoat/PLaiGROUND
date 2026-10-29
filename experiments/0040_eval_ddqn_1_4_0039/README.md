# 0040 — Evaluate 0039 DDQN on 1-4

Status: planned

Evaluate selected 0039 checkpoints on 1-4 canonical plus the clean train
and holdout start pools. Use the same held-input action semantics as 0039.

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0040_eval_ddqn_1_4_0039 \
  --model runs/ddqn_protected_1_1_1_2_1_3_1_4/dqn/checkpoints/<checkpoint>.zip \
  --tag 14_<checkpoint>
```

Select by the 1-4 coverage/timing gate, not by the final training step.

## Intermediate result

`ddqn_2350000_steps.zip` (125k steps into 0039):

```text
canonical:  0/1, timeout
train:      1/9, best 6.7833 s
holdout:    1/3, best 6.7833 s
spawns:     9/9 and 3/3 accepted
```

This confirms the pool and held-input path, but is not a selection gate.
