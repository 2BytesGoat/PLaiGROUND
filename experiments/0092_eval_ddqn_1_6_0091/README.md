# 0092 — evaluate the 0091 1-6 DDQN

Status: complete  
Started: 2026-10-29

Configuration-only eval experiment. It runs the deterministic canonical trace at
4-frame control for 1-6, 1-5, 1-4, 1-3, 1-2, and 1-1 against an 0091 checkpoint.

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0092_eval_ddqn_1_6_0091 \
  --model runs/ddqn_expert_replay_1_6_lean_stick/dqn/checkpoints/ddqn_1705000_steps.zip \
  --tag eval_1705000
```

The selected 1-6 checkpoint is **1.705M at 4.5667 s canonical**; the 1.68M
checkpoint is 4.6833 s. The 1.705M model is canonical-only: it finishes 13/18
on the new state pool. The 1-6 row is the selected gate; the old-level rows show
whether the shared backbone retained the chain. The 1-6 model is allowed to
become the 1-6 member of a per-level chain if the old levels drift.
