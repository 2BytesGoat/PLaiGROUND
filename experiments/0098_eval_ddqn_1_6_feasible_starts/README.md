# 0098 — evaluate the 1-6 feasible-start expert

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Purpose

Deterministic evaluation for the 0097 1-6 expert:

```text
canonical 1-6:   one run
eval_all:        32 unique NEAT feasible state starts
spawn check:     every state start must report spawn_ok
```

The evaluation set is
`runs/neat_speed_1_6_from_0090/pool_state_50/eval_all_start_pool.json`
(built from all 263 clean 0093 routes; canonical is evaluated separately).

## Run

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0098_eval_ddqn_1_6_feasible_starts \
  --model <checkpoint.zip> \
  --tag <checkpoint>
```

## Selection

A checkpoint is eligible only if:

```text
canonical finish <= 4.28 s
all 32 state starts finish
spawn_ok = 32/32
no death/timeout hidden as a finish
```

The final checkpoint is not automatically selected; sweep every saved
checkpoint. Baseline 1.705M on this set is 11/32 finishes. The 0097 sweep is
recorded in
[`../0097_ddqn_1_6_feasible_starts/README.md`](../0097_ddqn_1_6_feasible_starts/README.md).
