# 0110 — evaluate the 1-8 canonical repair

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Purpose

Evaluate the 0109 canonical-heavy repair checkpoints on 1-8 canonical plus the
40-state strict-clean pool.

## Setup

```text
level:          1-8
encoding:       lean_stick 191 -> 957
canonical:      one deterministic episode
accepted set:   runs/neat_transfer_1_8_from_1_7/pool_clean_g46/eval_clean_start_pool.json
states:         40
```

## Run

```bash
for ckpt in runs/ddqn_1_8_canonical_repair/dqn/checkpoints/*.zip; do
  .venv/bin/python tools/evaluate.py \
    --config experiments/0110_eval_ddqn_1_8_canonical_repair \
    --model "$ckpt" --tag "$(basename "$ckpt" .zip)"
done
```

## Result

```text
checkpoint      canonical   clean states   deaths   timeouts   best
1.944912M       9.5000      40/40          0        0          4.5167
1.949912M       4.5167      40/40          0        0          4.5167
1.954912M       4.5167      40/40          0        0          4.5167
1.959912M       4.5167      40/40          0        0          4.5167
1.964912M       4.5833      40/40          0        0          4.5167
```

The selected repair checkpoint is **1.949912M**: canonical **4.5167 s**, clean
**40/40**, 0 deaths/timeouts, median 7.3917, spawn 40/40. The canonical time is
still 0.0023 s above the provisional 4.5144 s gate.

## Artifacts

```text
runs/ddqn_1_8_canonical_repair/eval/ddqn_1949912_steps/summary.json
runs/ddqn_1_8_canonical_repair/eval/ddqn_1949912_steps/eval_clean_start_pool.csv
```
