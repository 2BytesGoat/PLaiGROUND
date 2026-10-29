# 0108 — all-success diagnostic for the 1-8 expert

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Purpose

Check whether the 0106 clean-start expert also covers the broader
NEAT-discovered feasible-start set, including routes with higher revisit
ratios. These states are grounded and come from the 161-route 1-8 archive.

## Setup

```text
level:          1-8
encoding:       lean_stick 191 -> 957
action repeat:  4
diagnostic set: runs/neat_transfer_1_8_from_1_7/pool_all_success_g46/eval_all_start_pool.json
states:         107 unique (102 train + 5 unique holdout)
canonical:      not rerun here; see 0107
```

The all-success routes were filtered at Jaccard 0.5, capped at 8 per cluster,
with a 25% cluster holdout. The filter saw 161 routes / 38 clusters and kept
129 routes (103 train / 26 holdout), expanding to 102 train states and 63
holdout states. The evaluation union has 107 unique states.

## Run

```bash
for ckpt in ddqn_1927416_steps ddqn_1939912_steps ddqn_1952408_steps; do
  .venv/bin/python tools/evaluate.py \
    --config experiments/0108_eval_ddqn_1_8_all_success \
    --model runs/ddqn_1_8_feasible_starts/dqn/checkpoints/$ckpt.zip \
    --tag all_$ckpt
done
```

## Result

```text
checkpoint      all-success coverage   deaths   timeouts   best     median
1.927416M       106/107                0        1          4.4833   9.8000
1.939912M       106/107                0        1          4.4833   9.5500
1.952408M       102/107                0        5          4.4833   9.9833
```

The selected 1.939912M covers **106/107** all-success states with one timeout
at `[22,18]` facing `-1` from the high-revisit route `c00e6aad...` (revisit
ratio above the strict-clean limit). The 1.952408M checkpoint is worse on this
set (102/107, 5 timeouts).

## Artifacts

```text
runs/ddqn_1_8_feasible_starts/eval/all_ddqn_1927416_steps/
runs/ddqn_1_8_feasible_starts/eval/all_ddqn_1939912_steps/
runs/ddqn_1_8_feasible_starts/eval/all_ddqn_1952408_steps/
```
