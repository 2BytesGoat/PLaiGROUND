# 0107 — evaluate the 1-8 feasible-start expert

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-30

## Purpose

Evaluate 0106 checkpoints on 1-8 canonical plus the 40-state strict-clean
feasible-start pool.

## Setup

```text
level:          1-8 (DOUBLE_JUMP)
encoding:       lean_stick 191 -> 957
action repeat:  4
canonical:      one deterministic episode
accepted set:   runs/neat_transfer_1_8_from_1_7/pool_clean_g46/eval_clean_start_pool.json
accepted count: 40 state starts (37 train + 3 unique holdout)
holdout:        route-cluster split; 11 state keys are shared by train/holdout
```

The 40-state pool is the union of the filtered 1-8 train and holdout pools.
Canonical is evaluated separately. Every state is grounded and comes from a
strict-clean NEAT route.

## Gate

```text
canonical finish; provisional <= 4.5144 s
40/40 clean states finish
report deaths and timeouts separately
report spawn_ok for every start
```

The provisional gate is par 4.2144 s + 0.30 s. The user has not fixed a final
1-8 target yet.

## Run

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0107_eval_ddqn_1_8_feasible_starts \
  --model runs/ddqn_1_8_feasible_starts/dqn/checkpoints/<checkpoint>.zip \
  --tag <checkpoint>
```

## Result

All eight 0106 checkpoints were evaluated on the 40-state clean pool:

```text
checkpoint      canonical   clean states   deaths   timeouts   best
1.889928M       4.6000      25/40          0        15         4.5167
1.902424M       timeout     29/40          0        11         4.6167
1.914920M       4.6833      26/40          0        14         4.6000
1.927416M      10.0667      40/40          0         0         4.5167
1.939912M       4.5167      39/40          0         1         4.5167
1.952408M       4.5167      39/40          0         1         4.5167
1.964904M       4.5833      36/40          0         4         4.5167
1.977400M       9.8667      37/40          0         3         4.5167
```

The selected checkpoint is **1.939912M**. It repeats exactly:

```text
canonical:   4.5167 s
clean pool:  39/40, 0 deaths, 1 timeout, median 7.2667, best 4.5167
spawn_ok:    40/40
```

The one clean timeout is `[32,18]` facing `-1` from route `9537113b...`.
The canonical time is **0.0023 s** above the provisional 4.5144 s gate.

## Artifacts

```text
runs/ddqn_1_8_feasible_starts/eval/<tag>/summary.json
runs/ddqn_1_8_feasible_starts/eval/<tag>/eval_clean_start_pool.csv
runs/ddqn_1_8_feasible_starts/eval/<tag>/1-8_canonical.csv
```
