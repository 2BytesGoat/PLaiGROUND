# 0114 — repair the remaining 1-8 clean start

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

The 0112 canonical fine-tune reached **4.4667 s** but lost the `[4,21]` facing
`-1` start. Can a targeted suffix buffer restore that start without losing the
canonical speed?

## Data

```text
base:            runs/ddqn_1_8_canonical_route/.../ddqn_1952408_steps.zip
training pool:   runs/neat_transfer_1_8_from_1_7/pool_clean_g46_canonical_heavy/train_start_pool.json
start weights:   canonical 0.70 / state 0.30
failing start:   [4,21], facing -1, run_time 5.2667, route 1501831d...
expert suffix:   step 79 to route finish, repeated 4x
buffer:          runs/ddqn_1_8_canonical_route/expert_1_8/repair_suffix_buffer.pkl (320 transitions)
budget:          10,000 steps; checkpoints every 2.5k
```

## Result

```text
checkpoint      canonical   clean states   deaths   timeouts   best
1.954904M       4.4667      38/40          0        2          4.5667
1.957400M       6.6500      39/40          0        1          4.5667
1.959896M       4.5167      40/40          0        0          4.5667
1.962392M       4.4667      39/40          0        1          4.5667
```

The repair fixed `[4,21]` but `[3,21]` facing `-1` now times out at 1.962392M.
The full-coverage checkpoint 1.959896M is back at canonical 4.5167 s. The
1.962392M checkpoint keeps canonical **4.4667 s** and is the base for the final
`[3,21]` suffix repair in 0116.

## Artifacts

```text
runs/ddqn_1_8_start_repair/dqn/checkpoints/ddqn_1962392_steps.zip
runs/ddqn_1_8_start_repair/expert_1_8/repair_suffix_buffer.pkl
runs/ddqn_1_8_start_repair/eval/ddqn_1962392_steps/summary.json
```
