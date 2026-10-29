# 0116 — repair the last 1-8 clean start

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

The 0114 repair left one clean timeout at `[3,21]` facing `-1`. Can a targeted
suffix buffer for that state close the last gap while keeping canonical speed?

## Data

```text
base:            runs/ddqn_1_8_start_repair/.../ddqn_1962392_steps.zip
training pool:   runs/neat_transfer_1_8_from_1_7/pool_clean_g46_canonical_heavy/train_start_pool.json
failing start:   [3,21], facing -1, run_time 5.3333, route 1501831d...
expert suffix:   step 80 to route finish, repeated 4x
buffer:          runs/ddqn_1_8_start_repair/expert_1_8/repair_3_21_suffix_buffer.pkl (316 transitions)
budget:          2,500 steps; checkpoints every 1.25k
```

## Result

```text
checkpoint      canonical   clean states   deaths   timeouts   best
1.963640M       4.8833      37/40          0        3          4.4500
1.964888M       4.4667      33/40          0        7          4.5667
```

The targeted suffix repair over-corrected and lost broader start coverage. It is
not selected. The full-coverage 1.959896M checkpoint from 0114 is the base for
the low-LR canonical nudge in 0118.

## Artifacts

```text
runs/ddqn_1_8_last_start_repair/dqn/checkpoints/ddqn_1962392_steps.zip
runs/ddqn_1_8_last_start_repair/expert_1_8/repair_3_21_suffix_buffer.pkl
runs/ddqn_1_8_last_start_repair/eval/ddqn_1962392_steps/summary.json
```
