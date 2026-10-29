# 0115 — evaluate the 1-8 start repair

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Purpose

Evaluate the 0114 start-repair checkpoints on 1-8 canonical plus the 40-state
strict-clean pool.

## Result

```text
checkpoint      canonical   clean states   deaths   timeouts   best
1.954904M       4.4667      38/40          0        2          4.5667
1.957400M       6.6500      39/40          0        1          4.5667
1.959896M       4.5167      40/40          0        0          4.5667
1.962392M       4.4667      39/40          0        1          4.5667
```

The 1.959896M checkpoint is the first full 40/40 clean-coverage model, but its
canonical time is 4.5167 s. The 1.962392M checkpoint passes canonical at
**4.4667 s** with one clean timeout (`[3,21]` facing `-1`). The follow-up 0116
targets that last state.

## Artifacts

```text
runs/ddqn_1_8_start_repair/eval/ddqn_1959896_steps/summary.json
runs/ddqn_1_8_start_repair/eval/ddqn_1962392_steps/summary.json
```
