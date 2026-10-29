# 0117 — evaluate the 1-8 last-start repair

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Purpose

Evaluate the 0116 last-start repair checkpoints on 1-8 canonical plus the
40-state strict-clean pool.

## Result

```text
checkpoint      canonical   clean states   deaths   timeouts   best
1.963640M       4.8833      37/40          0        3          4.4500
1.964888M       4.4667      33/40          0        7          4.5667
```

The repair over-corrected; neither checkpoint is selected. The full-coverage
1.959896M checkpoint from 0114 remains the base for the 0118 nudge.

## Artifacts

```text
runs/ddqn_1_8_last_start_repair/eval/ddqn_1963640_steps/summary.json
runs/ddqn_1_8_last_start_repair/eval/ddqn_1964888_steps/summary.json
```
