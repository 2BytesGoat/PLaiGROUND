# 0119 — evaluate the 1-8 canonical nudge

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Purpose

Evaluate the 0118 nudge checkpoints on 1-8 canonical plus the 40-state
strict-clean pool.

## Result

```text
checkpoint      canonical   clean states   deaths   timeouts   best     median
1.960144M       4.4667      40/40          0        0          4.5667   9.9833
1.960392M       4.4667      40/40          0        0          4.5500   9.8833
```

The selected **1.960392M** passes canonical at **4.4667 s** and covers all 40
strict-clean states with no deaths/timeouts. The repeat and all-success
diagnostic are recorded in 0120.

## Artifacts

```text
runs/ddqn_1_8_canonical_nudge/eval/ddqn_1960392_steps/summary.json
runs/ddqn_1_8_canonical_nudge/eval/selected_repeat/summary.json
```
