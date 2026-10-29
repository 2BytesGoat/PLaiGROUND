# 0120 — final 1-8 all-success diagnostic

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Purpose

Check the selected 1-8 expert (0118 **1.960392M**) on the broader 107-state
all-success pool. This pool includes feasible routes with higher revisit ratios
than the strict-clean gate.

## Setup

```text
level:          1-8
encoding:       lean_stick 191 -> 957
diagnostic set: runs/neat_transfer_1_8_from_1_7/pool_all_success_g46/eval_all_start_pool.json
states:         107 unique
canonical:      see 0119
```

## Result

```text
clean pool:        40/40, 0 deaths, 0 timeouts, best 4.5500, median 9.8833
all-success pool: 106/107, 0 deaths, 1 timeout, best 4.5500, median 10.0833
spawn_ok:          40/40 clean, 107/107 all-success
```

The one all-success timeout is `[16,21]` facing `1` from route `36a6b1e1...`
(revisit 0.226, not strict clean). The selected model therefore solves every
strict-clean start and all but one high-revisit diagnostic start.

## Artifacts

```text
runs/ddqn_1_8_canonical_nudge/eval/selected_repeat/summary.json
runs/ddqn_1_8_canonical_nudge/eval/selected_all_success/summary.json
```
