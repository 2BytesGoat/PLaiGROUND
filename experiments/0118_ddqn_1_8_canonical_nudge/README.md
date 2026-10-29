# 0118 — low-LR canonical nudge for the full-coverage 1-8 expert

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

The 0114 checkpoint 1.959896M is **40/40** on strict-clean states but canonical
is **4.5167 s**, one frame above the provisional 4.5144 s gate. Can a very
short, low-LR canonical-only nudge reach 4.4667 s without losing coverage?

## Data

```text
base:            runs/ddqn_1_8_start_repair/.../ddqn_1959896_steps.zip
training pool:   runs/ddqn_1_8_canonical_repair/canonical_only_start_pool.json
expert buffer:   runs/ddqn_1_8_canonical_repair/expert_1_8/canonical_fast_buffer.pkl (66 transitions)
budget:          500 steps; checkpoints every 250
lr:              1e-5
exploration:     0.02 flat
expert TD:       1.0
behavior margin: 1.0 x 0.5
```

## Result

```text
checkpoint      canonical   clean states   deaths   timeouts   best     median
1.960144M       4.4667      40/40          0        0          4.5667   9.9833
1.960392M       4.4667      40/40          0        0          4.5500   9.8833
```

Both checkpoints preserve full clean coverage and fix canonical. The selected
**1.960392M** has the slightly better state best/median:

```text
canonical:   4.4667 s (provisional gate 4.5144 s)
clean pool:  40/40, 0 deaths/timeouts, median 9.8833, best 4.5500
spawn_ok:    40/40
```

## Artifacts

```text
runs/ddqn_1_8_canonical_nudge/dqn/checkpoints/ddqn_1960392_steps.zip
runs/ddqn_1_8_canonical_nudge/eval/ddqn_1960392_steps/summary.json
```
