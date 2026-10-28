# 0000 — NEAT clean-route collection 1-1

Status: complete
Started: 2026-10-28
Completed: 2026-10-28

## Question

Can NEAT collect at least 450 unique clean 1-1 routes with checkpoint snapshots
for clean-start training?

## Setup

- Level: `1-1`
- Observation: legacy 58 (game-side NEAT)
- Population: 192
- Parallel processes: 8
- Action repeat: 4
- Speedup: 8
- Stop target: 450 clean signatures with checkpoints
- Output: `runs/neat_clean_450/`

## Provenance

```text
Game: prepare-for-expo @ adb9278 (dirty; docs/lean test changes)
```

## Result

Stopped at generation 25:

```text
clean routes:       452
checkpoint files:   1,486
raw route records:  910
best clean time:    1.6667 s
finishes:           1,486
timeouts:           3,314
deaths:             0
```

Post-filtering:

```text
95 clusters
216 selected routes
158 train / 58 holdout routes
13 train / 10 holdout grounded starts
```

See `../../docs/audit/2026-10-28-clean-route-filter-and-lean-ppo.md`.
