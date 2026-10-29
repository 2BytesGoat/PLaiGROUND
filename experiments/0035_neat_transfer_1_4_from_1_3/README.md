# 0035 — NEAT 1-3 → 1-4 route collection

Status: complete

Started: 2026-10-29

Completed: 2026-10-29

## Question

Can the protected 1-3 NEAT population produce enough clean 1-4 (STICK) routes
for the next DQN/DDQN start pool?

## Source

```text
checkpoint:  runs/neat_frequent_rebalance_1_3/neat-checkpoint-76
source level: 1-3
generation:  76
population:  384
```

This is the latest frequent-rebalanced 1-3 population, not the final 2.45M DDQN
model. The 1-3 route archive is **not** merged into the 1-4 archive: its routes
and checkpoint states are for a different level and are not valid 1-4 starts.

## Protocol

```text
zero-shot:     evaluate checkpoint 76 on 1-4 with --record-paths
evolution:     up to 20 generations from checkpoint 76
population:    384
parallel:      8 processes
speedup:       12
horizon:       300 decisions
action repeat: 4
checkpoint:    every 2 generations
rebalance:     every 2 generations + final
early stop:    25 clean checkpoint routes
```

The zero-shot pass seeds the fresh 1-4 archive with any routes the transfer can
already produce. If the population stalls without progress, fall back to a fresh
high-mutation 384-genome search rather than spending the whole budget on the
stalled population.

## Output

```text
runs/neat_transfer_1_4_from_1_3/
  zero_shot.console.log
  train.console.log
  neat-checkpoint-*
  rebalance-*.json
  archive/
    routes.json
    finishers.jsonl
    checkpoints/
    genomes/
```

## Zero-shot result

```text
finishes:       1 / 384
best time:      10.5333 s
deaths:         41
clean routes:   0
```

One transferred genome reached the exit but the route is not clean. The archive
seed is therefore minimal; evolution is required.

## Result

The 20-generation transfer run completed in **758.8 s**, but never improved the
route:

```text
final generation:  96 (77-96)
final finishes:    6 / 384
final deaths:      0
best time:         10.5333 s   (unchanged since generation 77)
best fitness:      617.79
archive routes:    4
clean routes:      0
```

No 1-4 start pool was built. The transfer population had one rough route family
and stalled; the next step is a fresh high-mutation search in
[`0036_neat_explore_1_4_high_mutation`](../0036_neat_explore_1_4_high_mutation/README.md).

## Follow-up

Do not filter or build a pool from this archive: it has no clean routes. Use the
fresh high-mutation search in
[`0036_neat_explore_1_4_high_mutation`](../0036_neat_explore_1_4_high_mutation/README.md);
only its clean routes can become the next 1-4 start pool.
