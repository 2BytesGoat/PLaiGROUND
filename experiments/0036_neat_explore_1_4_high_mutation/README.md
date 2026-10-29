# 0036 — Fresh high-mutation NEAT 1-4 exploration

Status: complete

Started: 2026-10-29

Completed: 2026-10-29

## Question

Can a fresh high-mutation 384-genome search find clean 1-4 (STICK) routes after
the 1-3 transfer (0035) plateaued with zero clean routes?

## Source

```text
source:        fresh population, no checkpoint
population:    384
config:        experiments/0036_neat_explore_1_4_high_mutation/neat_config.txt
```

This follows the 1-3 recovery pattern: when the transfer produced finishers but
no usable route diversity, drop the transfer seed and run a larger discovery
search with stronger mutation.

## Setup

```text
level:             1-4
generations:       30 initially
population:        384
parallel:          8
speedup:           12
horizon:           300 decisions
action repeat:     4
record paths:      true
clean-route target: 5
checkpoint every:  5 generations
rebalancing:       off during discovery
```

High-mutation settings are the 0021 1-3 recovery settings:

```text
weight_mutate_rate:       0.9
weight_mutate_power:      0.75
conn_add_prob:            0.15
node_add_prob:            0.08
compatibility_threshold:  4.0
max_stagnation:           30
species_elitism:          5
survival_threshold:       0.3
reset_on_extinction:      true
```

## Output

```text
runs/neat_explore_1_4_high_mutation/
  train.console.log
  neat-checkpoint-*
  archive/
    routes.json
    finishers.jsonl
    checkpoints/
    genomes/
```

## Result

The fresh high-mutation search completed 30 generations in **1009.9 s**:

```text
final finishes:   0 / 384
final deaths:     0
best fitness:     134.25
best time:        timeout
archive routes:   0
clean routes:     0
```

No genome reached the 1-4 exit. The population sat at the timeout fitness and
never produced a path to filter. This is a capability failure, not a clean-route
filtering problem.

## Follow-up

Do not filter or build a pool from this archive. The next attempt needs a
different search strategy: either a longer/broader high-mutation run or a
mutation-based continuation from the one rough 1-3-transfer finisher. The
current fresh population is not productive.
