# 0021 — Exploratory NEAT 1-3

Status: complete

Started: 2026-10-29

Completed: 2026-10-29

## Question

Can a larger, higher-mutation NEAT search find any clean 1-3 routes after the
192-genome 1-2 transfer plateaued at zero finishers?

## Setup

```text
level:         1-3
population:    384
generations:   30
parallel:      8
horizon:       300 steps
action repeat: 4
speedup:       8
target:        5 clean routes
```

This is a fresh exploratory population, not a continuation of the 192-genome
1-2 transfer. The previous transfer run had 0 finishes in 20 generations and
never got above y≈424 (exit is y=10).

## Changed search settings

```text
weight_mutate_rate:  0.8 → 0.9
weight_mutate_power: 0.5 → 0.75
conn_add_prob:       0.08 → 0.15
node_add_prob:       0.03 → 0.08
compatibility_threshold: 3.0 → 4.0
max_stagnation:      20 → 30
species_elitism:     2 → 5
survival_threshold:  0.2 → 0.3
reset_on_extinction: false → true
```

## Output

```text
runs/neat_explore_1_3/
  archive/routes.json
  archive/finishers.jsonl
  archive/checkpoints/
  archive/genomes/
```

## Result

Completed 30 generations in **1,568.7 s**:

```text
best fitness:        582.49
best finish:         4.2000 s
final-generation:    8 / 384 finishes
finisher records:    113
unique routes:       12
checkpoint files:    113
strict clean routes: 0
```

The population found real 1-3 finishers, but the strict clean filter rejected
all of them. The best route has:

```text
path length:     43
revisit ratio:   0.116   (strict limit 0.10)
backtracks:      0
action flips:    0.21
```

The first three routes (4.2000 s and two 5.0333 s routes) are close to clean and
should be considered with a relaxed filter. Longer routes have more revisits.

Follow-up: experiment 0022 continues this population for 40 more generations to
strengthen the 1-4 seed.

