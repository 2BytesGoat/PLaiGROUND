# 0024 — Frequent live rebalancing on 1-3

Status: complete

Started: 2026-10-29

Completed: 2026-10-29

## Question

Does rebalancing the live population every 2 generations preserve the
speed/median portfolio that one-shot rebalancing lost?

## Source

```text
checkpoint:  runs/neat_explore_1_3_long/rebalanced_fixed/rebalanced-66
generation:  66
population:  384
```

The fixed checkpoint has the innovation tracker advanced past the archived
innovations. The output archive starts merged from 0022 and 0023:

```text
0022 routes:  304
0023 routes:  364
merged:       458 routes / 458 genomes
```

## Setup

```text
level:                  1-3
generations:            10 (67–76)
population:             384
parallel:               8
speedup:                12
horizon:                300 steps
action repeat:          4
record_paths:           true
checkpoint interval:    2
live rebalance:         every 2 generations
final rebalance:        true
power:                  AC, Low Power Mode off
```

Rebalancing happens after the evaluated generation has reproduced, so the
protected portfolio becomes the parent population for the next evaluation.
The final rebalance happens after generation 76 and is saved as the seed
checkpoint; the final evaluated distribution is still recorded in
`genome_stats.csv`.

## Hypothesis

Compared with the one-shot 0023 run, frequent rebalancing should:

```text
keep the live speed/median bins nonzero
keep finish rate near or above 57%
keep clean routes and route clusters from collapsing
produce a final protected population suitable for the 1-4 seed
```

## Output

```text
runs/neat_frequent_rebalance_1_3/
  history.csv
  genome_stats.csv
  rebalance-*.json
  rebalance-*.portfolio.json
  neat-checkpoint-*
  archive/routes.json
  archive/finishers.jsonl
  archive/genomes/
```

## Result

The hypothesis held. The successful resumed segment took **230.0 s** for
generations 69–76 after the first two generations.

```text
final generation:     384 / 384 finishes (100%)
best time:            4.2333 s
archive best clean:   4.7667 s   [one-shot archive: 5.4667 s]
archive routes:       655
clean routes:         105
clusters @ 0.5:       60
clusters @ 0.7:       162
deployment genomes:   311
```

Final evaluated bin distribution:

```text
speed:         3
median:       78
slow:        227
exploration:  76   (all 384 finished; these are >10 s finishes)
```

Final protected population:

```text
speed:         3
median:       78
slow:        230
exploration:  73
```

Rebalance trajectory:

```text
after gen 67:  3 speed / 68 median / 195 slow / 118 exploration  (266 deployable)
after gen 69:  3 speed / 72 median / 203 slow / 106 exploration  (278 deployable)
after gen 71:  3 speed / 72 median / 217 slow /  92 exploration  (292 deployable)
after gen 73:  3 speed / 76 median / 223 slow /  82 exploration  (302 deployable)
after gen 75:  3 speed / 78 median / 229 slow /  74 exploration  (310 deployable)
after gen 76:  3 speed / 78 median / 230 slow /  73 exploration  (311 deployable)
```

The comparison with the one-shot 0023 run:

```text
                   one-shot 0023     frequent 0024
final finishes     238 / 384         384 / 384
speed               0                 3
median              7                 78
slow               224                227
exploration        153                 76
archive routes     364                655
clean routes        60                105
clusters @ 0.5      46                 60
archive best clean  5.4667 s           4.7667 s
```

The final checkpoint is:

```text
runs/neat_frequent_rebalance_1_3/neat-checkpoint-76
```

It is the current 1-4 seed, with the deployment portfolio in
`rebalance-76.json`. The raw archive is also preserved.

## Seed evaluation

The final protected checkpoint was evaluated independently:

```text
runs/neat_frequent_rebalance_1_3/neat-checkpoint-76

finishes:    384 / 384
best time:   4.2333 s
best clean:  5.4667 s   (the 4.7667 s clean route remains in the archive)
bins:        3 speed / 78 median / 228 slow / 75 exploration
```

This confirms the seed population is not just portfolio metadata: every genome
finishes on 1-3. The 75 exploration-bin genomes are >10 s finishes, not
timeouts.

## Incidents fixed during the run

- The first attempt stopped after generation 68 because NEAT reproduced 383
  genomes (not divisible by 8 processes). The trainer now repairs population
  size after restore and after every generation.
- The trainer now always saves the final checkpoint.
- The rebalancer advances the innovation tracker past archived innovations.

The final checkpoint is a protected portfolio, not the result of one final
selection pass. Use it as the 1-4 seed; if further 1-3 exploration is needed,
continue with the same frequent-rebalance setting.

