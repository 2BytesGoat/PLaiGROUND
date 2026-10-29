# 0023 — Rebalanced NEAT 1-3 continuation

Status: complete

Started: 2026-10-29

Completed: 2026-10-29

## Question

Does the rebalanced 384-genome population preserve its finisher/route portfolio
under 10 more generations of NEAT selection?

## Source

```text
checkpoint:  runs/neat_explore_1_3_long/rebalanced/rebalanced-66
generation:  66
population:  384
```

The source checkpoint contains the 213 archived finishers, selected by route
cluster and finish-time bin, plus 171 exploration genomes:

```text
speed:         3
median:       56
slow:        154
exploration: 171
clusters:     46
```

## Setup

```text
level:                 1-3
generations:           10 additional (67–76)
population:            384
parallel:              8
action repeat:         4
speedup:               12
horizon:               300 steps
record_paths:          true
checkpoint interval:   5
power:                 AC, Low Power Mode off
stop:                  fixed 10 generations
```

This is a validation run. It does not use strict clean-route stopping.

## Output

```text
runs/neat_rebalanced_1_3/
  history.csv
  archive/routes.json
  archive/finishers.jsonl
  archive/checkpoints/
  archive/genomes/
```

## Success criteria

```text
finish rate       near or above the 57% source level
clean routes      not collapsed
route clusters    remains diverse, near the 46-cluster source portfolio
speed/median      preserves or improves the source distribution
```

If diversity collapses, adjust the rebalancing quotas before using this
population as the 1-4 seed.

## Result

Completed 10 generations in **257.8 s**:

```text
finish rate:       238 / 384  (62.0%)   [source final: 220 / 384, 57.3%]
best time:         4.2333 s
best clean:        5.4667 s
archive routes:    364                   [source: 304]
new routes:        154
clean routes:      60                    [source: 53]
new clean routes:  25
clusters @ 0.5:    54                    [source: 46]
clusters @ 0.7:    137                   [source: 118]
```

Archive bin distribution:

```text
speed:         3
median:       65
slow:        187
exploration: 109
```

The archive grew in every useful sense: more routes, more clean routes, and
more route clusters. But the **live final population** collapsed toward slow
routes:

```text
source final (70):  5 speed / 4 median / 207 slow / 168 exploration
rebalanced final:   0 speed / 7 median / 224 slow / 153 exploration
```

The final generation was not saved because the interval checkpointer landed on
generation 72. The trainer now always saves the final population. A separate
4-generation replay from checkpoint 72 saved a usable checkpoint 76, but it did
not reproduce the original 73–76 trajectory:

```text
replay final:       219 finishes, 1 speed / 9 median / 205 slow / 169 exploration
```

That replay is a diagnostic, not the 0023 result.

## Conclusion

One rebalance at checkpoint 66 is not enough to preserve the speed tail under
NEAT selection. The archive is a strong discovery reservoir, but the live
population is not a deployment portfolio. The next run should rebalance every
**2–3 generations**, or use protected reproduction, rather than a single
checkpoint rebalance.

Follow-up: experiment 0024 rebalanced every 2 generations and restored the live
tail (3 speed / 78 median / 228 slow / 75 >10 s) with a 384/384 seed evaluation.
It is the current 1-4 seed.

