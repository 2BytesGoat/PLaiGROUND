# 0022 — Long exploratory NEAT 1-3

Status: complete

Started: 2026-10-29

Completed: 2026-10-29

## Question

Can another 40 generations of the high-mutation 384-genome search produce a
stronger 1-3 population for the 1-4 seed and more usable 1-3 starts?

## Source

```text
checkpoint:  runs/neat_explore_1_3/neat-checkpoint-30
generation:  30
population:  384
```

The first exploration run found 12 unique 1-3 routes and 113 checkpoint
snapshots, with a best time of 4.2000 s, but no route passed the strict clean
filter (best revisit ratio 0.116).

## Setup

```text
level:         1-3
generations:   40 additional (31–70)
population:    384
parallel:      8
horizon:       300 steps
action repeat: 4
speedup:       8
record_paths:  true
stop:          fixed 40 generations
```

No strict clean-route stopping target. The goal is a stronger population and
more finishers, not satisfying the strict clean classifier during training.

## Output

```text
runs/neat_explore_1_3_long/
  archive/routes.json
  archive/finishers.jsonl
  archive/checkpoints/
  archive/genomes/
```

## Result

Completed 40 additional generations in **1,610.1 s**:

```text
population:          384
final finishes:      220 / 384  (~57%)
best fitness:        582.49
best time:           4.2333 s
unique routes:       304
finisher records:    6,361
checkpoint files:    6,361
strict clean routes: 53
clean best time:     5.4667 s
clean median time:   5.9833 s
route clusters:      13 at Jaccard 0.5
                     26 at Jaccard 0.7
```

The fastest route is **not** clean; the fastest clean route is 5.4667 s. The
clean set spans 5.47–9.43 s with a 5.98 s median. This gives both a speedrunning
outlier and a reliable clean portfolio.

## Next step — two-tier discovery + deployment portfolio

Do not force a Gaussian on the live NEAT population. Split the system into two
roles.

### Discovery population

```text
population:       large, high mutation
non-finishers:    allowed
selection:        fitness + novelty/route diversity
output:           raw route archive, best genomes, stepping stones
```

This is where NEAT keeps discovering 1-3/1-4 behavior. Do not cull it too early.

### Deployment portfolio

```text
finishers only
best genome per route cluster × finish-time bin
```

Suggested portfolio shape:

```text
speedrunners:   10–15%   < 4.5 s
median:         60–70%   4.5–6.0 s, different route clusters
slow/robust:    10–15%   6.0–10.0 s, only if behaviorally distinct
exploration:    5–10%    kept outside the final seed set
```

“All must finish” is a **final portfolio filter**, not an early training
constraint. Vanilla NEAT parameters can widen the distribution, but they cannot
enforce this portfolio; that needs custom reproduction or a MAP-Elites-style
archive.

For the 1-4 seed, prefer a portfolio of genomes: one speedrunner, one robust
median finisher, and one diverse route representative.

## Rebalancer test

The first checkpoint rebalancer was run against `neat-checkpoint-66` and the
full 0022 route archive:

```text
checkpoint genomes:   384
archive genomes:      294
candidate genomes:    678
finishers:            213
route clusters:       46
selected population:  384
  speed:              3
  median:             56
  slow:               154
  exploration:        171
```

The source archive does not contain enough <4.5 s or 4.5–6.0 s finishers to
fill the target quotas, so the current rebalancer preserves all 213 finishers
and fills the rest with exploration genomes. That is acceptable for a discovery
population; the deployment portfolio must use the 213 finishers only.

Outputs:

```text
runs/neat_explore_1_3_long/rebalanced/rebalanced-66
runs/neat_explore_1_3_long/rebalanced/rebalanced-.portfolio.json
```

The rebalanced checkpoint loads as a valid 384-genome population.

## 1-3 start pool

Filtering the 53 clean routes at Jaccard 0.5 with a 25% cluster holdout:

```text
input clean routes:  53
clusters:            13
filtered routes:     46
train routes:        35
holdout routes:      11
train pool entries:  44  (43 state starts + canonical)
holdout starts:      26
shared state keys:   24
```

Artifacts:

```text
runs/neat_explore_1_3_long/filtered/
runs/neat_explore_1_3_long/pool/train_start_pool.json
runs/neat_explore_1_3_long/pool/holdout_start_pool.json
```
