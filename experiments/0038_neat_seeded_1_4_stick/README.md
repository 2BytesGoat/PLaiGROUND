# 0038 — Hold-aware seeded NEAT 1-4

Status: complete

Started: 2026-10-29

Completed: 2026-10-29

## Question

Can the 1-4 route discovery break through once the training action contract
supports a held jump and the observation exposes ceiling/hold state, while the
live population is rebalanced every two generations?

This is a recovery run, not a transfer probe. It keeps the 0037 route archive
and changes only the contract and population management.

## What changed from 0037

`0037` discovered a rough **5.9167 s** route but had two problems:

```text
1. no archive rebalancing -> live population collapsed to ~2/384
2. no ceiling/hold observation, and training jump input was edge-triggered
```

`0038` fixes both:

```text
observation:  legacy_stick (61 values)
              legacy 58 + on_ceiling + jump_held + stick_active
action:       reassert_held_jump = true
rebalancing:  every 2 generations + final
```

The first 58 observation values remain byte-for-byte legacy. Existing
58-input agents are not retrained or re-encoded; the new fields are opt-in.

## Seed

```text
source:     runs/neat_seeded_1_4_from_0035/archive
routes:     4 (5.9167, 6.5167, 7.6500, 10.5333 s)
checkpoint: runs/neat_seeded_1_4_stick/seed-checkpoint-0
```

The source genomes are 58-input. The 61-input seed preserves their original
connections; the three new inputs start unconnected and are available to
mutation. That is intentional: the route behavior is preserved while the
search can learn to use ceiling/hold state.

## Run

```text
level:             1-4
generations:       30
population:        384
parallel:          8
speedup:           12
horizon:           300 decisions
action repeat:     4
record paths:      true
clean-route target: 5
checkpoint every:  5 generations
rebalance:         every 2 generations + final
```

## Output

```text
runs/neat_seeded_1_4_stick/
  seed-checkpoint-0
  train.console.log
  neat-checkpoint-*
  archive/
```

## Result

The clean-route target was reached in **generation 1**:

```text
finishes:       173/384
deaths:         93
best fitness:   601.7922
best time:      5.7667 s
archive:        46 routes / 10 clean / 4 clusters
wall time:      49.7 s
checkpoint:     runs/neat_seeded_1_4_stick/neat-checkpoint-1
```

This is the direct comparison to 0037 generation 1:

```text
0037:  40/384 finishes, 5.9167 s, 0 clean routes
0038: 173/384 finishes, 5.7667 s, 10 clean routes
```

The hold/ceiling contract was the missing piece, not mutation alone.

## Rebalance caveat

The post-rebalance checkpoint is **not** a full finisher population. The route
archive deduplicates by route signature, so `rebalance-1.json` saw only 23
archived finisher genomes even though generation 1 had **173 finishers**.
`neat-checkpoint-1` therefore contains 23 deployment genomes plus exploration,
not all 173 finishers. It is valid for the 1-4 pool and the first DDQN stage,
but not for the 1-5 transfer seed.

Experiment 0044 attempted to fix this with live-finish metadata, but
`Population.run` reproduces before rebalancing, so only 26 evaluated finishers
survived. Experiment 0045 preserved evaluated finisher genomes but still lost
the 50 finishers slower than 10 s because they were classified as exploration.
Experiment 0046 preserves every finisher, including >10 s routes.

## Pool

```bash
.venv/bin/python tools/filter_routes.py \
  --routes runs/neat_seeded_1_4_stick/archive/routes.json \
  --output-dir runs/neat_seeded_1_4_stick/filtered \
  --level 1-4 \
  --jaccard 0.5 \
  --max-per-cluster 8 \
  --holdout-fraction 0.25 \
  --seed 42

.venv/bin/python tools/build_start_pool.py \
  --filtered-dir runs/neat_seeded_1_4_stick/filtered \
  --archive-dir runs/neat_seeded_1_4_stick/archive \
  --analysis ../dragon-jump-remaster/runs/level_analysis/1-4.json \
  --output-dir runs/neat_seeded_1_4_stick/pool \
  --level 1-4
```

Result:

```text
10 clean routes -> 4 clusters
5 train routes / 8 train states / canonical
5 holdout routes / 3 holdout states
best 1-4 time 5.7667 s (holdout)
```

All 11 non-canonical pool starts were reset-smoked in Godot; every requested
cell was accepted. The combined 1-4/1-3/1-2/1-1 train pool is
`runs/neat_seeded_1_4_stick/pool/multilevel_train_start_pool.json` (97 starts).

## Next

0039 trained the selected 2.225M 1-1/1-2/1-3 DDQN on 4×1-4 + 2×1-3 + 1×1-2 +
1×1-1, with protected replay and `reassert_held_jump=true`. It completed with
weak 1-4 coverage (0/1 canonical, 1/9 train, 1/3 holdout); no checkpoint was
selected. 0046 now rebuilds a finisher-rich 1-4 population for the next DDQN
stage and the 1-5 seed.
