# 0037 — Seeded high-mutation NEAT 1-4

Status: stopped

Started: 2026-10-29

Stopped: 2026-10-29

## Question

Can high-mutation exploration around the four rough 1-4 finishers from 0035
produce clean 1-4 routes, when both transfer and unseeded high mutation failed?

## Why this shape

This is the reusable recovery pattern for a new level:

```text
1. probe transfer on the new level
2. if it produces finishers but no clean routes, take the best route genomes
3. seed a high-mutation population around those genomes
4. collect paths and extend the archive
5. filter clean routes and build the start pool
6. only then train the next policy
```

The seed is one 1-4-valid genome per route, not the whole plateaued live
population. Each seed route keeps one elite copy; the other 380 genomes are
high-mutation copies.

## Seed

```text
source run:  runs/neat_transfer_1_4_from_1_3
routes:      4 finishers, 0 clean
times:       10.5333 / 10.8667 / 12.1167 / 16.3167 s
checkpoint:  runs/neat_seeded_1_4_from_0035/seed-checkpoint-0
```

The seed checkpoint was built with:

```bash
.venv/bin/python tools/seed_neat_checkpoint.py \
  --config experiments/0037_neat_seeded_1_4_from_0035/neat_config.txt \
  --genome runs/neat_transfer_1_4_from_1_3/archive/genomes/<route>.pkl \
  ... \
  --population 384 \
  --output-prefix runs/neat_seeded_1_4_from_0035/seed-checkpoint-
```

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
```

## Output

```text
runs/neat_seeded_1_4_from_0035/
  seed-checkpoint-0
  train.console.log
  neat-checkpoint-*
  archive/
```

## Selection rule

If any route is clean, filter with Jaccard 0.5 / max 8 per cluster / 25%
holdout and build the 1-4 start pool. If there are finishers but no clean
routes, extend the seeded search before another fresh random run.

## Result and blocker

Stopped during generation 25 after the seeded run plateaued:

```text
generation 1:       40/384 finishes, 5.9167 s
generation 24:      2/384 finishes, 10.53 s live
archive:            17 routes, 0 clean
best archive time:  5.9167 s
```

Two separate issues:

1. The run had no archive rebalancing, so the live population collapsed back to
   the old 10.5 s route even though the 5.9 s route was archived.
2. The 1-4 **STICK** mechanic requires holding jump for ceiling stick, but
   `PlayerAITrainingController.set_action()` applies the jump command only when
   the input changes. A held `1` is not reasserted after `JumpState.exit()`
   clears `wants_to_jump`. The heuristic controller explicitly reasserts the
   held input every frame for exactly this reason. The legacy 58-value
   observation also has no `on_ceiling` feature, only `on_wall`.

The next seeded run must fix the hold semantics and enable rebalancing before
another search. No pool was built from this archive directly; the recovery
branch is [0038](../0038_neat_seeded_1_4_stick/README.md), which kept the four
fast routes and added `legacy_stick`, `reassert_held_jump`, and rebalancing.
0038 found 10 clean routes in generation 1 and produced the 1-4 pool used by
0039.
