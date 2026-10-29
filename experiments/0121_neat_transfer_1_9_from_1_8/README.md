# 0121 — 1-9 NEAT transfer from the final 1-8 population

Status: stopped  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

Can the final 1-8 NEAT population discover feasible and clean 1-9 routes, and
produce enough grounded checkpoints to seed a 1-9 DDQN expert?

## Source

```text
seed:          runs/neat_transfer_1_8_from_1_7/neat-checkpoint-46
seed origin:   0105, final 1-8 population
source level:  1-8
level analysis: ../dragon-jump-remaster/runs/level_analysis/1-9.json
archive:       runs/neat_transfer_1_9_from_1_8/archive (new)
```

Only the population transfers. The 1-8 route archive is not copied into the
1-9 run.

## Search

```text
level:             1-9 (CHAIN_JUMPS)
population:        384
parallel:          8
action repeat:     4
max steps:         300
time fitness:      20
generations:       50
clean-route stop:  25
checkpoint every:  1 generation
rebalance every:   2 generations
port:              29000
```

## Target

```text
level par time:   2.7648 s
provisional gate: 3.0648 s (par + 0.30 s)
```

The user has not set a final 1-9 target. The immediate gates are strict-clean
route diversity and grounded start coverage.

## Run

```bash
caffeinate -dimsu .venv/bin/python scripts/genetic/01_neat.py \
  --level 1-9 --obs-encoding legacy_stick --reassert-held-jump \
  --resume runs/neat_transfer_1_8_from_1_7/neat-checkpoint-46 \
  --output-dir runs/neat_transfer_1_9_from_1_8 \
  --archive-dir runs/neat_transfer_1_9_from_1_8/archive \
  --generations 50 --parallel 8 --action-repeat 4 --speedup 12 \
  --max-steps 300 --record-paths --checkpoint-interval 1 \
  --rebalance-every 2 --rebalance-final --min-clean-routes 25 \
  --time-fitness-weight 20 --port 29000
```

## Result

The run was stopped at generation 89 after the user confirmed 1-9 is a
single-route precision map. It had already found enough clean routes for the
single-route expert:

```text
archive:           11 grounded finishers
strict clean:      11 routes / 6 clusters
best clean time:   2.9500 s
generation:        89
provisional gate:  3.0648 s (par + 0.30 s)
```

Post-processing with `tools/level.py data`:

```text
filtered routes:   11
train / holdout:   9 / 2 routes
state starts:      6 train / 8 holdout
unique eval set:   11 states
```

The selected single route is:

```text
signature:         926883585ecb0f7469bba7968907f4014e23b235
source time:       2.9500 s
path cells:        38
actions:           43 at action repeat 4
```

## Next after collection

1. Replay only the fastest route in the 191 `lean_stick` environment.
2. Build the one-route expert buffer, filtering replay gain above 0.25 s.
3. Train the 1-9 DDQN from the selected 1-8 expert (1.960392M), on 1-9 only,
   with canonical-only resets and no old-level environments or anchor.
4. Evaluate canonical 1-9 and select the checkpoint. The broad 11-state pool
   remains a diagnostic, not the accepted gate for this single-route task.

## Output

```text
runs/neat_transfer_1_9_from_1_8/
  train.console.log
  history.csv
  genome_stats.csv
  neat-checkpoint-<generation>
  archive/routes.json
  archive/finishers.jsonl
  best_time.json
  best_clean.json
```
