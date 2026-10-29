# 0105 — 1-8 NEAT transfer from the final 1-7 population

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

Can the final high-mutation 1-7 NEAT population discover feasible and clean
1-8 routes, and produce enough grounded checkpoints to seed a 1-8 DDQN expert?

## Source

```text
seed:          runs/neat_clean_1_7_from_0101_attempt2/neat-checkpoint-30
seed origin:   0102, final high-mutation 1-7 population
source level:  1-7
level analysis: ../dragon-jump-remaster/runs/level_analysis/1-8.json
archive:       runs/neat_transfer_1_8_from_1_7/archive (new)
```

Only the population transfers. The 1-7 route archive is not copied into the
1-8 run.

## Search

```text
level:             1-8 (DOUBLE_JUMP)
population:        384
parallel:          8
action repeat:     4
max steps:         300
time fitness:      20
generations:       50
clean-route stop:  25
checkpoint every:  1 generation
rebalance every:   2 generations
port:              19000
```

## Target

```text
level par time:   4.2144 s
provisional gate: 4.5144 s (par + 0.30 s)
```

The user has not set a final 1-8 target. The immediate gate is clean-route
diversity and grounded start coverage.

## Run

```bash
caffeinate -dimsu .venv/bin/python scripts/genetic/01_neat.py \
  --level 1-8 --obs-encoding legacy_stick --reassert-held-jump \
  --resume runs/neat_clean_1_7_from_0101_attempt2/neat-checkpoint-30 \
  --output-dir runs/neat_transfer_1_8_from_1_7 \
  --archive-dir runs/neat_transfer_1_8_from_1_7/archive \
  --generations 50 --parallel 8 --action-repeat 4 --speedup 12 \
  --max-steps 300 --record-paths --checkpoint-interval 1 \
  --rebalance-every 2 --rebalance-final --min-clean-routes 25 \
  --time-fitness-weight 20 --port 19000
```

## Result

The 1-7 population transferred immediately to 1-8. The search hit the clean-route
target after 16 generations (31-46) in **521.2 s**:

```text
archive:          161 grounded finishers
strict clean:     26 routes / 11 clusters at Jaccard 0.5
best time:        4.4167 s (generation 33)
best route:       revisit 0.067, backtracks 0, action flips 0.077
par / gate:       4.2144 s / 4.5144 s provisional
```

The best clean route passes the provisional gate by **0.0977 s**. The archive
contains enough clean-route diversity to build the 1-8 start pool without a
high-mutation cleanup run.

After filtering (Jaccard 0.5, 8 per cluster, 25% cluster holdout):

```text
filtered routes:  26 (11 clusters)
train / holdout:  16 / 10 routes
state starts:     37 train / 14 holdout
unique eval set:  40 states (canonical evaluated separately)
```

The post-processing is reproducible without rewriting this run:

```bash
.venv/bin/python tools/level.py data \
  --level 1-8 --run-dir runs/neat_transfer_1_8_from_1_7 --generation 46 \
  --analysis ../dragon-jump-remaster/runs/level_analysis/1-8.json \
  --output-root /tmp/level-data-1-8 \
  --quality auto --min-routes 25 --seed 0 \
  --canonical-weight 0.50 --state-weight 0.50
```

It reports `strict=26`, `near=108`, `all=161`, `37/14` states, `eval=40`,
and the holdout-only fastest-cluster warning for `c000`.

## Next after collection

1. Filter clean routes at Jaccard 0.5, cap 8 per cluster, 25% holdout.
2. Replay the train routes in the 191 `lean_stick` environment; drop misses.
3. Build the 1-8 start pool from grounded route checkpoints.
4. Build the stable expert buffer, filtering routes whose replay is more than
   0.25 s faster than the source.
5. Train the 1-8 DDQN from the selected 1-7 expert (1.877432M), on 1-8 only,
   with no old-level environments or anchor.
6. Evaluate canonical plus every 1-8 feasible start; select the fastest
   checkpoint that covers the full start pool.

## Output

```text
runs/neat_transfer_1_8_from_1_7/
  train.console.log
  history.csv
  genome_stats.csv
  neat-checkpoint-<generation>
  archive/routes.json
  archive/finishers.jsonl
  best_time.json
  best_clean.json
```
