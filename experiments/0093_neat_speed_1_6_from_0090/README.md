# 0093 — speed-select the 1-6 population from 0090

Status: complete  
Started: 2026-10-29

## Question

Can the 1-6 population improve from the first-generation **4.55 s** best to the
4.28 s acceptance gate, or below the user's 3.98 s, with the time penalty?

## Why extend

0091's best DDQN checkpoint was **4.5667 s** at 1.705M (1.68M was 4.6833),
0.29 s above the 4.28 s gate, and the final checkpoint degraded to 7.25 s. The
4.55 s expert was not fast enough to pull the policy below 4.28. The 1-5
precedent (4.68 s expert → 3.1667 s DDQN) shows the DDQN can improve a mediocre
expert, but 1-6 needs a faster route archive first.

## Setup

```text
seed:             runs/neat_transfer_1_6_from_1_5/neat-checkpoint-51
archive:          runs/neat_transfer_1_6_from_1_5/archive (cumulative)
level:            1-6
action repeat:    4
time fitness:     20
generations:      30
clean stop:       disabled (min_clean_routes=0)
target time:      3.98
port:             18700
```

The initial 4.55 s bests are copied into the new run so the speed search starts
from the known clean route. The archive stays cumulative at the 0090 path.

## Target

```text
user 1-6:        3.98 s
acceptable:      <= 4.28 s
```

## Run

```bash
.venv/bin/python scripts/genetic/01_neat.py \
  --level 1-6 --obs-encoding legacy_stick --reassert-held-jump \
  --resume runs/neat_transfer_1_6_from_1_5/neat-checkpoint-51 \
  --output-dir runs/neat_speed_1_6_from_0090 \
  --archive-dir runs/neat_transfer_1_6_from_1_5/archive \
  --generations 30 --parallel 8 --action-repeat 4 --speedup 12 \
  --max-steps 300 --record-paths --checkpoint-interval 1 \
  --rebalance-every 2 --rebalance-final --min-clean-routes 0 \
  --target-time 3.98 --time-fitness-weight 20 --port 18700
```

## Next after the speed search

1. Filter the cumulative archive at Jaccard 0.5 with seed 0.
2. Replay the faster train routes at 4 frames in the 191 environment.
3. Rebuild the 1-6 expert buffer.
4. Retrain the 1-6 DDQN, likely from the 1.705M 1-6 checkpoint and with more
   1-6 envs (the 0087 old-level anchor can stay for diagnostics, but 1-6 is
   expected to be a per-level model).

## Output

```text
runs/neat_speed_1_6_from_0090/
  train.console.log
  history.csv
  genome_stats.csv
  neat-checkpoint-<generation>
  best_time.json
  best_clean.json
```

## Result

The 30-generation speed search completed in **602 s**:

```text
best clean:    4.5000 s (generation 68)
previous:      4.5500 s
archive:       376 routes / 263 clean
target 3.98:   not reached
4.28 gate:     4.5000 is +0.22
```

The search improved the expert by 0.05 s and plateaued. The cumulative archive
is now the source for the 1-6-only retrain.
