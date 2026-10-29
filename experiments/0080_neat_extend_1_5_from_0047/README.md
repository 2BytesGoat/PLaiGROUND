# 0080 — extend the 1-5 NEAT collection

Status: complete  
Started: 2026-10-29

## Question

Can the 1-5 population from the 0047 diagnostic grow from 9 clean routes to the
pool target of 25 clean routes without losing its 384/384 finish coverage?

## Why this exists

Experiment 0047 performed the required 1-4 → 1-5 transfer:

```text
zero-shot:     6/384 finishes, best 6.05 s, 2 routes, 0 clean
seeded search: generation 20, 384/384 finishes, best 4.6833 s, 9 clean
```

That was a diagnostic, stopped because the 1-4 DDQN gate had priority. The
1-4 chain is now complete (0073/0079), so 1-5 needs a real route collection and
start pool before DDQN training can be considered.

The archive at generation 20 has 9 clean routes in only 3 Jaccard-0.5 clusters,
which yields 8 train / 1 holdout routes. That is too thin for a level-N pool.

## Setup

```text
source checkpoint: runs/neat_transfer_1_5_from_1_4/neat-checkpoint-20
source archive:    runs/neat_transfer_1_5_from_1_4/archive
level:             1-5
observation:       legacy_stick (61)
action:            reassert_held_jump
population:        384, parallel 8
budget:            30 more generations (to generation 50), stop at 25 clean
speedup:           12
max decisions:     300
rebalancing:       every 2 generations + final
game:              fe7821d, clean
```

The run continues the 1-5-adapted population rather than repeating the transfer.
The route archive remains cumulative at `runs/neat_transfer_1_5_from_1_4/archive`;
only logs, checkpoints, histories, and bests are written to the new run
directory. The 1-5 archive is never merged with the 1-4 archive: start states are
level-specific.

Because the archive predates the current game commit, every route used for
training must still be replayed and reset-smoked on `fe7821d`. Clean routes that
do not replay are dropped before the pool is built.

## Run

```bash
mkdir -p runs/neat_extend_1_5_from_0047
for f in history.csv genome_stats.csv best_time.json best_time_genome.pkl \
         best_clean.json best_clean_genome.pkl best_genome.pkl; do
  cp "runs/neat_transfer_1_5_from_1_4/$f" runs/neat_extend_1_5_from_0047/
done

caffeinate -dimsu .venv/bin/python scripts/genetic/01_neat.py \
  --level 1-5 --obs-encoding legacy_stick --reassert-held-jump \
  --resume runs/neat_transfer_1_5_from_1_4/neat-checkpoint-20 \
  --output-dir runs/neat_extend_1_5_from_0047 \
  --archive-dir runs/neat_transfer_1_5_from_1_4/archive \
  --generations 30 --parallel 8 --action-repeat 4 --speedup 12 \
  --max-steps 300 --record-paths --checkpoint-interval 1 \
  --rebalance-every 2 --rebalance-final --min-clean-routes 25 \
  --port 18100
```

## Next after collection

1. Filter clean routes at Jaccard 0.5, cap 8 per cluster, 25% holdout.
2. Replay the filtered train routes in the 191 environment; drop misses.
3. Build the 1-5 train/holdout pools and the combined 1-1..1-5 train pool.
4. Write the 1-5 DDQN expert-replay manifest, using the 1.555M checkpoint as
   the source model and the 0073 expert TD + margin recipe.

## Output

```text
runs/neat_extend_1_5_from_0047/
  train.console.log
  history.csv
  neat-checkpoint-21 .. neat-checkpoint-50
  rebalance-*.json
  genome_stats.csv
  node_buckets.csv
  best_clean.json
```

## Result

The run completed 30 more generations (20 -> 50) in **414.5 s**:

```text
generation 50: 382/384 finishes, best 4.6833 s
archive:       284 routes / 16 clean
clean target:  25 not reached
```

The archive kept improving finish coverage, but the best clean time stayed at
4.6833 s, so 0081 was started to add explicit speed selection.
