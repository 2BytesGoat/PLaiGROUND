# 0090 — 1-6 NEAT transfer from the 4-frame 1-5 population

Status: complete  
Started: 2026-10-29

## Question

Can the 4-frame 1-5 NEAT population find clean 1-6 routes fast enough to seed a
1-6 DDQN, without the control-rate mismatch that broke the 2-frame 1-5 attempt?

## Source

```text
seed:          runs/neat_extend_1_5_from_0047/neat-checkpoint-50
seed origin:   0080, 1-5 population at generation 50 (4-frame)
not used:      0081 checkpoint-95 (2-frame); the 1-6 DDQN is 4-frame
archive:       runs/neat_transfer_1_6_from_1_5/archive (new, level-specific)
```

## Search

```text
level:             1-6
population:        384
parallel:          8
action repeat:     4
max steps:         300
time fitness:      20 (finisher fitness = reward - 20 * run_time)
generations:       50
clean-route stop:  25
checkpoint every:  1 generation
rebalance every:   2 generations
port:              18500
```

The time penalty is used from the start because the raw reward ranked slow
finishers above the 4.68 s route on 1-5. There is no target-time early stop; the
run ends at 25 clean routes or 50 generations, then we decide whether to extend.

## Target

```text
user 1-6 time:   3.98 s
acceptable:      <= 4.28 s
```

## Run

```bash
.venv/bin/python scripts/genetic/01_neat.py \
  --level 1-6 --obs-encoding legacy_stick --reassert-held-jump \
  --resume runs/neat_extend_1_5_from_0047/neat-checkpoint-50 \
  --output-dir runs/neat_transfer_1_6_from_1_5 \
  --archive-dir runs/neat_transfer_1_6_from_1_5/archive \
  --generations 50 --parallel 8 --action-repeat 4 --speedup 12 \
  --max-steps 300 --record-paths --checkpoint-interval 1 \
  --rebalance-every 2 --rebalance-final --min-clean-routes 25 \
  --time-fitness-weight 20 --port 18500
```

## Next after collection

1. Filter clean routes at Jaccard 0.5, cap 8 per cluster, 25% holdout.
2. Replay the train routes in the 191 environment; drop misses.
3. Build the 1-6 start pool and merge it with the 1-1..1-5 multilevel pool.
4. Build the 1-6 expert buffer.
5. Train the 1-6 DDQN from the selected 1-5 checkpoint (4 x 1-6 + 1 x 1-5 +
   2 x 1-4 + 1 x 1-3), with the 1.555M all-level 4-frame anchor.

## Output

```text
runs/neat_transfer_1_6_from_1_5/
  train.console.log
  history.csv
  genome_stats.csv
  neat-checkpoint-<generation>
  archive/routes.json
  archive/finishers.jsonl
  best_time.json
  best_clean.json
```

## Result

The first 1-6 generation hit the clean-route stop immediately:

```text
generation 51: 223 finishes / 30 deaths, best 4.5500 s
archive:       31 unique routes, 25 clean
filter:        24 clean, 9 clusters, 22 train / 2 holdout
fastest:       4.5500 s (train; seed 0)
expert replay: 20/22 finished, 1,696 transitions
expert buffer: 1,559 transitions, 20 finishes
```

The clean-route target was 25, so the search stopped at generation 51 before
extending the speed tail. The best 4.55 s is above the 4.28 s acceptance gate,
but the 1-5 precedent shows the DDQN can improve on a 4.68 s expert. The 0091
DDQN uses the 22-route train set; if it cannot reach 4.28 s, resume the 1-6 NEAT
search with `--min-clean-routes 0` and a target time.
