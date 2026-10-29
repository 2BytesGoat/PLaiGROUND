# 0101 — 1-7 NEAT transfer from the 4-frame 1-6 population

Status: stopped after 12 generations; clean-route branch handed to 0102  
Started: 2026-10-29  
Stopped: 2026-10-29

## Question

Can the final rebalanced 4-frame 1-6 NEAT population discover clean 1-7
routes and enough grounded checkpoints to seed a 1-7 DDQN expert?

## Source

```text
seed:          runs/neat_speed_1_6_from_0090/neat-checkpoint-81
seed origin:   0093, final 1-6 population (4.50 s stable clean route)
level analysis: ../dragon-jump-remaster/runs/level_analysis/1-7.json
archive:       runs/neat_transfer_1_7_from_1_6/archive (new, level-specific)
```

The 1-6 archive is **not** copied into the 1-7 run. Only the population is
transferred; 1-6 route data is level-specific.

## Search

```text
level:             1-7
population:        384
parallel:          8
action repeat:     4
max steps:         300
time fitness:      20 (finisher fitness = reward - 20 * run_time)
generations:       50
clean-route stop:  25
checkpoint every:  1 generation
rebalance every:   2 generations
port:              18800
```

No target-time early stop is used. The first goal is enough clean-route
diversity and grounded start coverage; speed selection can be a separate
continuation if the archive is not fast enough.

## Target

```text
level par time:   4.9728 s
provisional gate: 5.2728 s (par + 0.30 s)
```

The final 1-7 gate is not fixed by the user yet. The provisional gate is only a
reporting threshold until the route search and expert evaluation are complete.

## Run

```bash
caffeinate -dimsu .venv/bin/python scripts/genetic/01_neat.py \
  --level 1-7 --obs-encoding legacy_stick --reassert-held-jump \
  --resume runs/neat_speed_1_6_from_0090/neat-checkpoint-81 \
  --output-dir runs/neat_transfer_1_7_from_1_6 \
  --archive-dir runs/neat_transfer_1_7_from_1_6/archive \
  --generations 50 --parallel 8 --action-repeat 4 --speedup 12 \
  --max-steps 300 --record-paths --checkpoint-interval 1 \
  --rebalance-every 2 --rebalance-final --min-clean-routes 25 \
  --time-fitness-weight 20 --port 18800
```

## Result

The 1-6 population transferred immediately:

```text
generation 82:   10 / 384 finishes, best 9.1333 s
generation 85:  348 / 384 finishes, best 8.0333 s
generation 89:  380 / 384 finishes, best 7.9667 s
generation 92:  256 / 384 finishes, best 7.9667 s
```

The run was stopped at generation 92 by choice: it had produced **240 routes**
and **10 route clusters**, but **zero strict clean routes**. The fastest routes
have revisit ratios 0.14–0.16 (strict limit 0.10), so the transfer population
is a feasible-route source, not a clean teacher source.

The archive and checkpoint are preserved under
`runs/neat_transfer_1_7_from_1_6/`. The best six cluster representatives seed
the high-mutation cleanup run in
[`../0102_neat_clean_1_7_from_0101/README.md`](../0102_neat_clean_1_7_from_0101/README.md).

## Next after collection

1. Filter clean routes at Jaccard 0.5, cap 8 per cluster, 25% holdout.
2. Replay the train routes in the 191 `lean_stick` environment; drop misses.
3. Build the 1-7 start pool from grounded route checkpoints.
4. Build the stable expert buffer, filtering routes whose replay is more than
   0.25 s faster than the source.
5. Train the 1-7 DDQN from the selected 1-6 expert (1.815M), on 1-7 only, with
   no old-level environments or anchor.
6. Evaluate canonical plus every 1-7 feasible start; select the fastest
   checkpoint that covers the full start pool.

## Output

```text
runs/neat_transfer_1_7_from_1_6/
  train.console.log
  history.csv
  genome_stats.csv
  neat-checkpoint-<generation>
  archive/routes.json
  archive/finishers.jsonl
  best_time.json
  best_clean.json
```
