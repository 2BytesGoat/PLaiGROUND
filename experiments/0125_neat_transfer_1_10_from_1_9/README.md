# 0125 — 1-10 NEAT transfer from the final 1-9 population

Status: stopped  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

Can the final 1-9 NEAT population discover feasible and clean 1-10 routes, and
produce enough grounded checkpoints to seed a 1-10 DDQN expert?

## Source

```text
seed:          runs/neat_transfer_1_9_from_1_8/neat-checkpoint-89
seed origin:   0121, final 1-9 population
source level:  1-9
level analysis: ../dragon-jump-remaster/runs/level_analysis/1-10.json
archive:       runs/neat_transfer_1_10_from_1_9/archive (new)
```

Only the population transfers. The 1-9 route archive is not copied into the
1-10 run.

## Search

```text
level:             1-10 (DOUBLE_SHORT)
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
level par time:   3.0336 s
provisional gate: 3.3336 s (par + 0.30 s)
```

The user has not set a final 1-10 target. The immediate gates are strict-clean
route diversity and grounded start coverage.

## Run

```bash
caffeinate -dimsu .venv/bin/python scripts/genetic/01_neat.py \
  --level 1-10 --obs-encoding legacy_stick --reassert-held-jump \
  --resume runs/neat_transfer_1_9_from_1_8/neat-checkpoint-89 \
  --output-dir runs/neat_transfer_1_10_from_1_9 \
  --archive-dir runs/neat_transfer_1_10_from_1_9/archive \
  --generations 50 --parallel 8 --action-repeat 4 --speedup 12 \
  --max-steps 300 --record-paths --checkpoint-interval 1 \
  --rebalance-every 2 --rebalance-final --min-clean-routes 25 \
  --time-fitness-weight 20 --port 29000
```

## Result

The first segment stopped at generation 91 on the clean-route target: 30 clean
checkpoint routes but a best time of only **4.9500 s**. The search was resumed
from checkpoint 91 for 40 more generations with the clean-route stop disabled
and an early stop at the provisional gate. It finished at generation 131
(maximum generations) without reaching the gate.

```text
generation:         131
finishers:          387
strict clean:       311 routes
near clean:         371 routes
best clean time:    4.7000 s (generation 114)
provisional gate:   3.3336 s — not reached
```

`tools/level.py data --generation 131 --quality auto` selected strict-clean:

```text
filtered routes:    109 routes / 22 clusters
train / holdout:    81 / 28 routes
state starts:       24 train / 6 holdout
unique eval set:    25 states
warnings:           none
```

The best train route is `1778c800e17b4aef32bfff1014357e0d67e7abeb` at
**4.7000 s**; the fastest holdout route is **4.7833 s**. The 4.70 s route is a
long detour (about 63 cells) versus the map's ~42-cell flow line. The game's
bundled AI controller also finishes 1-10 at **4.48 s**, and the game docs list
1-10 as a known ~22% slow level, so the 3.0336 s par is beyond the current
controllers. The expert is therefore a coverage/consistency experiment with the
speed gate open, not a gate-passing speed result.

## Output

```text
runs/neat_transfer_1_10_from_1_9/
  train.console.log
  history.csv
  genome_stats.csv
  neat-checkpoint-<generation>
  archive/routes.json
  archive/finishers.jsonl
  best_time.json
  best_clean.json
```
