# 0102 — high-mutation 1-7 cleanup

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

The 1-6 → 1-7 transfer found finishers immediately but no strict clean route
after 12 generations. Can a high-mutation population seeded from the best
1-7 route clusters produce clean 1-7 routes and grounded start checkpoints?

## Source

```text
source experiment:  0101
source checkpoint:  runs/neat_transfer_1_7_from_1_6/neat-checkpoint-92
source archive:     runs/neat_transfer_1_7_from_1_6/archive
source routes:      240
source clusters:    10 at Jaccard 0.5
best transfer:      7.9667 s, revisit 0.14–0.16, 0 strict clean
seed genomes:       six cluster representatives (7.97–11.17 s)
```

Only the genomes are copied. The 1-6 archive is not merged into the 1-7
archive.

## Search

```text
level:             1-7
population:        384
parallel:          8
action repeat:     4
max steps:         300
time fitness:      20
generations:       30
clean-route stop:  25
checkpoint every:  1 generation
rebalance every:   2 generations
port:              18900
```

The high-mutation NEAT settings are the 0021/0036 recipe: weight mutation 0.9
at power 0.75, connection add 0.15, node add 0.08, compatibility 4.0, species
elitism 5, survival threshold 0.3. The config uses **61 inputs** to match the
restored 1-6/1-7 population; the first attempt used the 0021 58-input config and
failed at the first activation (`Expected 58 inputs, got 61`). The failed
attempt is preserved under `runs/neat_clean_1_7_from_0101/`; the valid run is
`runs/neat_clean_1_7_from_0101_attempt2/`.

## Target

```text
level par time:   4.9728 s
provisional gate: 5.2728 s (par + 0.30 s)
```

The immediate gate is clean-route and grounded-start coverage, not the final
speed gate.

## Run

```bash
.venv/bin/python tools/seed_neat_checkpoint.py \
  --config experiments/0102_neat_clean_1_7_from_0101/neat_config.txt \
  --genome <six source genomes> \
  --population 384 --seed 42 \
  --output-prefix runs/neat_clean_1_7_from_0101_attempt2/seed-checkpoint-

caffeinate -dimsu .venv/bin/python scripts/genetic/01_neat.py \
  --config experiments/0102_neat_clean_1_7_from_0101/neat_config.txt \
  --level 1-7 --obs-encoding legacy_stick --reassert-held-jump \
  --resume runs/neat_clean_1_7_from_0101_attempt2/seed-checkpoint-0 \
  --output-dir runs/neat_clean_1_7_from_0101_attempt2 \
  --archive-dir runs/neat_clean_1_7_from_0101_attempt2/archive \
  --generations 30 --parallel 8 --action-repeat 4 --speedup 12 \
  --max-steps 300 --record-paths --checkpoint-interval 1 \
  --rebalance-every 2 --rebalance-final --min-clean-routes 25 \
  --time-fitness-weight 20 --port 18900
```

## Result

Pending.

## Result

Completed 30 generations in **808.3 s**:

```text
best time:        7.0333 s (generation 9)
best route:       revisit 0.107, backtracks 1, action flips 0.048
archive:          902 routes / 37 clusters at Jaccard 0.5
strict clean:     0
feasible states:  20 grounded state starts after filters
```

The best 7.0333 s route is just above the strict revisit-ratio limit (0.107 vs
0.10) and is the route used as the fastest near-clean expert. The high-mutation
search improved the transfer best from 7.9667 to 7.0333 s, but did not produce a
strict-clean route. The 902-route archive is the feasible-start source for
0103; the expert replay uses the near-clean train routes (`revisit_ratio <=
0.20`).

Expert replay from the 50 near-clean train routes: 23/50 finished, 7,019
transitions; the stable filter (`source_time - replay_time <= 0.25`) kept
**2,495 transitions / 18 routes**. The fastest stable route is 7.0333 s.
