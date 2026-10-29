# 0081 — speed-selected 1-5 NEAT search

Status: stopped  
Started: 2026-10-29

## Question

Can the 1-5 population find a route at or below the user's **3.11 s**, using an
explicit time penalty so fast finishers actually reproduce?

## Why the penalty

The 0080 archive's fastest route is **4.6833 s**, but the fitness-best genome is
a **7.1167 s** finisher:

```text
4.6833 s clean route: fitness 561.08
7.1167 s best genome: fitness 570.93
18.72 s finisher:      fitness 561.70
```

The game's finish bonus and flow/efficiency shaping outweigh its small time
pressure, so the raw reward selects slow routes. Rebalancing keeps the fast
route archived but does not make it the parent of the next generation.

`--time-fitness-weight 20` adds:

```text
finisher fitness = game reward - 20 * run_time
```

Non-finishers are unchanged, so partial-progress shaping and the death penalty
still work. At this weight the 4.68 s route beats the 7.12 s route by about 39
points.

## Setup

```text
source checkpoint: runs/neat_extend_1_5_from_0047/neat-checkpoint-50
source archive:    runs/neat_transfer_1_5_from_1_4/archive
level:             1-5
observation:       legacy_stick (61)
action:            reassert_held_jump, action_repeat 2 (30 Hz)
population:        384, parallel 8
budget:            100 generations, stop at 3.11 s
speedup:           12
rebalancing:       every 2 generations + final
time weight:       20
```

The route archive stays cumulative at the 0047 path. Only new checkpoints,
logs, and raw best files go to the 0081 run directory. The clean-route early
stop is disabled: the archive already has 16 clean slow routes, and stopping at
25 would end the speed search before it reaches the shortcut.

## Acceptance

```text
user 1-5:        3.11 s
acceptable:      <= 3.41 s (user + 0.30)
goal:            <= 3.11 s
current best:    4.6833 s -> red flag
```

## Run

```bash
caffeinate -dimsu .venv/bin/python scripts/genetic/01_neat.py \
  --level 1-5 --obs-encoding legacy_stick --reassert-held-jump \
  --resume runs/neat_extend_1_5_from_0047/neat-checkpoint-50 \
  --output-dir runs/neat_speed_1_5_from_0080 \
  --archive-dir runs/neat_transfer_1_5_from_1_4/archive \
  --generations 100 --parallel 8 --action-repeat 2 --speedup 12 \
  --max-steps 300 --record-paths --checkpoint-interval 1 \
  --rebalance-every 2 --rebalance-final \
  --target-time 3.11 --time-fitness-weight 20 --port 18300
```

## Why 30 Hz

Experiments 0047/0080 used `action_repeat = 4` (15 decisions/s). The 1-5
bunny-hop line needs tighter jump timing, so this run uses `action_repeat = 2`
(30 decisions/s). The checkpoint is compatible — action repeat is an
environment parameter, not part of the genome — but the existing fast genomes
must adapt to the finer control interval.

## If it stalls

1. Seed from the top 8 fastest archived genomes with high mutation and run
   another 50–100 generations at the same time weight.
2. Re-run at `action_repeat = 4` as the control; the 0080 run is the existing
   15 Hz baseline.
3. Use the user's 3.11 s ghost path as a route-following reference. The ghost
   stores positions, not inputs, so it cannot be replayed directly.

## Output

```text
runs/neat_speed_1_5_from_0080/
  train.console.log
  history.csv
  genome_stats.csv
  neat-checkpoint-51 .. neat-checkpoint-150
  rebalance-*.json
  best_time.json
  best_clean.json
```

## Result

Stopped at **generation 95** after the speed search plateaued:

```text
best clean:      4.39999 s (generation 87)
archive:         569 routes / 130 clean
2-frame subset:  114 clean / 6 clusters at Jaccard 0.5
seed-0 split:    32 train / 4 holdout routes
expert replay:   28/32 finished, 4,405 transitions
```

The time penalty worked: the rebalanced population went from 0 speed-bin
genomes to roughly 88-100% in the `<4.5 s` bin, and the best time improved from
4.6833 to 4.4000 s. It then converged in the 4.40-4.45 s band, still above the
3.41 s acceptance gate. The 4.40 s route and the 32-route 2-frame train set were
handed to 0082 for DDQN expert replay.
