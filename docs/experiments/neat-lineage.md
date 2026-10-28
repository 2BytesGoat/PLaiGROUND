# NEAT lineage through the campaign

Status: **available/historical** — the current NEAT reference work uses
`scripts/genetic/01_neat.py` with the 8×24 archive, not the full lineage runner.
Last updated: 2026-10-27

This is the single-population experiment. It is intentionally different from
`scripts/genetic/02_neat_all_levels.py`, which trains one independent population per level.

`scripts/genetic/03_neat_lineage.py` carries **one population** through campaign order:

```text
1-1 → 1-2 → 1-3 → … → 1-16
```

At each level:

1. The current population is evaluated **zero-shot** on the next level.
2. Training resumes the same population and adapts it to that level.
3. The stage endpoint is copied to `snapshots/<level>/` and kept immutable.
4. One evaluation of the copied population runs on the current level.

The lineage itself continues from the snapshot checkpoint; snapshots are for
rollback and later comparisons, not separate lineages.

## Stability stop

A stage stops when all of these hold:

- at least `--min-generations` generations (default 10)
- no best-fitness improvement for `--patience` generations (default 4)
- at least `--min-finish-rate` of the population finishes for
  `--min-finish-streak` consecutive generations (defaults: 30%, 3)

`--max-generations` is the hard cap (default 30). If a level never reaches the
finish-rate gate, the stage is recorded as a failure and the same population
continues to the next level.

## Control cadence

`--action-repeat` controls how many physics ticks share one network decision.
The default is **4** — 15 decisions/s, 67 ms — which is close to Atari-style
frame skip and much more human-like than a decision every physics tick. The
same value is used for training and checkpoint evaluation.

## Run

Headless training with live logs:

```bash
cd /Users/giani.statie/personal/PLaiGROUND

.venv/bin/python scripts/genetic/03_neat_lineage.py \
  --levels all \
  --output-dir runs/neat_lineage \
  --population 100 \
  --max-generations 10 \
  --min-generations 10 \
  --patience 4 \
  --min-finish-rate 0.30 \
  --min-finish-streak 3 \
  --action-repeat 4 \
  --max-steps 600 \
  --speedup 8 \
  --eval-speedup 8
```

This quick pass deliberately caps stages at 10 generations and 10 seconds
(600 ticks) of episode time. Levels that do not reach the 30% gate are marked
as failures instead of consuming the full long-run budget.

To resume one lineage from an existing snapshot without retraining earlier
levels:

```bash
.venv/bin/python scripts/genetic/03_neat_lineage.py \
  --levels all \
  --start-level 1-3 \
  --initial-checkpoint runs/neat_lineage/snapshots/1-2/neat-checkpoint-25 \
  --output-dir runs/neat_lineage_r4 \
  --population 100 \
  --max-generations 10 \
  --min-generations 10 \
  --patience 4 \
  --min-finish-rate 0.30 \
  --min-finish-streak 3 \
  --action-repeat 4 \
  --max-steps 600 \
  --speedup 8 \
  --eval-speedup 8
```

This quick pass deliberately caps stages at 10 generations and 10 seconds
(600 ticks) of episode time. Levels that do not reach the 30% gate are marked
as failures instead of consuming the full long-run budget.

Watch the live stream from another terminal:

```bash
tail -f runs/neat_lineage/lineage.log
```

`--clean-logs` (default on) filters known Godot startup/shutdown noise from the
console while keeping full details in each stage/eval `*.log`. Use
`--no-clean-logs` for raw output. Add `--show` to open the 100-agent evaluation
windows. Training runs are headless; only the zero-shot, post-stage, and final
evaluations open a window.

## Output

```text
runs/neat_lineage/
  stages/1-1/
    history.csv
    genome_stats.csv
    node_buckets.csv
    best_genome.pkl
    neat-checkpoint-*
    train.log
  snapshots/1-1/
    ...copy of the stage endpoint...
  evals/pre-1-2/
  evals/post-1-1/
  final-evals/1-1/
  summary.csv
  final_summary.csv
  lineage_learning.png
```

`genome_stats.csv` has one row per genome per generation:

```text
generation, genome_id, nodes, enabled_connections, fitness,
outcome, run_time, steps, final_x, final_y
```

`node_buckets.csv` aggregates that by topology size:

```text
generation, nodes, count, finishes, finish_rate, deaths,
death_rate, timeouts, mean_fitness, best_fitness, best_time
```


`summary.csv` contains, per stage:

- zero-shot best time and finish count on the new level
- training generations and training best/finish count
- post-training finish count and rate

`final_summary.csv` is the final lineage evaluated across the campaign.

## Why not parallel levels?

A lineage is sequential: level N's population is level N+1's starting point.
The CPU is also the limiting factor: one 100-agent Godot process takes roughly
40–45 seconds per generation on this machine, and 8× vs 16× speedup made no
measurable difference.
