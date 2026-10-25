# NEAT lineage through the campaign

This is the single-population experiment. It is intentionally different from
`08_neat_all_levels.py`, which trains one independent population per level.

`09_neat_lineage.py` carries **one population** through campaign order:

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
- at least `--min-finish-rate` of the population finishes (default 10%)

`--max-generations` is the hard cap (default 20). If a level produces no
finish by the cap, the stage is recorded as a failure and the same population
continues to the next level.

## Run

Headless training with live logs:

```bash
cd /Users/giani.statie/personal/PLaiGROUND

.venv/bin/python src/09_neat_lineage.py \
  --levels all \
  --output-dir runs/neat_lineage \
  --population 100 \
  --max-generations 20 \
  --min-generations 10 \
  --patience 4 \
  --min-finish-rate 0.10 \
  --max-steps 900 \
  --speedup 8 \
  --eval-speedup 8
```

Watch the live stream from another terminal:

```bash
tail -f runs/neat_lineage/lineage.log
```

Add `--show` to open the 100-agent evaluation windows. Training runs are
headless; only the zero-shot, post-stage, and final evaluations open a window.

## Output

```text
runs/neat_lineage/
  stages/1-1/
    history.csv
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
