# 0014 — NEAT 1-1 population → 1-2 route collection

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

Can the final 1-1 NEAT population produce useful clean 1-2 examples without
starting a fresh population? How many clean routes and start snapshots does the
transfer yield?

## Setup

```text
source level:      1-1
source checkpoint: runs/neat_clean_450/neat-checkpoint-25
target level:      1-2
population:        192
parallel:          8 Godot processes
observation:       legacy 58 (game-side NEAT)
action repeat:     4
speedup:           8
max steps:         150
collection:        --record-paths
target:            25 clean route signatures
```

The initial zero-shot evaluation of all 192 genomes on 1-2 produced:

```text
finishes:        46/192
unique routes:   23
clean routes:    5
best time:       3.5000 s
clean times:     3.5000, 3.5000, 3.5000, 3.9000, 6.1333 s
```

The same population was then resumed on 1-2 for six generations (26–31). It
reached the 25-clean-route target and stopped after 60.7 s.

```bash
.venv/bin/python scripts/genetic/01_neat.py \
  --level 1-2 \
  --resume runs/neat_clean_450/neat-checkpoint-25 \
  --generations 20 \
  --parallel 8 \
  --action-repeat 4 \
  --speedup 8 \
  --max-steps 150 \
  --record-paths \
  --min-clean-routes 25 \
  --output-dir runs/neat_clean_450_1_2_transfer \
  --archive-dir runs/neat_clean_450_1_2_transfer/archive
```

## Result

```text
unique routes:        79
clean routes:         25
best clean time:      3.3667 s
median clean time:    3.6333 s
checkpoint snapshots: 576
generations:          26–31
```

The permissive filter (Jaccard 0.5, up to 8 routes per cluster) gives:

```text
clusters:             7
cluster sizes:        [2, 14, 2, 2, 2, 1, 2]
filtered routes:      19
train routes:         16
holdout routes:       3
all with checkpoints: 19
```

The start pool from that filter has **32 train entries (31 state + canonical)**
and **20 holdout states**, but **22 of the 32 train entries come from one
cluster (c001)**. That is not diverse enough for a route-family test.

## Path diversity

Stricter clustering separates the 25 clean routes into more path families:

| Filter | Clusters | Routes | Train / holdout routes | Train starts | Largest cluster |
|---|---:|---:|---:|---:|---:|
| Jaccard 0.5, cap 8 | 7 | 19 | 16 / 3 | 32 | 22 / 32 |
| Jaccard 0.6, cap 2 | 10 | 16 | 12 / 4 | 31 | 8 / 31 |
| **Jaccard 0.7, cap 1** | **12** | **12** | **9 / 3** | **31** | **11 / 31** |

The recommended diversity set is **`filtered_j07_cap1` / `pool_j07_cap1`**:
one representative per path cluster, 12 distinct clusters, 9 train and 3
holdout routes. Its pool has **31 train entries (30 state + canonical)** and
**20 holdout states**. The largest train cluster now contributes 11/31 starts
instead of 22/32.

The route split is cluster-held out, but every holdout start state is also
reachable from at least one train route (`shared_state_keys=20`). This is the
same kind of state overlap seen in the 1-1 pool; do not treat the holdout as
fully state-disjoint.

## Output

```text
runs/neat_clean_450_1_2_transfer/
  archive/routes.json
  archive/finishers.jsonl
  archive/checkpoints/          # 576 snapshots
  archive/genomes/
  filtered/                     # permissive: 7 clusters, 19 routes
  pool/                         # permissive: 32 / 20 starts
  filtered_j07_cap1/            # diverse: 12 clusters, one route each
  pool_j07_cap1/                # diverse: 31 / 20 starts
  best_clean_genome.pkl
  best_time_genome.pkl
```

Reproduce the recommended diverse pool:

```bash
.venv/bin/python tools/filter_routes.py \
  --routes runs/neat_clean_450_1_2_transfer/archive/routes.json \
  --output-dir runs/neat_clean_450_1_2_transfer/filtered_j07_cap1 \
  --level 1-2 \
  --jaccard 0.7 \
  --holdout-fraction 0.25 \
  --max-per-cluster 1

.venv/bin/python tools/build_start_pool.py \
  --filtered-dir runs/neat_clean_450_1_2_transfer/filtered_j07_cap1 \
  --archive-dir runs/neat_clean_450_1_2_transfer/archive \
  --analysis ../dragon-jump-remaster/runs/level_analysis/1-2.json \
  --output-dir runs/neat_clean_450_1_2_transfer/pool_j07_cap1 \
  --level 1-2
```

The pool is a first usable 1-2 batch, not a 450-route archive. Extend the
population on 1-2 if more train starts or a larger holdout are needed.
