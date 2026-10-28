# PPO clean-start on 1-1

Status: **active; 1M training run started 2026-10-28**
Started: 2026-10-28
Last updated: 2026-10-28
Level: 1-1 only
Driver: [`../../scripts/reinforcement/01_ppo.py`](../../scripts/reinforcement/01_ppo.py)
Preprocessing: [`../../src/plaiground/observations/`](../../src/plaiground/observations/)
Route filter: [`../../src/plaiground/data/route_filter.py`](../../src/plaiground/data/route_filter.py)
Route filter CLI: [`../../tools/filter_routes.py`](../../tools/filter_routes.py)
Input decision: [`../audit/2026-10-28-raw-state-observation-decision.md`](../audit/2026-10-28-raw-state-observation-decision.md)
Lean decision: [`../audit/2026-10-28-lean-state-encoding.md`](../audit/2026-10-28-lean-state-encoding.md)
Start/reset contract: [`../audit/2026-10-28-clean-start-state-capture.md`](../audit/2026-10-28-clean-start-state-capture.md)
Route reservoir decision: [`../audit/2026-10-28-clean-route-reservoir.md`](../audit/2026-10-28-clean-route-reservoir.md)
Previous PPO lineage: [`ppo-pilot.md`](ppo-pilot.md)

> **Budget decision.** This run uses **1,000,000 environment timesteps**,
> matching the target of the last 794 sensor PPO 1-1 run. It does **not** use
> 1,000,000 PPO updates. With 8 envs × 150 steps, that is about **833 PPO
> rollout updates**.
>
> The `900k` number in the old lineage was a **checkpoint/resume point**, not a
> separate update count.

> **Data decision.** This is **1-1 only**. Godot emits the **lean 188-value**
> packet: 49 primary tile IDs + 49 rotations + 49 signed phases + 9 player +
> 24 collision + 8 flow. Python one-hots the categorical fields for the model.
> The 794 `multi_hot_sensor` and prefix-replay paths are not part of this
> experiment.

> **Start decision.** Train from multiple NEAT-derived clean-start states plus
> canonical, with a **route-level holdout**. The holdout is never included in
> the training pool.

> **Route decision.** The NEAT collection target is **450 unique clean route
> signatures with checkpoint snapshots**. That is a reservoir, not a claim of
> 450 independent route families. Path near-duplicates are filtered **after
> collection** with `tools/filter_routes.py`: greedy Jaccard clustering,
> a per-cluster cap, and an optional whole-cluster train/holdout split. If the
> map yields only one route cluster, use a time/cell holdout and label it
> accordingly; do not force 450 clusters.

## 1. What the previous 1M/900k run actually was

The last 794 sensor PPO 1-1 run is in
`runs/ppo_fresh_1_1_sensor_794/run_1m`.

Main run:

```text
target timesteps: 1,000,000
processes:        8
action_repeat:    4
n_steps:          150 per env
batch_size:       120
net_arch:         64 x 64
```

The `run_900k_lr1e4_ent1e3_batch300_ar2` directory is a **resume from the 900k
checkpoint**, not a separate 900k-update experiment:

```text
timesteps run in extension: 100,000 (900k -> 1M)
action_repeat:              2
batch_size:                 300
learning_rate:              1e-4
ent_coef:                   1e-3
```

The selected 1-1 794 model was the 600k/ar2 checkpoint at **2.1333 s**
canonical, not necessarily the last 1M checkpoint. The existence of a 1M
checkpoint does not mean 1M updates.

### Terminology

| Phrase | Count in this plan |
|---|---|
| timestep / decision | one policy action step (4 physics frames at `action_repeat=4`) |
| PPO rollout update | `n_steps × n_envs = 150 × 8 = 1,200` timesteps |
| 1M timesteps | about 833 rollout updates |
| 900k timesteps | about 750 rollout updates |
| 1M PPO updates | 1.2B timesteps; **not this experiment** |

## 2. New task

Train a fresh PPO policy on **1-1** from the new NEAT clean-start scenarios and
see whether it can execute the discovered route states.

This is not a 794 continuation, not a prefix-replay continuation, and not a
copy of the old 794 policy.

## 3. NEAT data source

The current 8-route capture is smoke data from a 192-agent evaluation of
`runs/neat_1_1_8x24_archive/neat-checkpoint-61`:

```text
87 finishers
87 checkpoint files
8 clean routes after filtering
10 clean-start states after grounded/state filtering
+ canonical
```

It is enough to test the pipeline, not enough for the final holdout.

The real collection target is **450 unique clean route signatures with
checkpoint snapshots**. The old 8×24 archive shows that this is a realistic
volume: it crossed 450 at generation 53 and collected 460 by generation 65.

The current run (`runs/neat_clean_450/`) stopped on the target at generation
25 with **452 unique clean routes**, 1,486 checkpoint files, and 2.4 s wall time
for collection. Filtering at `jaccard=0.5`, `max_per_cluster=8`, and
`holdout_fraction=0.25` produced:

```text
95 clusters
216 selected routes
158 train routes / 58 holdout routes
```

Expanding the selected checkpoint files through the grounded `Idle`/`Move`,
`run_time > 0.02`, and `time_to_exit >= 0.5` gates produced only:

```text
13 train states
10 holdout states
```

Most snapshots are mid-air. All 10 holdout state keys also occur on train
routes, so this is a route-cluster split, not a state-unseen holdout.

Collect with the same 8×24 shape and stop on the route target:

```bash
.venv/bin/python scripts/genetic/01_neat.py \
  --level 1-1 \
  --population 192 \
  --parallel 8 \
  --generations 80 \
  --action-repeat 4 \
  --speedup 8 \
  --record-paths \
  --min-clean-routes 450 \
  --patience 0 \
  --checkpoint-interval 5 \
  --output-dir runs/neat_clean_450
```

`--min-clean-routes 450` counts unique clean signatures **with a checkpoint
snapshot**, not raw finishers or route clusters. The 80-generation cap is a
guard against a stalled run; it should stop earlier if it reaches the target.

The source route archive for the current smoke is:

```text
runs/neat_checkpoint_smoke3/clean/clean_routes.json
```

If the final collection has not reached 450, record more generations rather
than reusing a thin pool. If it has reached 450 but the filtered state pool is
still thin, add another checkpoint population rather than counting route
variants as independent data.

## 4. Path filtering and start-pool holdout

Filter paths **after collection**, before expanding checkpoint snapshots:

```bash
.venv/bin/python tools/filter_routes.py \
  --routes runs/neat_clean_450/archive/routes.json \
  --output-dir runs/neat_clean_450/filtered \
  --jaccard 0.5 \
  --max-per-cluster 8 \
  --holdout-fraction 0.25 \
  --seed 42
```

The filter:

- keeps the fastest route for each exact signature;
- greedily clusters near-duplicate paths by Jaccard overlap of visited cells;
- caps routes per cluster so one dominant path family cannot fill the pool;
- spreads kept routes across the cluster's run-time range;
- reserves whole clusters for holdout when requested;
- writes:

```text
filtered/filtered_clean_routes.json
filtered/route_clusters.json
filtered/summary.json
filtered/train_clean_routes.json
filtered/holdout_clean_routes.json
```

The route filter is implemented; the pool extractor and PPO evaluator still
need to consume its filtered route files and preserve the cluster split.

Then expand checkpoint snapshots from the selected routes:

```bash
.venv/bin/python tools/build_start_pool.py \
  --filtered-dir runs/neat_clean_450/filtered \
  --archive-dir runs/neat_clean_450/archive \
  --analysis /Users/giani.statie/personal/dragon-jump-remaster/runs/level_analysis/1-1.json \
  --output-dir runs/neat_clean_450/pool \
  --level 1-1 \
  --min-time-to-exit 0.5 \
  --canonical-weight 0.70 \
  --state-weight 0.30
```

The extractor:

- no holdout route states in the training pool;
- no global `(cell, facing)` dedupe that leaks holdout routes into training;
- state key remains `(cell, facing, powerup_stack)`;
- train pool weights: canonical **0.70**, training state starts **0.30**;
- canonical and each state keep their source checkpoint `run_time` and
  `world_state`.

Write:

```text
pool/split.json
pool/train_start_pool.json
pool/holdout_start_pool.json
```

The holdout pool is only for evaluation. It is not sampled by training.

## 5. Observation and preprocessing

Godot sends the raw local packet:

```text
7x7 tile IDs            49
7x7 dynamic tile states 245
player fields            16
collision rays            24
flow deltas               8
= 342 values
```

Godot emits the **lean 188-value** packet:

```text
49 primary tile IDs
49 rotations (radians)
49 signed phases
9 player values
24 collision values
8 flow deltas
= 188
```

Python then one-hots the categorical fields (tile IDs, player `state_id`,
powerup slots, collision types). The model input width is vocabulary-dependent,
not 188.

Per-cell phase:

```text
+1.0   solid / present / fully interactable
+1..0  active transition (dissolving)
-1..0  inactive transition (disabled/repairing)
 0.0   transition boundary
```

The lean encoder is implemented in `src/plaiground/observations/lean.py`:

```python
from plaiground.observations import build_lean_pipeline

pre = build_lean_pipeline(tile_names=game_tile_names)
model_x = pre.fit_transform(lean_batch)
assert lean_batch.shape[-1] == 188
assert pre.output_size > 188
```

The baseline model is intentionally simple: Python one-hot features feed the
declared `src/plaiground/models/mlp.py` with `512 × 512` hidden layers. `lean_policy_kwargs()`
provides the matching SB3 policy configuration. No custom embedding extractor is
used.

Current limitations:

- `Level.get_primary_symbol()` implements the terrain → secret → object/static
  priority, but cards/pickups still default to `phase=+1`;
- card/pickup respawn phase and moving-platform phase are not exposed yet;
- rotation currently covers bounce orientation.

`scripts/reinforcement/01_ppo.py` now accepts `--obs-encoding lean --preprocessing lean`, fits
the pipeline from the game's advertised 15-tile vocabulary, and saves it as
`preprocessing.pkl` / `preprocessing.json` in the run directory. Evaluation
loads the same pipeline and transforms both canonical and start-pool
observations.

The game-side raw 342 encoding remains available for diagnostics/comparison,
but the Python preprocessing package is lean-only and raw is not the target
input.

## 6. Training plan

Match the last 1M run's control cadence and PPO shape. The active run uses:

```bash
.venv/bin/python scripts/reinforcement/01_ppo.py \
  --level 1-1 \
  --n-envs 8 \
  --timesteps 1000000 \
  --obs-encoding lean \
  --preprocessing lean \
  --start-pool runs/neat_clean_450/pool/train_start_pool.json \
  --action-repeat 4 \
  --max-decisions 150 \
  --n-steps 150 \
  --batch-size 120 \
  --epochs 10 \
  --net-arch 512 512 \
  --speedup 8 \
  --save-freq 100000 \
  --eval-every 0 \
  --patience 0 \
  --eval-start-pool runs/neat_clean_450/pool/train_start_pool.json \
  --eval-start-pool runs/neat_clean_450/pool/holdout_start_pool.json \
  --output-dir runs/ppo_clean_start_1_1
```

Notes:

- `--timesteps` is the SB3 environment-step counter, not PPO updates.
- `--save-freq 100000` writes checkpoints at 100k, 200k, …, 1M.
- No early stopping; fixed 1M budget.
- The start pool is sampled per episode, not fixed to one cell.
- `--eval-every 0` is intentional for the first run; evaluate checkpoints
  after training so selection uses the held-out route pool.
- The 512×512 architecture is the basic lean v1 policy: one-hot features are
  larger, and the previous 794 lineage used comparable 512×512 capacity. A
  later architecture ablation can test smaller/larger nets; do not mix it into
  this baseline.

## 7. Evaluation plan

Evaluate every checkpoint from 100k through 1M with the deterministic policy:

### Groups

1. **Canonical** — full run from the normal start.
2. **Train state starts** — seen clean-start states.
3. **Holdout state starts** — unseen routes, never in training.

### Per-start fields

- `start_kind`
- `start_cell`
- `start_facing`
- source route/time
- `spawn_ok`
- `finished`
- `run_time`
- deaths / timeouts
- steps

### Selection rule

- Holdout median time is the primary metric.
- Canonical finish/time is a hard non-regression gate.
- Do not select on training reward or training finish rate alone.
- Do not compare holdout times directly with the old 794 prefix pool.

A checkpoint that is fast only on seen training states is not a valid result.

## 8. Implementation status

Implemented:

- lean preprocessing:
  - Godot emits 49 IDs + 49 rotations + 49 signed phases + 9 player +
    collisions + flow = 188;
  - `src/plaiground/observations/lean.py` decodes and one-hots tile IDs, `state_id`,
    powerup slots, and collision types;
  - `build_lean_pipeline(tile_names=...)` returns the model input.
- `scripts/reinforcement/01_ppo.py`:
  - `--obs-encoding lean --preprocessing lean`;
  - fits the pipeline from the game's advertised tile vocabulary;
  - passes it through `SingleAgentGodotEnv`;
  - applies it in canonical and start-pool evaluation;
  - saves `preprocessing.pkl` and `preprocessing.json`;
  - `--eval-start-pool` evaluates every state start in one multi-agent world.
- route post-processing:
  - `src/plaiground/data/route_filter.py` clusters near-duplicate paths;
  - `tools/filter_routes.py` is the thin CLI;
  - caps routes per cluster;
  - splits whole clusters into train/holdout route files.
- clean-start extractor:
  - `src/plaiground/data/start_pool.py` + `tools/build_start_pool.py`;
  - consumes the train/holdout route files;
  - expands checkpoint snapshots;
  - validates grounded `Idle`/`Move`, source time, powerup/world-state, and
    `time_to_exit`;
  - writes train/holdout start pools with the canonical 0.70 / state 0.30 split.
- tests: preprocessing, lean shape/one-hot, route filtering/cluster split, and
  start-pool construction.

Remaining gaps:

- card/pickup respawn phase and moving-platform phase are not exposed;
- the current route split has no state-unseen holdout: all 10 holdout state keys
  also occur in train routes;
- full clean-start validation on 1-8+ powerup routes and in-game promotion
  gates are still pending.

## 9. Acceptance and reporting

This run is a smoke-to-real baseline, not a game install:

- 1M timesteps completed without crashes;
- every training start is canonical or train-state, never holdout;
- state resets match requested cells;
- canonical remains competitive with the NEAT teacher (2.15 s) before any
  speed claim;
- holdout finish rate/time reported separately;
- no 794/prefix entry appears in the active pool;
- no model is installed until the full 16-level/in-game gates pass.

## 10. References

- [`../status.md`](../status.md)
- [`ppo-pilot.md`](ppo-pilot.md)
- [`dqn-clean-start.md`](dqn-clean-start.md)
- [`../audit/2026-10-28-clean-route-filter-and-lean-ppo.md`](../audit/2026-10-28-clean-route-filter-and-lean-ppo.md)
- [`../audit/2026-10-28-clean-route-reservoir.md`](../audit/2026-10-28-clean-route-reservoir.md)
- [`../audit/2026-10-28-clean-start-state-capture.md`](../audit/2026-10-28-clean-start-state-capture.md)
- [`../audit/2026-10-28-raw-state-observation-decision.md`](../audit/2026-10-28-raw-state-observation-decision.md)
- [`../audit/2026-10-28-start-pool-clock-bias.md`](../audit/2026-10-28-start-pool-clock-bias.md)
