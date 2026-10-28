# Clean-route filter, start pools, and lean PPO launch

Date: 2026-10-28
Status: active; PPO training in progress
Level: 1-1
Game repo: `../dragon-jump-remaster`, branch `prepare-for-expo`, HEAD `adb9278`
(`more state changes`), game tree dirty (docs/lean work).
PLaiGROUND: branch `neat`, HEAD `b9dd103`, working tree dirty.

## Input

`runs/neat_clean_450/archive/routes.json`

```text
raw route records:       910
clean checkpoint routes: 452
checkpoint files:        1,486
best clean time:         1.6667 s
NEAT stop:               generation 25, clean-route target
```

The archive is a signature-keyed mapping whose values did not repeat
`route_signature`. `route_filter.load_routes()` now preserves the mapping key as
the signature; future `07` archives also write the field into each value.

## Filter

```bash
.venv/bin/python tools/filter_routes.py \
  --routes runs/neat_clean_450/archive/routes.json \
  --output-dir runs/neat_clean_450/filtered \
  --level 1-1 \
  --jaccard 0.5 \
  --max-per-cluster 8 \
  --holdout-fraction 0.25 \
  --seed 42
```

Result:

```text
input records:       910
unclean skipped:     458
unique signatures:   452
clusters:            95
filtered routes:     216
train routes:        158 (71 clusters)
holdout routes:      58  (24 clusters)
```

The four largest route families contained 124, 66, 40, and 28 raw routes; the
per-cluster cap kept 8 from each.

## Start pools

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

Gates:

- `on_floor == true`;
- state name `Idle` or `Move`;
- `run_time > 0.02`;
- level-analysis `time_to_exit >= 0.5`;
- valid cell/position/velocity;
- `world_state` present when a powerup stack is held.

Result:

```text
train routes:          158
holdout routes:        58
train states:          13
holdout states:        10
state key:             (cell, facing, powerup_stack)
canonical weight:      0.70
state weight:          0.30 / 13
```

Why only 13 train states: the one-pass capture stores many mid-air snapshots.
Of 6,935 train checkpoint rows, 4,992 were airborne, 1,372 were grounded but had
`time_to_exit < 0.5`, and 340 were valid before state-key deduplication. The
holdout had 2,425 rows, 1,660 airborne, 574 late-route rejects, and 132 valid
before deduplication.

**Split limitation.** All 10 holdout state keys also occur on train routes. This
is a whole-cluster route split at the route level, but not a state-unseen
holdout. Do not use the holdout state result as an unseen-route generalization
claim.

## Lean PPO integration

`scripts/reinforcement/01_ppo.py` now supports:

```text
--obs-encoding lean
--preprocessing lean
--preprocessing-path <pickle>
--eval-start-pool <pool.json>   (repeatable)
```

The game advertises 15 tile names. `build_lean_pipeline()` fits a fixed
vocabulary and the observation becomes:

```text
188 raw lean values -> 954 model inputs
```

The preprocessor is saved as `preprocessing.pkl` plus `preprocessing.json`.
Training and evaluation use the same object. The model is the basic 512×512
Tanh MLP.

## Smoke gates

1 env × 2,048 timesteps:

```text
188 -> 954
state resets: yes
44 episodes
7 finishes from state starts; 4 canonical finishes
```

2 env × 512 timesteps through `SubprocVecEnv`: passed; preprocessor pickled
across worker processes.

Evaluation smoke on the 2k smoke model:

```text
canonical:   0/2 at 20 decisions (deliberately truncated)
train pool:  7/14 finishes, spawn_ok 14/14, median 2.383 s
holdout:     6/10 finishes, spawn_ok 10/10, median 5.225 s
```

The 20-decision cap explains the canonical zeros; it was an integration test,
not a quality result.

## Active run

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

Checkpoints every 100k; canonical, train-state, and holdout-state evaluation
run after the 1M budget. No model is installed in the game.
