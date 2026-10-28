# DQN from NEAT route spawns

Status: **historical 794/prefix path — no result; clean-start uses raw state**
Started: 2026-10-28
Last updated: 2026-10-28
First level: 1-1
Driver: [`../../src/41_dqn_route_spawns.py`](../../src/41_dqn_route_spawns.py)
Detailed review: [`../audit/2026-10-28-dqn-pipeline-review.md`](../audit/2026-10-28-dqn-pipeline-review.md)
Clock-bias audit: [`../audit/2026-10-28-start-pool-clock-bias.md`](../audit/2026-10-28-start-pool-clock-bias.md)

> **Superseded for clean-start (2026-10-28).** The 794/prefix plan below is
> retained as a historical experiment record. The clean-start target is raw
> state with source-checkpoint time, held powerups, and a dynamic `world_state`
> snapshot; 794 is not used. See
> [`../audit/2026-10-28-clean-start-state-capture.md`](../audit/2026-10-28-clean-start-state-capture.md)
> and [`dqn-clean-start.md`](dqn-clean-start.md). Do not start a new clean-start
> run from the 794 prefix pool or the old cell-only extractor.

> **Historical design pivot (2026-10-28).** The first run used the start-pool
> recipe that trained the 794 PPO 1-1 model, not the cell-checkpoint extractor
> below. The 794 run used
> `runs/neat_1_1_8x24_archive/ppo/start_pool_focus.json`
> (canonical 0.65 / medium 0.10 / hard 0.10 / focus 0.05 / NEAT prefix 0.10)
> and reached 2.1333 s canonical. Cell starts zero `run_time` in the game
> reset, so the old time reward was inflated for late spawns; see the clock-bias
> audit. The 794 prefix smoke/baseline is legacy, not the clean-start path.

## Goal

Train a fresh DQN that starts from the start pools discovered by NEAT routes.

NEAT is used only as a route/spawn generator. It is not used as an action
teacher, and NEAT actions are not imitation labels. Prefix entries replay NEAT
actions during reset only; those actions never enter the replay buffer.

## Scope (historical 794/prefix path)

- Historical target: **1-1 only**, 794 `multi_hot_sensor`, 512×512. Do not use
  this for clean-start; clean-start is raw state.
- Clock-safe pool: canonical + NEAT prefix, or canonical + route-time-prior
  cells + prefix. Do not use the default 60% cell pool.
- The 58-value cell-checkpoint path is the fallback, not the first run.
- No old PPO checkpoints or trajectory datasets.
- 1-2 and 1-3 are deferred until 1-1 works.

## Inputs and network

```text
794 observations (multi_hot_sensor)
  -> 512 ReLU
  -> 512 ReLU
  -> 2 Q-values (idle, jump)
```

Fallback: 58 observations → 256 → 256 → 2. Action repeat: 4 physics frames.

## Route and spawn pipeline

**Primary path (start pool):**

1. Load `runs/neat_1_1_8x24_archive/ppo/start_pool_focus.json` or a
   clock-safe rebuild.
2. Drop or repair clock-zero cell entries: keep canonical + NEAT prefix, or
   attach a route-time prior to each cell entry.
3. Validate each entry by resetting once and checking `spawn_cell` and
   `run_time`.
4. Train DQN from the weighted pool.

**Fallback path (cell checkpoints):** the original extractor steps below.

1. Copy the 1-1 NEAT clean routes into the new run directory.
2. Filter routes to the fastest 10%, clustered by path similarity.
3. For each selected route:
   - take path cells,
   - keep standable cells,
   - infer facing from the next path cell (current code records the game's
     actual facing instead),
   - validate the checkpoint in the game (current code does **not** check
     spawn acceptance),
   - reject cells that are not valid grounded spawns.
4. Deduplicate `(cell, facing)` checkpoints.
5. Hold out complete route clusters for evaluation (see caveat below).
6. Build a weighted spawn pool: canonical + route checkpoints.
7. Reset DQN episodes from the pool; training reset force-starts the run.

## DQN training

- Double DQN (SB3 `DQN`, `double_q=True` by default).
- Fresh initialization.
- Replay buffer, target network.
- Reward: simplified game-side five-term contract (fixed +500 finish,
  clamped `gold_ratio` time term; see
  [`../reference/reward-contract.md`](../reference/reward-contract.md)).
- 794 `multi_hot_sensor`, 512×512, 1-1.
- The agent collects its own transitions after the reset. Prefix entries replay
  NEAT actions during reset only; those actions are not labels and do not enter
  the replay buffer. The strict “no NEAT action replay” rule applies to training
  transitions, not reset initialization.
- Clock-safe pool only; see the
  [clock-bias audit](../audit/2026-10-28-start-pool-clock-bias.md).

## Command (start-pool path)

```bash
# Clock-safe pool: canonical + prefix; the 0-weight cell entries are dropped
# by PPO.load_start_pool, leaving 151 effective starts.
.venv/bin/python src/34_make_ppo_start_pool.py --level 1-1 \
  --canonical-weight 0.7 --medium-weight 0 --hard-weight 0 \
  --prefix-weight 0.3 --prefix-count 150 --seed 42 \
  --output runs/neat_1_1_8x24_archive/ppo_clock_safe/start_pool.json

# Smoke: 1 env, 10k steps, no patience.
.venv/bin/python src/41_dqn_route_spawns.py \
  --run-dir runs/dqn_startpool_1_1_smoke \
  --start-pool runs/neat_1_1_8x24_archive/ppo_clock_safe/start_pool.json \
  --obs-encoding multi_hot_sensor --net-arch 512 512 \
  --n-envs 1 --timesteps 10000 --buffer-size 10000 --learning-starts 1000 \
  --eval-every 0 --patience 0 train

# Full run: 4 envs, 200k steps, checkpoints every 25k.
.venv/bin/python src/41_dqn_route_spawns.py \
  --run-dir runs/dqn_startpool_1_1 \
  --start-pool runs/neat_1_1_8x24_archive/ppo_clock_safe/start_pool.json \
  --obs-encoding multi_hot_sensor --net-arch 512 512 \
  --n-envs 4 --timesteps 200000 --buffer-size 100000 --learning-starts 2000 \
  --eval-every 25000 --patience 0 train

# Evaluate the final model on canonical + strata starts.
.venv/bin/python src/41_dqn_route_spawns.py \
  --run-dir runs/dqn_startpool_1_1 \
  --obs-encoding multi_hot_sensor --net-arch 512 512 \
  --eval-tag final eval
```

To evaluate a checkpoint, add
`--model runs/dqn_startpool_1_1/dqn/checkpoints/dqn_<steps>_steps.zip` and a
unique `--eval-tag`.

## Evaluation

Canonical 1-1, the four focus-mirror holdout starts, and the full 43-start
strata coverage. Report canonical separately: 39 of the 44 starts in
`eval_starts_heldout.json` are also training starts, so the full set is a
coverage check, not a route holdout.

Baselines:

| Model | 1-1 |
|---|---:|
| NEAT teacher | 2.150 s |
| Deployed 58 student | 2.333 s (some game docs say 2.35) |
| 794 sensor PPO (`start_pool_focus`) | 2.1333 s |
| 754 PPO teacher | 1.633 s |

The first gate is:

- canonical finishes,
- canonical time competitive with the NEAT teacher,
- held-out spawn coverage does not collapse.

## Current status (2026-10-28)

- Script and original plan committed (`d7e173d`); docs updated at `b9dd103`.
- `runs/dqn_route_spawns_1_1/` has the 460 clean routes and two Godot logs.
- `spawns/` is empty; there is no `dqn/` or `eval/`.
- The first extract stopped right after the game booted (`TRAINING_SPEEDUP`)
  and before the pool was written. No training or evaluation has run.
- **Design pivot (2026-10-28):** use the 794 start-pool recipe with a
  clock-safe pool; the cell extractor is the fallback. See the
  [clock-bias audit](../audit/2026-10-28-start-pool-clock-bias.md).
- **Smoke passed (2026-10-28):** 794 obs, 151-start clock-safe pool
  (canonical 0.7 / prefix 0.3), 1 env × 10k steps, 61/93 episodes finished.
  Only canonical and prefix resets were sampled.
- **Full run stopped (2026-10-28):** 4 envs × 200k steps in
  `runs/dqn_startpool_1_1`; stopped at ~129.5k for a design discussion.
  Checkpoints 25k–125k exist; no final model or eval. The run is the
  prefix-replay baseline, not the clean-start design.
- The working tree has an uncommitted simplification of `41` (no route
  clustering/holdout, `--min-time-to-exit`, canonical patience callback).
- Treat every number above as a **baseline**, not a DQN result.

## Before the next run (historical path)

This section is retained for the 794/prefix baseline. **Do not use it for
clean-start.**

**Historical path:** build a clock-safe pool (canonical + NEAT prefix, or
canonical + route-time-prior cells + prefix), validate every entry, then smoke
1 env × 10k before scaling. See the
[clock-bias audit](../audit/2026-10-28-start-pool-clock-bias.md).

**Clean-start path:** use raw state, a source-checkpoint `world_state`, held
powerups, and strict NEAT-side candidate validation. See
[`../audit/2026-10-28-clean-start-state-capture.md`](../audit/2026-10-28-clean-start-state-capture.md)
and [`dqn-clean-start.md`](dqn-clean-start.md). Do not train mid-route cell-only
spawns or old prefix entries; both are clock-unsafe and the prefix actions assume
a first-jump start.

Fallback extractor fixes:

1. Incremental extract output + a Python run log.
2. `powerup_count == 0` and dissolve-cell filtering.
3. Spawn acceptance check (`last_reset_spawns` / `spawn_cell`).
4. Persistent-env evaluation instead of one Godot launch per spawn.
5. `SubprocVecEnv` for multi-env training (or smoke with 1 env).
6. The checkpoint-vs-route holdout distinction.

See [`../known-issues.md`](../known-issues.md) and the
[audit](../audit/2026-10-28-dqn-pipeline-review.md) for details.

## Outputs

```text
runs/dqn_startpool_1_1/        # primary (794 start-pool path)
  pool/          copied start pool + summary
  dqn/           checkpoints, monitor.csv, episodes.csv, tensorboard
  eval/          summary.json / results.json
  godot/         raw Godot logs
  userdata/      isolated HOME/XDG per process

runs/dqn_route_spawns_1_1/     # fallback (58-value cell-checkpoint path)
  routes/        copied clean routes/finishers
  spawns/        spawn pool(s)
  dqn/           checkpoints, monitor.csv, tensorboard
  eval/          summary.json / results.json
  godot/         raw Godot logs
  userdata/      isolated HOME/XDG per process
```
