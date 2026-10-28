# Known issues

Living list of traps, stale claims, and unfinished work. For the full DQN
investigation see
[`audit/2026-10-28-dqn-pipeline-review.md`](audit/2026-10-28-dqn-pipeline-review.md).
For the start-pool clock bias, the measured 794 start-kind effect, and the PPO
A/B plan see
[`audit/2026-10-28-start-pool-clock-bias.md`](audit/2026-10-28-start-pool-clock-bias.md).
For the clean-start state/world-state decisions see
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md).

## DQN route-spawns (clean-start core implemented; validation pending)

### 1. Mid-route cell resets zero the run clock — critical

`PlayerAITrainingController.reset()` zeroes `_run_time` unless a full `state`
snapshot is supplied. The simplified time term uses
`gold_ratio = clamp(gold_time / run_time, 0, 3)`, so a mid-route spawn at
`run_time = 0` reads `gold_ratio = 3` (faster than gold) even though the agent
has skipped the route.

The finish is now a fixed `+500`, so cell starts no longer inflate the terminal
reward directly. The remaining bias is in the dense time term and the
`gold_ratio` observation: a late route start looks like a record pace until the
restored clock catches up. The old `time` profile finish ratio (`20 + 80 * ...`)
is gone.

**Fix:** capture the grounded source-checkpoint fields (`position`,
`velocity`, `state_name`, `run_time`, `best_run_time`, `run_active`,
`started_walking`), the held powerup stack, and the dynamic `world_state` for
each checkpoint. Reset with one dict; the source `run_time` and world state must
come from the same run (no mean/min across routes). The game now restores the
physical state, held powerups, and dynamic `world_state` for a one-dict reset;
full 1-8 route/eval gates are still pending. Training reset force-starts the run;
do not inject a start jump. Prefix replay and cell-only starts are not clean-start
fallbacks. A Python wrapper that corrects the terminal reward is a last resort
because it duplicates game logic. See
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md).

**Measured in the 794 1-1 run**
(`runs/ppo_fresh_1_1_sensor_794/run_1m/episodes.csv`): canonical finish term
~97, medium ~103, hard ~106, focus ~161, prefix ~73. The 25% cell weight is not
uniform damage; the late focus starts carry most of it. See the
[clock-bias audit](audit/2026-10-28-start-pool-clock-bias.md).

### 2. Extraction is all-or-nothing — high

`extract()` writes the spawn pool only after replaying every selected route. If
the process is stopped, the run directory keeps the copied routes but loses all
replay work. The first run in `runs/dqn_route_spawns_1_1/` is in exactly this
state: routes and Godot logs, no `spawns/1-1_spawn_pool.json`.

The script prints progress only to stdout; there is no Python-side run log, so a
terminal that is closed or a killed process leaves no traceback.

**Fix:** write per-route/per-split checkpoint files, log to
`runs/<run>/extract.log`, and make the pool merge idempotent.

### 3. Spawn validation is incomplete — high

The extractor keeps a state if `on_floor` is true, `run_time > 0.02`, and
`time_to_exit >= 0.5`. It does not check:

- whether the state has a supported dynamic world snapshot (powerups,
  dissolve blocks, moving platforms, reset blocks);
- whether the held powerup IDs resolve and match the powerup kind stack;
- whether an unheld pickup overlaps the start position;
- whether the first action window has a hazard;
- whether the cell is a dissolve/bounce/timed object;
- whether the game actually accepted the spawn on reset (`last_reset_spawns`,
  `spawn_cell`, or a `spawn_ok` equivalent).

Invalid spawns silently fall back to the level start. That contaminates both
training and evaluation without an error.

**Fix:** reject unsupported dynamic state, unresolved/mismatched powerup IDs,
pickup overlap, and hazard-window starts; verify each pool entry by resetting
from it once and checking the returned `spawn_cell` / `last_reset_spawns`.
Never silently fall back to canonical for a candidate. Store the validation
result in the pool. See
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md)
§7.

### 3a. Clean-start validation and promotion gates are incomplete — high

The reset/world-state/powerup/`info_request` path is implemented and smoke-tested,
but the clean-start experiment is not yet validated or promotable:

- 1-8+ powerup routes have not been re-recorded through the one-pass capture and
  full reset/eval path; the powerup/world-state unit test is not a route gate.
- The one-pass capture records non-grounded checkpoints (jump states); the
  extractor filters them, but the route registry still needs an explicit
  grounded/standable gate.
- The facing-mirror and neighboring-cell holdouts are not built yet.
- Strict NEAT rejection currently covers the basic grounded/cell/run-time/
  unsupported-object cases; pickup overlap, first-action hazard windows, and
  complete stack/ID matching still need tests.
- The reset reply advances one physics tick after the restored clock (for
  example, a 1.2667 s source state reports ~1.3333 s), so the first observation
  and position are one action interval past the source checkpoint. Measure this
  against zero-velocity drift or seed the clock after reply if exactness matters.
- The legacy 794/prefix path still exists in archived tooling and must not be
  mistaken for the clean-start target.

**Fix:** run the 1-1 and 1-8 clean-start round-trip and strict-rejection gates,
build the facing/neighbor holdout, re-record the powerup levels, then smoke and
evaluate the raw-state pool. Do not promote a model before the full gates. See
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md).

### 4. The holdout is checkpoint-level, not route-level — high

The doc says “hold out complete route clusters,” but the pool dedupes globally
by `(cell, facing)`. A quick offline pass over the actual 460 routes gave:

- top 10% = 46 routes → 34 Jaccard clusters at 0.80;
- 27 train clusters / 7 holdout clusters;
- 45 pre-dedupe holdout checkpoints, 31 of which also occur on train routes;
- 78 of 97 holdout route cells also occur on train routes;
- final pool estimate: 43 train + 23 holdout checkpoints after global dedupe.

The final keys are disjoint, so this is a valid **checkpoint** holdout. It is not
an unseen-route holdout: most holdout route cells were available in training
under another checkpoint/facing. The gate can still measure checkpoint
generalization; do not claim route generalization.

**Fix:** if route holdout is the goal, remove every train checkpoint that lies on
a holdout route, or hold out connected route segments and build train/holdout
pools from disjoint segments. The clean-start one-pass capture must group by
route signature and freeze the train/holdout split before expanding checkpoints;
do not globally dedupe `(cell, facing)` across the split. Checkpoint scenarios
are keyed by `(cell, facing, powerup stack)`.

### 5. Evaluation launches one Godot process per spawn — high

`run_eval_episode()` creates a fresh `GodotEnv` for every holdout spawn. A
23–45-spawn eval means 23–45 cold game launches (macOS cold start can take
~20 s), plus a new isolated `userdata/eval_<port>` tree each time.

**Fix:** reuse one persistent env and call `reset(spawns=...)` per start, as
`35_eval_ppo_strata.py` does. This also makes the eval deterministic and much
faster.

### 6. Training uses a sequential `DummyVecEnv` — medium

`train()` wraps 4 envs in `DummyVecEnv`, which steps them one at a time in the
parent process. Each game process is separate, but the Python loop serializes
the decisions. The PPO pilot uses `SubprocVecEnv` for `n_envs > 1`.

**Fix:** start with 1 env for a smoke, then use `SubprocVecEnv` (or the
`MultiAgentGodotVecEnv` path) for throughput. `DummyVecEnv` is fine only for a
debug run.

### 7. The DQN script is coupled to the 83 KB PPO pilot — medium

`41_dqn_route_spawns.py` loads `10_ppo_pilot.py` through `importlib` to get
`SingleAgentGodotEnv`, `resolve_env_path()`, and `isolated_home()`. Any edit to
the PPO pilot can break DQN, and the PPO pilot is the largest/most-changed file
in the repo.

**Fix:** move the shared env factory/helpers into `src/wrappers/` (or a small
`src/env_factory.py`) and have both PPO and DQN import it normally. Keep the DQN
driver small and testable.

### 8. The default level-analysis path is machine-specific — medium

`DEFAULT_ANALYSIS` is an absolute path:

```text
/Users/giani.statie/personal/dragon-jump-remaster/runs/level_analysis/1-1.json
```

It works on this machine only. `32`/`33`/`35` also embed absolute paths.

**Fix:** resolve the game repo relative to this repo (or an env var), and fail
with a clear message when the analysis file is missing.

### 9. Doc/code drift inside the DQN plan — low

- The doc says “infer facing from the next path cell”; the code records the
  actual `facing_direction` from the replay.
- The doc says “validate the checkpoint in the game”; the code never checks
  spawn acceptance.
- The doc says “reset from the pool using the start-jump convention”; that
  convention is retired. Training reset force-starts the run; mid-route states
  need a restored clock, world state, and held powerups, with no injected
  action.
- The clean-start target is raw state, not 794. Do not treat the old 794 prefix
  pool as a clean-start source.

**Fix:** align the plan/code with
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md)
and remove active-path 794/prefix assumptions.

### 10. DQN is out of scope in the game vault — process

The game vault says DQN/DDQN is deferred until after the convention
(`docs/project/active-backlog.md` A32/A33, `docs/project/decisions.md`
2026-10-27). This repo says “active experiment.” If DQN is a deliberate side
experiment, say so explicitly; if it should be official, add a backlog/decision
row in the game vault. Do not install a DQN model into the game without the full
gate.

## Cross-repo drift

### 11. Game docs say state snapshots are unsupported; the code supports them

The game's `docs/technical/training.md` says only position and facing are
restored. The committed controller (`bb76299`) accepts a `state` dictionary and
restores grounded velocity, `run_time`, `best_run_time`, `run_active`, and
`started_walking`; `archive/2026-10-27-cleanup/src/30_trajectory_holdout.py` has
a `state_from_info()` helper. The docs are stale. The limitation is narrower
than “no snapshots”: no jump states, powerups, or dissolve timers.

### 12. The game repo is dirty

`../dragon-jump-remaster` is on `prepare-for-expo` at `360b055` with
uncommitted raw-state, reward, force-start, and terminal-death changes. Any
number measured now is from a dirty tree. Record the dirty state with the
result.

### 13. The deployed-model baseline is quoted differently

`docs/reference/deployed-model-58.md` and the game status say **2.333 s** for 1-1;
some game docs say **2.35 s**. The difference is small but the ship number
should be one clean bench. Do not mix it with the 2.15 s NEAT or 1.633 s 754
numbers.

## Repo and tooling

### 14. `src/.config` ships `ACTION_REPEAT=0`

The shared config is used by the older notebook/wrapper path. `ACTION_REPEAT=0`
would crash the game's modulo if a script sends it. Active drivers override the
value (`4`), but a new script that trusts `.config` will fail. The game-side A4
clamp is still listed as open.

### 15. The branch name is historical

The branch is `neat`, but it contains the PPO, distillation, 794 sensor, and DQN
work. Do not infer the active experiment from the branch name; use
[`status.md`](status.md). Rename/split the branch only with the owner's consent.

### 16. The README's OS guidance is upstream, not local

The upstream README says local setup is Linux/Windows only and macOS should use
Docker. This checkout also has a macOS source shim
(`environments/DragonJump.app/Contents/MacOS/DragonJump`) that runs the game
source with the local Godot app. The shim is not an exported app.

### 17. The README and old experiment docs contain historical claims

The README is a mix of setup, current state, and old experiments. Use
`docs/status.md` for current state. Old NEAT/PPO results may be pre-reward-fix
and are not comparable to post-2026-10-26 runs without re-baselining.

### 18. Links to the archived experiment log can be stale

`docs/model-improvement-experiments.md` moved to
`archive/2026-10-27-cleanup/docs/model-improvement-experiments.md`. The live
docs here were updated, but old commits, external links, and the archived log
itself may still use the old path. New docs should link the archive path (or a
current experiment doc), not the old root path.

### 19. `runs/` is large and has no committed index

`runs/` is ~1 GB across the active experiment directories and is gitignored.
There is no per-run README. Use
[`reference/run-artifacts.md`](reference/run-artifacts.md) as the index; add a
row when a run becomes important.

### 20. Test coverage is thin

`tests/` only covers the static/columnar info protocol and binary observation
frame decoding. There are no tests for route selection, clustering, spawn-pool
construction, reward-clock handling, or the DQN environment. Add small pure
tests for the pool builder before another long run.

### 21. `src/processing/` and the legacy SB3 wrapper are not active

`src/processing/` and `src/wrappers/stable_baselines_wrapper.py` are tutorial /
legacy paths. Active drivers use `GodotEnv`, `SingleAgentGodotEnv`, and
`MultiAgentGodotVecEnv`. Do not add new work to the legacy path without a
reason.

### 22. Training observations are pre-processed in Godot — raw-state refactor in progress

The game currently builds the 58/754/794 feature vectors; Python only decodes
them. The 754 grid is 7×7×15, but 7 of the 15 values per cell are powerup/
dissolve/bounce state that is zero on 1-1/1-2/1-3; the 8 type bits are mostly
one-hot; the 9-value powerup tail is zero on those levels; `time_to_exit` and
the route gradient are static per cell. The 794 adds a 20-ray fan with distance
and hit but no collision type. These are archived baselines; the clean-start
target is the raw state, not 794. The `raw` encoding is implemented and
validated (342 values, 1-1 and 1-8): Godot sends a local 7×7 window (tile IDs +
tile states), player state, 8-way collisions, and 8-way flow. Python builds the
encoder; normalization stays game-side, one-hot stays Python-side. The optional
Python lean encoder is still pending; the raw 342-value vector is usable directly
and has passed a 1-1 DQN smoke. The reset/powerup/world-state restore is
implemented. The 58 runtime encoder stays in Godot. The default reward is now the
simplified game-side five-term contract (fixed 500 finish, clamped `gold_ratio`
time term); `info` also exposes `gold_ratio`, `flow_cost`, `flow_in_bounds`, and
`wasted_air_frames` so Python can build a custom reward. The clean-start
one-dict reset and dynamic `world_state` snapshot are implemented; the NN state
must not carry reset/world-state metadata. See
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md).
The old `reward_profile` values are ignored. See
[`reference/reward-contract.md`](reference/reward-contract.md). Do not add new
game-side feature engineering before this refactor finishes.

## Historical experiment caveats

- **Pre-fix PPO/NEAT numbers** were collected before the run-clock/start/reward
  fixes (2026-10-26). Treat them as legacy baselines; re-run a small A/B before
  comparing.
- **Pre-2026-10-28 reward numbers** used finish 100–220 and previous-best/time
  profiles. The shipped reward now has a fixed 500 finish and a clamped
  `gold_ratio` time term; returns are not comparable. Re-baseline before
  comparing old checkpoints.
- **Old start-jump checkpoints** were trained with a forced first jump and an
  idle reset. Training now force-starts the run; their first-step distribution
  is out of distribution. Retrain or use the compatibility flag for evaluation.
- **794/prefix clean-start numbers are legacy.** The 10k DQN smoke and 42/44
  strata eval used the 794/prefix path; they validate the shipped
  reset/reward/death mechanics, not clean-start quality. The clean-start target
  is raw state with source-checkpoint world snapshots.
- **The 1-1 PPO 512-wide plateau** (2.40 s) is a cell/start-pool result, not a
  canonical-only result. Do not compare it to the 2.15 s NEAT or 1.633 s 754
  teacher without checking the start distribution.
- **The 754/794 models are not installed.** The shipped game runtime is the
  58-value JSON MLP. A 754/794 model needs a runtime/export path and the all-16
  in-game gate.
