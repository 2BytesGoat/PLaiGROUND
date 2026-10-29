# Known issues

Living list of traps, stale claims, and unfinished work. For the full DQN
investigation see
[`audit/2026-10-28-dqn-pipeline-review.md`](audit/2026-10-28-dqn-pipeline-review.md).
For the start-pool clock bias, the measured 794 start-kind effect, and the PPO
A/B plan see
[`audit/2026-10-28-start-pool-clock-bias.md`](audit/2026-10-28-start-pool-clock-bias.md).
For the clean-start state/world-state decisions see
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md).

## Active traps

Start here. These are the current traps; resolved items below are kept for
history. Read the matching section below before touching the area.

- **Mid-route cell resets zero the run clock** (critical, DQN route-spawns) —
  start time and reward must come from the source checkpoint, not a fresh reset.
- **One-shot NEAT rebalancing does not preserve the live distribution** (high)
  — rebalance every few generations and verify the live population.
- **DQN extraction is all-or-nothing** (high) — a failed extraction writes no
  pool; check the output before launching training.
- **Spawn validation is incomplete** (high) — check `spawn_ok` and the start
  pool before training.
- **Clean-start validation and promotion gates are incomplete** (high).
- **The holdout is checkpoint-level, not route-level** (high) — cluster routes
  first, then split train/holdout.
- **Changing `action_repeat` breaks the old-level anchor** (high).
- **Single-model 1-5 training damages 1-4 and late 1-2** (high).
- **Checkpoint selection is manual and finals regress** (high).
- **Canonical eval is one deterministic trace** (medium) — it is not a
  robustness measure.
- **NEAT fitness does not rank fast finishers first** (medium).

## NEAT checkpoint and portfolio traps

### 1. Interval checkpointing can miss the final generation — resolved

NEAT's interval checkpointer can end on a generation that is not a multiple of
the interval (0022 stopped at 70 with the last checkpoint at 66; 0023 stopped at
76 with the last checkpoint at 72). The trainer now always saves the final
population after training. Existing runs may still lack a final checkpoint; use
the last saved checkpoint and the route archive rather than assuming the final
generation is recoverable.

### 2. One-shot rebalancing does not preserve the live distribution — high

The 0023 continuation started from a rebalanced population containing 3 speed /
56 median / 154 slow finishers. After 10 generations, the archive had grown to
364 routes / 60 clean / 54 clusters, but the live population was 0 speed / 7
median / 224 slow / 153 exploration. NEAT selection culled the speed tail.

**Fix:** rebalance every 2–3 generations, or protect the speed/median genomes
during reproduction. Do not treat a single rebalanced checkpoint's final live
population as a deployment portfolio. The archive can still provide the
genomes; use the fixed rebalancer and rebuild the live population.

### 3. NEAT can return `pop_size - 1` after reproduction — resolved

NEAT species/survival rounding can leave the next population one genome short
(0024 produced 383 from a 384 target). With `--parallel 8`, the evaluator then
fails because 383 is not divisible by 8. The trainer now repairs the population
to `config.pop_size` after restore and after every generation by cloning/mutating
a parent; it also re-speciates after the repair.

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

- 1-8 powerup routes are now re-recorded through the one-pass capture and
  replayed/evaluated in `lean_stick` (0105/0106/0118); 1-9+ still need the same
  route-level validation, and the powerup/world-state unit test alone is not a
  route gate.
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

The PPO clean-start plan explicitly requires this route split before training and
excludes holdout routes from the train pool. `tools/filter_routes.py` now
performs route clustering, per-cluster caps, and whole-cluster train/holdout
splitting. `tools/build_start_pool.py` consumes those files and preserves the
split when expanding checkpoint snapshots.

Measured on the current 452-route archive (2026-10-28):

```text
95 clusters / 216 selected routes
158 train routes / 58 holdout routes
13 train states / 10 holdout states after grounded/time-to-exit gates
10 of 10 holdout state keys also occur on train routes
```

The pool split is now integrated, but the data has **zero state-unseen holdout
keys**. Do not call the holdout state result an unseen-route or unseen-state
result. A valid holdout needs route segments or checkpoint keys that do not occur
on train routes. See [`experiments/ppo-clean-start.md`](experiments/ppo-clean-start.md)
and [`audit/2026-10-28-clean-route-reservoir.md`](audit/2026-10-28-clean-route-reservoir.md).

### 5. Evaluation launches one Godot process per spawn — resolved

`run_eval_episode()` was removed. `evaluation/rollout.py` now uses one
persistent `GodotEnv` per canonical trace or start set and calls
`reset(spawns=...)` once, so a 23–45-spawn eval is one cold launch instead of
23–45. `tools/evaluate.py` is the only evaluation entry point.

### 6. Training uses a sequential `DummyVecEnv` — resolved

`envs/factory.py` uses `DummyVecEnv` only for `n_envs == 1` and
`SubprocVecEnv(start_method="spawn")` for larger runs. PPO and DQN share it.

### 7. Shared environment helpers partially extracted — resolved

The shared layers are now:

```text
experiments/runner.py       config resolution and the common CLI
envs/factory.py             env, vec env, eval env, ports, isolated HOME
observations/lean_io.py     lean preprocessing load/create
evaluation/rollout.py       canonical/pool rollout and summaries
data/pools.py               start-pool loading
data/route_starts.py        DQN route extraction
```

PPO and DQN are thin adapters (154 and 116 lines). Neither imports the other.

### 8. The default level-analysis path is machine-specific — resolved

`DEFAULT_ANALYSIS` was removed. The path comes from `[data].level_analysis` in
the experiment manifest, resolved relative to the repository root, and
`tools/extract_route_starts.py` fails with a clear message if it is missing.
Legacy tools may still embed absolute paths; do not copy those defaults into
new manifests.

### 9. Doc/code drift inside the DQN plan — low

- The doc says “infer facing from the next path cell”; the code records the
  actual `facing_direction` from the replay.
- The doc says “validate the checkpoint in the game”; the code never checks
  spawn acceptance.
- The doc says “reset from the pool using the start-jump convention”; that
  convention is retired. Training reset force-starts the run; mid-route states
  need a restored clock, world state, and held powerups, with no injected
  action.
- The clean-start target is raw/lean state, not 794. Do not treat the old 794
  prefix pool as a clean-start source.

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

### 12. Game provenance can drift from the manifest

The game repo is on `prepare-for-expo`, currently `50dcc66` (clean). The active
experiment manifests record `fe7821d`; the two commits after it (`editor` fix,
Steam auth tool) do not change the training/observation code, but every future
result should record the actual HEAD and dirty state. The old caveat about
`360b055` is historical. Use `tools/record_provenance.py` or compare the
manifest commit at run start; do not infer provenance from the manifest alone.

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

### 20. Test coverage is still thin

`tests/` now covers the static/columnar info protocol, binary observation
decoding, preprocessing, the lean MLP, route filtering, start-pool construction,
and experiment manifests. It still has no game-side integration test for the
DQN environment or for reset/spawn behavior.

### 21. `src/plaiground/legacy/processing/` and the legacy SB3 wrapper are not active

`src/plaiground/legacy/processing/` and `src/plaiground/envs/stable_baselines_wrapper.py` are tutorial /
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
encoder; normalization stays game-side, one-hot stays Python-side. The Python
lean encoding is implemented in Godot: 49 primary tile IDs + 49 rotations + 49
signed phases + 9 player + collisions + flow = 188 values. Python decodes it
and one-hots tile IDs, `state_id`, powerup slots, and collision types; the
model width is vocabulary-dependent. The raw 342-value vector remains usable
directly and has passed a 1-1 DQN smoke. Card respawn/moving-platform phase and
driver `--preprocessing lean` wiring are still pending. The reset/powerup/world-state restore
is implemented. The 58 runtime encoder stays in Godot. The default reward is now the
simplified game-side five-term contract (fixed 500 finish, clamped `gold_ratio`
time term); `info` also exposes `gold_ratio`, `flow_cost`, `flow_in_bounds`, and
`wasted_air_frames` so Python can build a custom reward. The clean-start
one-dict reset and dynamic `world_state` snapshot are implemented; the NN state
must not carry reset/world-state metadata. See
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md)
and
[`audit/2026-10-28-lean-state-encoding.md`](audit/2026-10-28-lean-state-encoding.md).
The old `reward_profile` values are ignored. See
[`reference/reward-contract.md`](reference/reward-contract.md). Do not add new
game-side feature engineering before this refactor finishes.

### 23. PPO/DQN harness is config-driven and shared — resolved

The active reinforcement surface is now:

```text
ppo.py                    154 lines  model + learn
dqn.py                    116 lines  model + learn
experiments/runner.py     356 lines  config -> run paths/overrides
envs/factory.py           159 lines  env/vec-env/eval-env construction
observations/lean_io.py   131 lines  lean preprocessing
evaluation/rollout.py     492 lines  one canonical/pool evaluator
data/route_starts.py      505 lines  NEAT checkpoint extraction
tools/evaluate.py          41 lines  shared evaluation CLI
tools/extract_route_starts.py 91 lines
```

The CLI is `--config`, `--timesteps`, `--resume`, `--output-dir`, `--show`,
`--device`, `--verbose`. The manifest is the source of truth.

Removed: the PPO-specific CLI/callbacks/env/eval/preprocessing modules, the
PPO multi-agent `--agents-per-process` branch, DQN `PatienceCallback` and
`run_eval_episode`, in-training eval/patience, and the custom episode CSV. New
runs use `VecMonitor`'s `monitor.csv` and SB3's `CheckpointCallback`.

There is **no automatic best-checkpoint selection during training** now. Train
a fixed budget, then evaluate selected checkpoints with `tools/evaluate.py`.
If selection is needed later, add one shared callback rather than algorithm
specific ones. See
[`audit/2026-10-28-training-harness-composition.md`](audit/2026-10-28-training-harness-composition.md).

### 24. Canonical eval is one deterministic trace; policy robustness still needs work — medium

The old canonical evaluator reset 100 agents to the same canonical start and
reported identical deterministic trajectories as 100 independent trials. It now
reports **one deterministic trial**. The 400k horizon checkpoint is **1/1 at
2.15 s** canonical, 12/12 train, and 8/8 holdout on the corrected pool.

This fixes the measurement, not the policy. The 575k checkpoint enters a
deterministic jump-loop and times out; it diverged from the 400k policy at
canonical step 5. Do not treat `2.15 s` as a 100-trial robustness result.
Re-run with
`tools/evaluate.py --config experiments/0002_ppo_clean_start_1_1_horizon_1m --model runs/ppo_clean_start_1_1_horizon_1m/checkpoints/ppo-400008.zip`.

The old train pool also contained non-standable state starts `(7,7)` and
`(24,10)`. The builder now rejects them, the runtime env rejects a silent spawn
fallback, and the corrected pool has 12 train / 8 holdout starts. The old pool is
at `runs/neat_clean_450/pool_pre_standability/`.

See
[`audit/2026-10-28-ppo-timeout-root-cause.md`](audit/2026-10-28-ppo-timeout-root-cause.md).

For 1-6, the deterministic 1.705M checkpoint is 4.5667 s (repeatable at speedup
12 and 16), but the 0094 2% exploration anchor has 155 finishes under 4.28 s
(including 96 at 3.3333 s). The canonical trace is not a reliability result;
report the deterministic gate separately from the stochastic distribution.

### 25. `--speedup` below 1.0 is silently 1x — low

The game's `_apply_speedup()` clamps `--speedup` to `[1.0, 16.0]` and returns
early when the result is `<= 1.0`
(`../dragon-jump-remaster/godot/src/training/synchronizer.gd`). A show manifest
with `speedup = 0.5` (0078) or `0.25` therefore runs at normal speed; there is
no slow-motion path today. True slow motion needs a game-side branch that keeps
`Engine.physics_ticks_per_second` at 60 and only lowers `Engine.time_scale`;
lowering the tick rate instead changes the per-tick simulation.

### 26. NEAT fitness does not rank fast finishers first — medium

The raw game reward is dominated by the finish bonus and flow/efficiency
shaping. In the 1-5 archive, the 4.6833 s route scores 561.08 while a 7.1167 s
route scores 570.93 and an 18.72 s finisher scores 561.70. Reproduction therefore
selects slow finishers; rebalancing only archives the fast genome. Use
`--time-fitness-weight W` (default 0) to subtract `W * run_time` from finisher
fitness. At W=20 the 4.68 s route ranks above the 7.12 s route. Do not confuse
the fitness-best genome with the fastest route.

### 27. Changing action_repeat breaks the old-level anchor — high

The 1.555M chain model was trained at `action_repeat=4`. Re-collecting the
1-1..1-4 anchor at `action_repeat=2` (0083) gives 2,229 episodes, median 3.25 s,
best 1.6167 s, and 409 timeouts. The 0082 deterministic 1.58M eval then failed:
1-5 0/1 (death), 1-4 timeout, 1-1 2.9333 s. Do not mix 4-frame anchor
transitions with a 2-frame online env: the reward horizon and the next-state
delay differ. Use one action repeat for the online env, the expert replay, and
the anchor. The 2-frame expert was archived; 0085 retried at 4 frames.

Also, `collect_dqn_buffer.py` without `--rare-buffer` writes a plain
`ReplayBuffer`; the DDQN trainer rejects it as an anchor. Use
`--rare-buffer --gap-quantile 0.9`.

### 28. Single-model 1-5 training damages 1-4 and late 1-2 — high

The 0085 final 1.655M checkpoint gives **1-5 3.1667 s**, 1-1 1.5667 s, and 1-3
3.1667 s, but **1-4 fails** (one death + one timeout) and **1-2 regresses to
5.3500 s**. The 1-5 schedule has only one 1-4 env and the old anchor has no 1-4
transitions. Use the 1.555M chain model for 1-1..1-4. A 25k retention fine-tune
with 2 x 1-4 and 2 x 1-2 plus a 1.555M all-level anchor (0088) was tried: its
final restored 1-2 (3.0167 s) and 1-3 (3.2000 s) but still timed out on 1-4 and
pushed 1-5 to 3.5000 s, outside the 3.41 s gate. Do not repeat that recipe; a
single-model attempt needs a frozen 1-5 head or a dedicated 1-4 expert buffer.
Keep the 0085 final as the 1-5 fallback; do not use it for 1-2 or 1-4.

### 29. `collect_dqn_buffer` “best” includes deaths — medium

The 1-6 anchor summary reported `best=2.4333 s`, but that row is
`finished=False`, length 35, reward -48.49: a death. `_fmt_stats` took
`min(run_time)` over all canonical episodes without filtering on `finished`, so
the 2.43 s death was presented as a fast route. The 2.6–2.8 s entries in the
1-6 retrain monitor are also deaths. The collector now reports
`best_finished_time`, `best_death_time`, `canonical_finished`, and
`canonical_deaths`, and computes medians/speed bins from finished episodes only.
The 1-6 anchor's 3.0667/3.2/3.3333 s entries are **real finished episodes** under
2% exploration, not deaths; they are stochastic and are not deterministic
checkpoints. The existing anchor metadata still contains the old
`best=2.4333`; treat it as the death value. See
[`audit/2026-10-29-1-6-ddqn-investigation.md`](audit/2026-10-29-1-6-ddqn-investigation.md).

### 30. Checkpoint selection is manual and finals regress — high

`dqn_final.zip` is the last save, not the best model. 0091's final is 7.25 s
while 1.705M is 4.5667 s; 0060's final was worse than 1.3M; 0095's final is
6.3833 s (best of that run, but worse than the 0091 1.705M source). The first
0091 selection skipped 1.705M entirely and used 1.68M (4.6833 s). Evaluate
**every** saved checkpoint with `tools/evaluate.py`, then record the selected
path in the experiment README, `docs/status.md`, and
`docs/reference/level-ledger.md`. See
[`audit/2026-10-29-run-reliability.md`](audit/2026-10-29-run-reliability.md).
`tools/level.py data` prepares the pools but deliberately does not select a
checkpoint.

### 31. The 1-6 fast mode is stochastic; the 4.07 s replay is not reproducible — high

The 0094 1-6 anchor has 1,697 finishes; 155 are under 4.28 s, including 96 at
3.3333 s and 19 at 3.2 s. These are real finishes from the 2% exploration
policy, but no deterministic checkpoint selects them. More importantly, the
0095 replayed routes are not stable in isolation: a second replay gave 4.7167 s
for the 3.1833 s route, 5.0500 s for the 3.9833 s route, and a death at 2.7167 s
for the 4.0667 s route. Route replay order/RNG carries over between routes.
**Do not use the 3.3/4.07 replays as deterministic teacher actions.** The stable
NEAT route is 4.50 s; the 0099 any-start expert is 4.60 s canonical. See
[`audit/2026-10-29-1-6-ddqn-investigation.md`](audit/2026-10-29-1-6-ddqn-investigation.md).

### 32. 1-6 had no state-start pool — addressed

The original 1-6 pool was canonical-only because `runs/level_analysis/1-6.json`
was missing. It is now exported. Expanding all 263 clean 0093 routes gives
**32 unique state starts**, not 11: the old 0090 pool only used 22 routes.
0097/0099 trained on canonical + 31 state starts and selected the 1.815M
checkpoint, which finishes **32/32** with 0 deaths/timeouts (median 6.0417 s,
canonical 4.6000 s, spawn_ok 32/32). The 1.705M canonical reference covers only
11/32, confirming the old gap. Use
`runs/neat_speed_1_6_from_0090/pool_state_70/` for the current 0.7/0.3 pool and
`pool_state_50/` for the 0.5/0.5 training pool.

### 33. Resumed DQN ignored the manifest learning rate — fixed

SB3 restores the optimizer and learning-rate schedule from a checkpoint.
Before 2026-10-29, `run.resume_from` kept the source run's LR even when the new
manifest declared another value (0095's 5e-5 ran at 1e-4). `dqn.py` now applies
the manifest LR after loading, rebuilds the schedule, updates optimizer groups,
and 0099 logs `learning_rate 5e-05`. Use a fresh output directory when testing
a new LR; checkpoints retain their optimizer state. The exploration epsilon
schedule still follows the absolute timestep counter: a run resumed at 1.8M
with `exploration_fraction=0.5` reports `current=0.02`, not a fresh 0.10 ramp.
0103 therefore ran expert/replay-driven at 2% epsilon; a true exploration ramp
needs a schedule reset or an explicit epsilon override.

### 34. 1-7 has no strict-clean route archive — near-clean gate used

The 1-7 transfer (0101) and the high-mutation cleanup (0102) both produced many
successful routes but **zero strict-clean routes** under the standard
`revisit_ratio <= 0.10` filter. The best 1-7 route is **7.0333 s** with revisit
0.107, backtracks 1, and action flips 0.048. The accepted 1-7 gate therefore
uses near-clean routes (`revisit_ratio <= 0.20`, `backtracks <= 2`, action flips
<= 0.75), giving 10 grounded starts. The 20-state all-success diagnostic
includes two timeouts from a high-revisit route (`8f687...`, revisit 0.43) and
is not the clean gate. Do not describe the 18/20 all-success result as the
accepted clean-start result; use the 10/10 near-clean result and record the
strict-clean exception explicitly.

### 35. 1-8 clean gate and the high-revisit all-success timeout

Unlike 1-7, 1-8 produced strict-clean routes: 26 routes / 11 clusters, best
4.4167 s. The selected 0118 **1.960392M** expert covers **40/40** strict-clean
states with 0 deaths/timeouts and canonical **4.4667 s**. The broader
all-success diagnostic is **106/107**, with one timeout at `[16,21]` facing `1`
from route `36a6b1e1...` (revisit 0.226, not strict clean). Do not use the
106/107 all-success result as the clean gate; the clean gate is 40/40. The
fastest overall NEAT route (4.4167 s) is in holdout cluster `c000`; the expert
reproduces the fastest train route (4.4667 s). That is a coverage/speed split
tradeoff, not a model-selection failure. The final 1-8 target is still open.

### 36. Canonical timing can be one frame above the provisional gate

The 0109 repair reached full 1-8 clean coverage at canonical 4.5167 s, which is
0.0023 s above the provisional 4.5144 s gate. The 0118 low-LR nudge fixed it to
4.4667 s without losing coverage. If a future level lands within one frame of a
provisional gate, prefer a short low-LR canonical nudge over accepting the
near-miss or retraining from scratch.

### 37. Route tiers and eval-pool unions were manual — resolved

The 1-7/1-8 runs produced `strict_clean_routes`, `near_clean_routes`,
`all_success_routes`, and `eval_*_start_pool.json` with ad-hoc snippets. The
`clean`/`quality` fields were inconsistent, and the eval union was a manual
train+holdout merge. `tools/level.py data` now recomputes strict/near/all tiers
from the archive, filters the accepted tier, builds the pools, and writes the
deduplicated eval union. It reproduces the 1-8 result exactly (26 strict, 37/14
states, 40 eval) and the 1-7 near-clean fallback (10/10). It also warns when the
fastest cluster is holdout-only. The checkpoint sweep remains manual; see
known issue #30 and the runbook.

### 38. 1-9 is a single-route expert, not an any-start or retention model

The user confirmed 1-9 is a precision map with one route to optimize, so 0122
was trained on the fastest clean route only (`926883...`, 2.9500 s) with
canonical-only resets. The 11-route / 6-cluster pool from 0121 is retained as a
diagnostic, not as an accepted start gate.

The selected 1-9 checkpoint (`ddqn_2060360_steps.zip`) is a 1-9-only fine-tune
of the 1-8 expert. Deterministic canonical 1-1..1-9 retention is partial:

```text
finish:     1-1 2.5167, 1-3 6.0167, 1-9 2.9500
pass gate:  1-1, 1-9 only
fail:       1-2, 1-4, 1-5, 1-6, 1-7, 1-8
```

Do not use the 1-9 checkpoint as evidence of old-level retention. A shared
student still needs the per-level teachers and a dedicated retention/distillation
evaluation.

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

## Held-stick action semantics in exported models — high

`legacy_stick` 61 and `lean_stick` 191 policies require
`reassert_held_jump=true`. Python now defaults that from the observation
encoding, keeps training/evaluation/preview on one `Run.reassert_held_jump`
property, and fails fast when Godot reports a different value.

**Godot export TODO:** when the bundled/inference policy is a held-stick model,
`PlayerAIModelController` must set `reassert_held_jump` from the exported model
metadata (or the held-stick encoding), not only `obs_encoding`. Otherwise an
exported 1-4 policy can be evaluated or shipped with legacy edge-triggered jump
semantics. Add an export gate for a held-stick model that checks the flag and a
short ceiling-stick replay.
