# Status

**Updated:** 2026-10-28
**Branch:** `neat`
**HEAD:** `707b745` — `lean inputs`
**Working tree:** the repository restructure (`src/plaiground/`, `scripts/`,
`tools/`, `experiments/`, `Makefile`, package metadata), the clean-start plan
and audits, the lean encoder, route filtering/start pools, reward/start
changes, and the active PPO run docs are uncommitted.
**Game source:** `../dragon-jump-remaster` (`prepare-for-expo`, HEAD `adb9278`
`more state changes`; the raw/lean observation, force-start, reward,
terminal-death, and clean-start world-state code is in HEAD; game docs are
dirty with the new lean decision/status updates. World-state/powerup restore
and `info_request` are implemented, but the clean-start pool/eval gates are
not yet run.)

> This page is the current state. Historical sections in `README.md` and older
> experiment docs are not authoritative. When they disagree, this page and
> [`index.md`](index.md) win.

## Current focus

**Repository restructure (2026-10-28).** The code is now split into an importable
`src/plaiground/` package, thin `scripts/`, data/evaluation `tools/`, and
committed `experiments/` manifests. The package is installed editable with
`uv pip install --python .venv/bin/python -e . --no-deps`. The 1M PPO process
that was running during the move has since completed; new invocations use
`scripts/reinforcement/01_ppo.py`. The old-to-new
mapping and verification are in
[`audit/2026-10-28-repository-restructure.md`](audit/2026-10-28-repository-restructure.md).

**DQN route-spawns (2026-10-28, design pivoted).** The 794 `multi_hot_sensor`
prefix run is now a historical baseline, not the clean-start target. The
simplified reward, force-start reset, and terminal spike death are shipped.
The prefix baseline was stopped at ~129.5k/200k for the clean-start design
discussion; checkpoints 25k–125k exist, with no final model or eval. The
clean-start core is now implemented: raw state, source-checkpoint time, held
powerups, dynamic `world_state`, one-pass NEAT capture, and a canonical/state
pool. The next step is full clean-start validation and evaluation. See
[`experiments/dqn-route-spawns.md`](experiments/dqn-route-spawns.md), the
pipeline review in
[`audit/2026-10-28-dqn-pipeline-review.md`](audit/2026-10-28-dqn-pipeline-review.md),
and the clock-bias audit in
[`audit/2026-10-28-start-pool-clock-bias.md`](audit/2026-10-28-start-pool-clock-bias.md).

**PPO clean-start baseline (2026-10-28, complete; 225k selected).** The clean
1M attempt collapsed around 296k and is preserved in
`runs/ppo_clean_start_1_1_failed_1m/`. A stable continuation resumed from
`ppo-200000.zip` with `lr=1e-4`, `target_kl=0.02`, `ent_coef=0.02`, and 25k
checkpoint/evaluation cadence. It stopped at 300k on patience and selected the
**225k** checkpoint in `runs/ppo_clean_start_1_1/`.

Deterministic results (old clone evaluation: one canonical trace repeated 100
times, not 100 independent trials):

```text
canonical:    100/100 clones, median 2.3167 s
train pool:   14/14 finishes, median 3.5583 s, best 2.1500 s
holdout pool: 10/10 finishes, median 5.2250 s, best 3.4167 s
```

The policy uses the Godot **lean 188-value** packet, Python one-hot
preprocessing (188 → 954), and canonical 0.70 / train-state 0.30 sampling. It
deliberately does **not** use 794 or prefix replay. The holdout is
route-cluster-level only; it is not state-unseen. The model is not installed.
Compact JSON/distillation options are in
[`experiments/compact-policy-deployment.md`](experiments/compact-policy-deployment.md).

**PPO clean-start horizon test (2026-10-28, stopped at 972.6k; 400k selected).**
Experiment 0002 resumed the 225k policy and ran to 972,600 timesteps. It found a
new best at **400k: 2.1500 s canonical**, then oscillated between the healthy
2.15 s mode and deterministic jump-loop timeouts. The run was stopped manually
for investigation; no retraining was performed.

The timeout investigation found:

- canonical evaluation was running 100 identical deterministic clones and
  reporting them as 100 independent trials;
- the 575k policy diverged from the 400k policy at canonical step 5, jumped into
  a wall, and looped until timeout;
- the old pool contained non-standable state starts at `(7,7)` and `(24,10)`.

Fixes now in place:

- canonical evaluation reports **one deterministic trial**;
- the pool builder rejects non-standable cells, and the runtime env rejects a
  silent spawn fallback;
- `tools/trace_ppo.py` records per-step action/state traces.

The corrected 400k evaluation is canonical **1/1 at 2.15 s**, train pool
**12/12**, holdout pool **8/8**. See
[`../experiments/0002_ppo_clean_start_1_1_horizon_1m/`](../experiments/0002_ppo_clean_start_1_1_horizon_1m/)
and
[`audit/2026-10-28-ppo-timeout-root-cause.md`](audit/2026-10-28-ppo-timeout-root-cause.md).

**Lean training harness (2026-10-28, complete).** Both reinforcement trainers
are now config-driven and share the non-algorithm plumbing:

```text
scripts/reinforcement/01_ppo.py     train only, 7 flags
scripts/reinforcement/02_dqn.py     train only, 7 flags
tools/evaluate.py                   one evaluator for PPO and DQN
tools/extract_route_starts.py       DQN route extraction

ppo.py                    154 lines  model + learn
dqn.py                    116 lines  model + learn
experiments/runner.py     356 lines  config -> run paths/overrides
envs/factory.py           159 lines  env, vec env, eval env, ports, HOME
observations/lean_io.py   131 lines  load/create lean preprocessing
evaluation/rollout.py     492 lines  canonical/pool rollout + summaries
data/route_starts.py      505 lines  NEAT checkpoint extraction
```

The command surface is now `--config`, `--timesteps`, `--resume`,
`--output-dir`, `--show`, `--device`, `--verbose`. The manifest supplies the
level, environment shape, observation encoding, model, hyperparameters,
checkpoint cadence, data paths, and evaluation sets.

Removed from the active path:

- `ppo_cli.py`, `ppo_callbacks.py`, `ppo_env.py`, `ppo_evaluation.py`, and
  `ppo_preprocessing.py`;
- the PPO multi-agent `--agents-per-process` branch;
- DQN `PatienceCallback`, `run_eval_episode`, and the in-trainer extractor;
- the custom episode CSV callback and per-eval checkpoint writer. `VecMonitor`
  writes `monitor.csv`, and SB3's `CheckpointCallback` writes checkpoints;
- in-training eval/patience. Evaluation is a post-run tool, which matches the
  `eval_every = 0` practice used by the active PPO runs.

Commands:

```bash
.venv/bin/python scripts/reinforcement/01_ppo.py \
  --config experiments/0002_ppo_clean_start_1_1_horizon_1m
.venv/bin/python tools/evaluate.py \
  --config experiments/0002_ppo_clean_start_1_1_horizon_1m \
  --model runs/ppo_clean_start_1_1_horizon_1m/checkpoints/ppo-400008.zip \
  --tag 400k
```

DQN has a matching manifest at
[`../experiments/0003_dqn_clean_start_1_1/`](../experiments/0003_dqn_clean_start_1_1/).
The 400k result remains the acceptance test for evaluation parity
(canonical 1/1 at 2.15 s, train 12/12, holdout 8/8). The suite is 55 tests.
See
[`audit/2026-10-28-training-harness-composition.md`](audit/2026-10-28-training-harness-composition.md).
The 0002 training process is stopped.

The collection run stopped at generation 25 with **452 unique clean routes**
and 1,486 checkpoint files. Filtering at `jaccard=0.5`, `max-per-cluster=8`, and
`holdout-fraction=0.25` produced **95 clusters / 216 routes**: 158 train and 58
holdout routes. Checkpoint expansion through the grounded/state/time-to-exit
gates produced only **13 train states and 10 holdout states** because most
snapshots are mid-air. All 10 holdout state keys also occur on train routes, so
this is a route-cluster split, not a state-unseen holdout.

`src/plaiground/data/route_filter.py` + `tools/filter_routes.py` perform the
path filter; `src/plaiground/data/start_pool.py` +
`tools/build_start_pool.py` build the train/holdout pools. The PPO manifest
selects the lean 188-value encoding and Python preprocessing; the fitted
pipeline is saved with the run and applied in canonical and start-set
evaluation.
See [`../experiments/0001_ppo_clean_start_1_1/`](../experiments/0001_ppo_clean_start_1_1/),
[`experiments/ppo-clean-start.md`](experiments/ppo-clean-start.md), and
[`audit/2026-10-28-clean-route-reservoir.md`](audit/2026-10-28-clean-route-reservoir.md).

**Clean-start implementation (2026-10-28).** The one-dict reset, dynamic
`world_state` snapshot, held-powerup restore, and NEAT-side state rejection are
implemented. The game now exposes `Level.capture_world_state()` /
`restore_world_state()`, `Powerup.hold()`/`apply_state()`,
`Player.restore_powerup()`, and an on-demand `info_request` / `world_state`
block. `scripts/genetic/01_neat.py` records checkpoint snapshots + world state during the
population; the historical clean-archive coverage tool preserved the checkpoint file;
`src/plaiground/data/route_starts.py` +
`tools/extract_route_starts.py` build the canonical 0.70 / state 0.30 pool without
replay when checkpoints exist. Validation so far: game AI suite 37/37,
training reset/spawn suites 9/9 (the visuals test requires a truly headless
GdUnit runner), Python suite 28/28, a 192-agent NEAT evaluation produced
87 finishes and 87 checkpoint files, and a raw-state 1-1 DQN smoke trained from
the resulting 10-state pool. Training quality/promotion is still pending. See
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md)
and [`experiments/dqn-clean-start.md`](experiments/dqn-clean-start.md).

**Reward/start/death shipped (2026-10-28).** The training reward is now the
simplified five-term contract: progress, efficiency, gold-ratio time pressure
(`-3.75/s * (1 - gold_ratio)`, `gold_ratio` clamped 0–3), fixed `+500` finish,
and `-50` death. `previous_best` is telemetry only; `reward_profile` is ignored.
Training reset force-starts the run (no idle, no forced first jump). Spike
contact is terminal at the hazard; the next reset uses the normal sampled start,
not canonical. Raw `info` adds `gold_ratio`, `flow_cost`, `flow_in_bounds`, and
`wasted_air_frames` for Python rewards. See
[`reference/reward-contract.md`](reference/reward-contract.md).

**Observation path (2026-10-28).** The raw local-state packet is implemented
and validated: 342 values (7×7 tile IDs + tile states, player state, 8-way
collisions, 8-way flow). The raw player state now carries `gold_ratio` instead
of the old normalized run clock. Godot normalizes units; Python builds the
feature encoding. The 794 sensor path is unchanged but is not the clean-start
target. The one-dict reset, dynamic `world_state` snapshot, held-powerup
restore, and NEAT validation are implemented; the raw 342-value clean-start
state trains end-to-end. [`src/plaiground/observations/`](../src/plaiground/observations/) now
provides the lean-only `StatePipeline` path: `DecodeLeanState` and
`LeanFeatures` for the game's 188-value packet. Godot emits 49 primary tile IDs
+ 49 rotations + 49 signed phases + 9 player + 24 collision + 8 flow; Python
one-hots tile IDs, `state_id`, powerup slots, and collision types, so the model
width is vocabulary-dependent. Game-side primary-symbol priority is implemented;
card respawn and moving-platform phase are still pending. The lean v1 baseline
uses Python one-hot features with the declared `src/plaiground/models/mlp.py` `LeanMLP` 512 × 512
architecture and `lean_policy_kwargs()`; there is no custom extractor. The lean path is wired
into `scripts/reinforcement/01_ppo.py`; the game-side raw 342 encoding remains
available for diagnostics, not as the active preprocessing path.
See
[`audit/2026-10-28-raw-state-observation-decision.md`](audit/2026-10-28-raw-state-observation-decision.md),
[`audit/2026-10-28-lean-state-encoding.md`](audit/2026-10-28-lean-state-encoding.md),
and
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md).

**794 sensor PPO lineage (most recent working pipeline).** PPO with the
794-value `multi_hot_sensor` encoding has 1-1 and 1-2 checkpoints and a 1-3
transfer attempt. 1-1 canonical is **2.1333 s** with full 1-1 strata coverage;
1-2 canonical is **3.0667 s** with full coverage; 1-3 is **not solved**
(canonical 0/1, medium 25/46, hard 5/27). A 1-3 distilled candidate reaches
**3.9 s** canonical with 43/46 medium and 25/27 hard, but is not installed.

**754 multi-hot PPO reference.** The 200k multilevel checkpoint
(`best_multilevel_3maps.zip`) finishes all 242 starts across 1-1/1-2/1-3 at
canonical **1.6667 / 2.9667 / 5.2333 s**. The 300k continuation kept coverage
but regressed canonical to 1.6667/3.0000/5.5667, so 200k remains selected. Not
installed; the shipped JSON runtime is 58-only.

**Frozen game baseline.** The bundled `student_obs58_256.json` remains the
convention AI baseline: 16/16, 100/100 per level, zero deaths, **78.583 s**
campaign sum, 1-1 **2.333 s**. Do not replace it without the all-16 in-game
gate. See [`reference/deployed-model-58.md`](reference/deployed-model-58.md).

## Verified results

These are from the run artifacts and the game-side docs; they are not
re-measured here.

| Track | Level(s) | Result | Artifact |
|---|---|---|---|
| NEAT 8×24 | 1-1 | **2.15 s** canonical; 460 clean routes; 59/60 starts (51/52 meaningful) | `runs/neat_1_1_8x24_archive/` |
| NEAT transfer | 1-2 | **3.0167 s**; 12 clean routes; 71/72 starts | `runs/neat_1_2_transfer_8x24/` |
| NEAT transfer | 1-3 | **3.5667 s**; 26 clean routes; 62/86 starts | `runs/neat_1_3_transfer_8x24/` |
| 754 PPO multilevel, 200k | 1-1/1-2/1-3 | **1.6667 / 2.9667 / 5.2333 s**; 60/60, 84/84, 98/98 starts | `runs/ppo_multilevel_1_1_1_2_1_3/` |
| 754 PPO multilevel, 300k | 1-1/1-2/1-3 | 1.6667 / 3.0000 / 5.5667 s — worse than 200k | same run dir |
| 794 sensor PPO | 1-1 | **2.1333 s** canonical; full 1-1 strata | `runs/ppo_fresh_1_1_sensor_794/` |
| 794 sensor PPO | 1-2 | **3.0667 s** canonical; full 1-2 strata | `runs/ppo_sensor_1_2_from900k/` |
| 794 sensor PPO | 1-3 | not solved: canonical 0/1, medium 25/46, hard 5/27 | `runs/ppo_sensor_1_3_from1_2/` |
| 794 distilled | 1-3 | **3.9 s** canonical; medium 43/46, hard 25/27, easy 24/24 | `runs/distill_1_3_merged_available/` |
| Deployed 58/256 | 1-1…1-16 | 16/16, 100/100, 0 deaths, **78.583 s** sum; 1-1 2.333 s | `godot/src/resources/ai/student_obs58_256.json` in the game repo |

**2026-10-28 contract validation:**

- Game reward tests: **16/16 pass** (force-start, fixed finish, clamped
  `gold_ratio`, per-second time term, terminal spike death, raw telemetry).
- Raw observation check: 1-1 and 1-8, **342 values**, cell matches 10/10 and
  20/20, raw `gold_ratio` matches `info`.
- 58 and 794 observation regressions: 1 env × 200 steps, no errors.
- 10k DQN smoke on the clock-safe pool with force-start/new reward: 116
  episodes, `ep_rew_mean` **≈438**, no crash. A 10k deterministic strata eval
  finished **42/44** starts (canonical 1/1 at 3.43 s; medium 27/28; hard
  14/15; no model promotion). Old start-jump checkpoints are out of
  distribution.
- 10k PPO 1-env smoke runs end-to-end; canonical eval is **0/100** at 10k
  (too short to judge; no crash). Use a longer run before comparing.


## DQN status (2026-10-28)

- `scripts/reinforcement/02_dqn.py` trains from a config manifest;
  `tools/extract_route_starts.py` builds the pool and `tools/evaluate.py`
  evaluates checkpoints. The 794 path is a legacy baseline; the clean-start
  target is raw state. The 58-value cell extractor is archived/fallback only.
- `runs/dqn_startpool_1_1_smoke/` — 10k smoke on the clock-safe pool
  (canonical 0.7 / prefix 0.3): 61/93 finishes, no cell resets.
- `runs/dqn_startpool_1_1/` — prefix-replay baseline. Stopped at ~129.5k/200k
  on 2026-10-28 for the design discussion. Checkpoints 25k–125k exist; no
  final model or deterministic eval. Training stats: 79% finish, last 200
  97.5%, median 4.0 s, best 1.62 s stochastic. Do not quote the 1.62 s as a
  result.
- `runs/dqn_route_spawns_1_1/` — original extractor attempt; `spawns/` is
  empty. The first extract stopped before writing the pool.
- No clean-start run exists; the plan is
  [`experiments/dqn-clean-start.md`](experiments/dqn-clean-start.md).

## Next

1. **Decide scope.** The game vault currently says DQN/DDQN is deferred until
   after the convention (A32/A33, decisions 2026-10-27). This repo says “active
   experiment.” Record the decision in the game vault or keep DQN explicitly
   side-only.
2. **Raw-state path.** The Godot packet is implemented and validated (342
   values, 1-1 and 1-8) and now includes `gold_ratio`. Add the Python lean
   encoder and parity/throughput gates, then cut over. See
   [`audit/2026-10-28-raw-state-observation-decision.md`](audit/2026-10-28-raw-state-observation-decision.md).
3. **Validate the clean-start contract.** The game world-state/powerup
   restore, one-dict reset, `info_request`, Python pass-through, and NEAT
   one-pass checkpoint capture are implemented. Re-record 1-8+ powerup routes
   and run the full 1-1/1-8 round-trip, strict NEAT rejection, and raw smoke
   gates. Use the source-checkpoint time and the facing/neighbor holdout. See
   [`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md)
   and [`experiments/dqn-clean-start.md`](experiments/dqn-clean-start.md).
   Do not start with cell-only spawns or 794.
4. **Compact the PPO candidate for deployment.** The original 1M run collapsed
   at ~296k; the horizon run found the **400k / 2.1500 s** checkpoint before it
   was stopped. Preserve that teacher and distill it into a smaller flat MLP for
   JSON deployment. Start with `954 → 256 → 256`, then 192/128 if quality holds.
   Review the compact-policy options before building the supervised pipeline.
   The current holdout is route-cluster-only, not state-unseen. See
   [`experiments/compact-policy-deployment.md`](experiments/compact-policy-deployment.md)
   and [`../experiments/0002_ppo_clean_start_1_1_horizon_1m/`](../experiments/0002_ppo_clean_start_1_1_horizon_1m/).
5. **Smoke the new reward/start/death contract.** The 10k 794/prefix DQN smoke
   and 42/44 legacy strata eval validate the shipped reset/reward/death paths,
   not clean-start quality. PPO 10k canonical is 0/100; run a longer smoke
   before 4 envs × 200k.
6. **Recompute the clean-start pool.** Record checkpoint snapshots and
   `world_state` during the population (one-pass), keep the best final time per
   route signature, filter near-duplicate paths with `42`, split whole clusters
   into train/holdout, and build the raw-state pool from the selected routes.
   If capture overhead is too high, use the selected-route replay fallback.
   The old 794/prefix A/B is historical.
7. **Keep the game baseline frozen.** No DQN/PPO/754/794 model is installed in
   the game until the full 16-level and in-game gates pass.
8. **Use the lean harness.** PPO and DQN are config-driven thin adapters; the
   shared evaluator is `tools/evaluate.py`. Do not add in-training eval or
   algorithm-specific callbacks without a concrete need. The old strata
   evaluator, archive-coverage tool, and per-spawn DQN evaluator are removed.
   The 0002 training process is stopped. See
   [`audit/2026-10-28-training-harness-composition.md`](audit/2026-10-28-training-harness-composition.md).

## Known issues

See [`known-issues.md`](known-issues.md) for the living list. The highest-impact
current items are clean-start validation/holdout coverage, the one-tick reset
clock drift, the all-or-nothing extraction, and the checkpoint-vs-route holdout
ambiguity. The old per-spawn evaluation cost and the separate DQN
evaluator/patience surface are resolved; evaluation is now
`tools/evaluate.py`.
