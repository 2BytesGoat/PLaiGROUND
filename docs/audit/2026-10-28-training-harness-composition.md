# Training harness composition and PPO dead-code audit

Status: PPO split and pruned; remaining generic evaluator/selector work deferred  
Date: 2026-10-28  
Branch: `neat`  
PLaiGROUND commit at audit: `af5ce9d` plus the uncommitted baseline-eval fix  
Related: [`../../experiments/0002_ppo_clean_start_1_1_horizon_1m/`](../../experiments/0002_ppo_clean_start_1_1_horizon_1m/),
[`../known-issues.md`](../known-issues.md)

This note records the architecture discussion around
`src/plaiground/algorithms/reinforcement/ppo.py` (pre-split): why it grew to
2,285 lines, what is shared with DQN/NEAT/supervised code, what is dead or
legacy-only, and the composition plan that produced the first split.

No behavior changes were made as part of the original audit. The PPO split and
prune were completed after the 1M run was stopped.

## Final lean-harness update (2026-10-28)

The first split kept the old experiment surface (45 PPO flags, two custom PPO
callbacks, three evaluation paths, and a duplicated DQN evaluator). The final
pass replaced that surface:

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

Removed: `ppo_cli.py`, `ppo_callbacks.py`, `ppo_env.py`, `ppo_evaluation.py`,
`ppo_preprocessing.py`, the PPO multi-agent CLI path, DQN `PatienceCallback`
and `run_eval_episode`, and in-training eval/patience. New runs use
`VecMonitor`'s `monitor.csv` and SB3's `CheckpointCallback`. Evaluation is
post-run and shared through `tools/evaluate.py`.

The tradeoff is explicit: no automatic best-checkpoint selection during
training. If it returns, it should be one shared callback built on SB3's
`EvalCallback`, not algorithm-specific callbacks.

## First implementation update (2026-10-28)

The PPO monolith is now split:

```text
ppo.py                 245 lines  orchestration only
ppo_evaluation.py      454 lines  one canonical/start-pool rollout engine
ppo_callbacks.py       277 lines  episode CSV, patience
ppo_cli.py             152 lines  active CLI parser
ppo_env.py              97 lines  env/vec-env construction
ppo_preprocessing.py   138 lines  lean preprocessor
```

Shared modules now used by both PPO and DQN:

```text
envs/runtime.py    resolve_env_path, isolated_home
data/pools.py      flatten_reset_state, load_start_pool
```

DQN no longer imports `ppo.py` or its private reset helper. The 400k
checkpoint still evaluates at **2.15 s** after the split and prune.

### Prune update

Removed from the active PPO path:

- spawn-pool training/evaluation and `SpawnEvalCallback`;
- legacy CNN/autoencoder encoders;
- student/head initialization and observation expansion/freezing;
- trajectory capture and cosmetic A/B;
- diagnostic stochastic/legacy eval modes and `--start-delay`;
- the literal no-op `--additional-timesteps`;
- the now-unused `ppo_checkpointing.py`.

The CLI is down from 69 to 45 flags. Still separate by design: DQN's env factory,
evaluation loop, and patience callback. The legacy strata evaluator and
`tools/clean_archive_coverage.py` were removed.


## Summary

The 8-line PPO entry point is thin, but the implementation module became the
whole training application:

```text
scripts/reinforcement/01_ppo.py                 8 lines
src/.../reinforcement/ppo.py                2,285 lines (pre-split)
PPO CLI arguments                                 69
```

This is not just a size problem. It is a coupling problem:

- DQN imports PPO to reuse generic environment and pool helpers.
- The same environment-path and isolated-home helpers are copied in two other
  tools.
- PPO and DQN each maintain their own env factory, evaluation loop, and
  patience/best-checkpoint callback.
- NEAT duplicates small history/level helpers.
- A large part of PPO is reachable only through optional legacy flags that no
  committed script or experiment invokes.

The right response is composition of the experiment plumbing, not a giant
`BaseAlgorithm` inheritance hierarchy.

## Measured surface

### PPO file

```text
lines:              2,285
CLI flags:             69
top-level definitions: 32
```

The largest individual responsibilities are already too large:

```text
main                         ~304 lines
parse_args                   ~283 lines
SpawnEvalCallback            ~275 lines
evaluate_policy              ~245 lines
evaluate_start_pool          ~155 lines
```

### What the active lean PPO path uses

The current 0002 command sets 19 unique flags and exercises:

- lean 188-value observation and 954-value preprocessing;
- the 8-process single-agent environment factory;
- train start-pool sampling;
- `EpisodeCSVCallback`;
- `EvalPatienceCallback`;
- deterministic canonical evaluation;
- the main PPO training loop and final start-pool evaluation.

The active path does **not** exercise spawn-pool evaluation, legacy grid
encoders, autoencoders, student/head initialization, input expansion, cosmetic
A/B settings, or trajectory capture.

## What is duplicated or misplaced

| Concern | PPO | DQN | Other copies |
|---|---|---|---|
| Env path + isolated HOME | `resolve_env_path`, `isolated_home` | imports from shared runtime | old tools removed |
| Env factory / vec env | `ppo_env.py` | `make_train_env` | — |
| Start-pool loading | `load_start_pool` | imports from PPO | — |
| Reset-state normalization | `_flatten_reset_state` | imports the private PPO function | — |
| Canonical eval loop | `evaluate_policy` | `run_eval_episode` | — |
| Pool evaluation | `evaluate_start_pool` | `evaluate` | — |
| Patience / best checkpoint | `EvalPatienceCallback` | `PatienceCallback` | — |
| Run-dir setup / logging | PPO-specific | DQN-specific | — |
| `parse_levels` | `ppo.py` | — | `all_levels.py`, `lineage.py` |
| `read_history` / checkpoint generation | — | — | duplicated in two NEAT modules |

The smoking gun is in `dqn.py`:

```python
from . import ppo as PPO

PPO.resolve_env_path()
PPO.isolated_home(...)
PPO.load_start_pool(...)
PPO._flatten_reset_state(...)
```

DQN is using PPO as a utility library, including a private symbol. The shared
code exists, but it lives in the wrong module.

## What is actually dead or legacy-only

### Literal no-op

`--additional-timesteps` is parsed but never read. It does nothing.

### No committed entry point invokes these

A search of committed scripts, configs, and tools found no callers outside
`ppo.py` for:

```text
--spawn-pool
--spawn-eval-every
--grid-encoder
--init-student
--init-head
--expand-from
--eval-legacy-per-agent
--eval-stochastic
--training-no-cosmetics
--trajectory-dir
```

Some appear in old experiment docs. They are still technically reachable
through the CLI, but none is part of the active workflow.

### Approximate legacy budget

| Feature | Approx. lines |
|---|---:|
| Spawn pool/evaluation path | ~350 |
| Legacy CNN/autoencoder/student init/expand path | ~250 |
| Diagnostic eval modes | ~50 |
| Cosmetics/trajectory capture | ~30 |
| CLI surface for these branches | ~100 |

That is roughly **600–700 lines** of legacy or diagnostic surface before
accounting for shared code that should move elsewhere.

### Why static dead-code tools will not find most of it

Every top-level function and class has at least one reference. The deadness is
at the **feature-branch** level, not the function-symbol level. Most branches
are only reachable through optional flags that no current script uses.

The one exception is the unused `--additional-timesteps` flag.

## Why the monolith happened

The module started as the first PPO experiment and absorbed neighboring work in
place:

- lean preprocessing and tile probing;
- spawn and start pool handling;
- canonical, spawn, and start-pool evaluation;
- multi-level scheduling;
- legacy CNN and autoencoder branches;
- student initialization and input expansion;
- patience/best-checkpoint selection;
- CSV and TensorBoard logging;
- the full CLI.

Each addition was locally convenient. The repository restructure moved the
module into the new package but did not decompose it.

In practice, `PPO` became the shared utility library for the repo.

## Target composition

Do not introduce a single `BaseAlgorithm` class. PPO, DQN, NEAT, and supervised
training differ too much in their training loops. Compose the parts around
them.

```text
AlgorithmAdapter       PPO or DQN: model construction + learn()
EnvFactory             ports, isolated HOME, Godot processes, vec env
ObservationPipeline    lean / raw / legacy encoding
StartProvider          canonical, train pool, holdout pool
RolloutEvaluator       deterministic canonical/pool evaluation
CheckpointSelector     best metric, patience, save best
RunRecorder            episodes, TensorBoard, provenance
TrainingRunner         wires the components together
```

Suggested layout:

```text
src/plaiground/
  envs/
    launcher.py        # resolve_env_path, isolated_home, ports
    factory.py         # make_env, build_vec_env
  observations/
    pipeline.py        # lean preprocessing load/save/probe
  data/
    start_pool.py      # pool loading and validation
  evaluation/
    rollout.py         # generic deterministic evaluator
    pools.py           # canonical/start-pool summaries
  training/
    config.py          # typed config, not Namespace everywhere
    callbacks/
      patience.py      # evaluator-backed selector
      recorder.py      # episode CSV / TensorBoard
    runner.py          # composition root
  algorithms/
    reinforcement/
      ppo.py           # adapter
      dqn.py           # adapter
    genetic/neat/...
    supervised/...
```

The evaluator should depend on a small action-policy protocol:

```python
class ActionPolicy(Protocol):
    def act(self, observations, deterministic: bool) -> np.ndarray: ...
```

Then SB3 `model.predict` is wrapped once for both PPO and DQN. Canonical and
pool evaluation become identical across algorithms.

## Invariants

- No algorithm module imports another algorithm module.
- No cross-module private imports.
- No function receives the entire 69-flag `Namespace`.
- Env launch, pool loading, evaluation, and checkpoint selection live outside
  `algorithms/`.
- PPO and DQN use the same evaluator and selector contracts.
- The active lean path stays frozen until the current run finishes.

## Migration order

1. Extract `resolve_env_path`, `isolated_home`, port allocation, env creation, and
   vec-env construction into `envs/`.
2. Update DQN and tools to import the shared env helpers. Remove the PPO
   dependency.
3. Extract start/spawn pool loading into `data/` and remove private
   `_flatten_reset_state` imports.
4. Extract the generic deterministic evaluator and port PPO first, then DQN.
5. Replace both patience callbacks with one evaluator-backed selector.
6. Deduplicate `parse_levels`, history reading, and checkpoint-generation
   helpers used by the NEAT modules.
7. Introduce a typed config layer and a thin composition root.
8. Reassess every legacy PPO flag. Move still-useful features into named modules;
   delete only the branches with no caller and no documented workflow.
9. Add tests and run both PPO and DQN smokes.

Do the extraction mostly as verbatim moves first. Behavior cleanup is a separate
step.

## Validation gates

Before declaring the refactor complete:

- `python -m unittest discover -s tests` passes;
- PPO `--help` and a 2k-step lean smoke run pass;
- DQN `--help` and a 2k-step raw smoke run pass;
- the 225k baseline evaluation reproduces the same canonical result;
- JSON/CSV/TensorBoard/metric outputs retain their existing schemas;
- no active experiment command changes behavior.

After the refactor, add tests for:

- CLI defaults and config parsing;
- env path and isolated-home construction;
- canonical evaluator result parsing;
- patience scheduling and best-checkpoint selection;
- start-pool loading and reset-state normalization.

## Retention questions to settle before deleting

- Is PPO spawn fine-tuning still a real workflow, or was it superseded by DQN?
- Is `--init-student` / `--init-head` still needed for supervised warm starts?
- Is `--expand-from` still needed for observation expansions?
- Are legacy CNN/autoencoder encoders still used by archived experiments only?
- Is `--eval-stochastic` still useful for diagnostics?
- Does trajectory capture have a current consumer?

Until these are answered, move features rather than delete them.

## Revisit checklist after the 1M run

1. Check in the final 0002 result and selected checkpoint.
2. Re-read this audit and the dead-code inventory.
3. Create a refactor worktree or branch; do not work in the active run directory.
4. Extract the shared env/pool helpers first and update DQN/tools.
5. Extract the evaluator and patience selector.
6. Reduce `ppo.py` to an adapter and composition root.
7. Audit legacy flags against committed scripts, docs, and experiment configs.
8. Delete or archive only what has no caller and no documented workflow.
9. Run the full test suite, both smokes, and the 225k evaluation parity check.
