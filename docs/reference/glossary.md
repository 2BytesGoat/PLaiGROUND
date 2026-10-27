# Glossary

Terms used across PLaiGROUND docs and run artifacts. When a term also exists in
the game vault, the game is the source of truth for behavior.

## Models and training

**Teacher** — a per-level policy trained to clear one level. Teachers are used as
label sources for a student or as performance baselines; they are not shipped.

**Student** — one level-agnostic policy trained from teacher labels (supervised
distillation/DAgger). The deployed `student_obs58_256.json` is the 58-value,
256×256 student.

**Specialist** — a policy trained/evaluated on one level, not expected to
generalize. A specialist's canonical time is not a campaign result.

**Lineage** — carrying one population/policy through levels in order
(1-1 → 1-2 → …), with zero-shot and post-stage evaluations.

**Zero-shot** — evaluating a policy on a level it has not trained on.

**DAgger** — dataset aggregation: run the student, label the states it visits
with the teacher, retrain. Used to correct state-distribution mismatch.

**Distillation** — supervised training of a student on teacher actions or action
probabilities. Not reinforcement learning; no game reward in the loss.

**PPO** — on-policy policy-gradient algorithm (SB3 `PPO`).

**DQN / Double DQN** — off-policy Q-learning with a replay buffer and target
network. SB3's `DQN` uses Double DQN by default (`double_q=True`).

**NEAT** — neuroevolution of augmenting topologies; a population of networks,
fitness = the game reward. Used here as a route/teacher generator.

**Replay buffer** — stored transitions reused for off-policy updates (DQN).

**Target network** — a slowly updated copy of the Q-network used to compute DQN
targets.

## Observations and actions

**Legacy 58** — the shipped observation: 7×7 tile IDs (49) + exit direction (2),
flow scalar (1), velocity direction (2), on-floor (1), on-wall (1), jump phase
(1), has-powerup (1).

**Multi-hot 754** — 7×7×15 semantic grid (735) + 19 player/route tail values.

**Sensor 794** — 754 + 20 head-ray distances + 20 hit flags (`multi_hot_sensor`).

**Action repeat** — physics frames per policy decision. Default 4 → 15 decisions
per second at 60 Hz.

**Speedup** — wall-clock multiplier for the training loop. It scales physics
ticks and `Engine.time_scale`; per-tick simulation and run times stay the same.

**Start jump** — reset convention: press jump once after a stationary reset so
the run starts, then hand control to the policy. Not a learned action; apply the
same way in training and eval. Do not apply to a state snapshot whose run is
already active.

**State snapshot** — optional reset `state` with position, velocity, state name,
`run_time`, `best_run_time`, `run_active`, `started_walking`. Supported for
grounded, powerup-free states only; not jump states, powerups, or dissolve
timers.

**Prefix replay** — reset to canonical, replay a recorded action prefix, then
hand control to the policy. Preserves the run clock without a state snapshot.

## Routes and evaluation

**Canonical start** — the level's normal start cell/facing.

**Clean route** — a successful NEAT trajectory with no deaths/backtracks (as
defined by the archive tooling), stored with `path_cells` and `actions`.

**Finisher** — one successful trajectory record in the NEAT archive.

**Route signature** — hash identifying a route; used to dedupe finishers.

**Stratum** — start-cell band by BFS distance to the exit: `canonical`,
`medium`, `hard`, `easy`/near-exit. `easy` is usually excluded from training.

**Start pool / spawn pool** — weighted list of reset starts: canonical, cell +
facing, state snapshot, or prefix replay.

**Holdout** — starts/routes reserved for evaluation and excluded from training.
Holdout by `(cell, facing)` is a checkpoint holdout, not a route holdout; see
[`../known-issues.md`](../known-issues.md).

**Strata evaluation** — deterministic evaluation from every start in a stratum,
reporting finish count, best/median/worst time, and spawn acceptance.

**Coverage** — how many starts finish, not how fast. Keep it separate from
canonical speed.

## Repo/run terms

**Run directory** — `runs/<name>/`; generated and gitignored. Contains
checkpoints, logs, monitor/episode CSVs, evaluation summaries, and isolated
`userdata/`.

**Monitor CSV** — SB3 `VecMonitor` episode summary (`r, l, t` and info columns).

**Episodes CSV** — PPO per-episode rows including start kind/source, finish,
timeout, run time, efficiency, flow.

**Eval summary** — `eval*/summary.json`: per-stratum `starts`, `finished`,
`spawn_ok`, `best`, `median`, `worst`, and failures.

**Userdata** — per-process private `HOME`/`XDG` tree so training never touches
the real save/settings/ghosts/medals/leaderboards.

**Archive** — `archive/2026-10-27-cleanup/`: historical scripts/runs/docs moved
out of the active tree. `archive/runs/` is gitignored and large.
