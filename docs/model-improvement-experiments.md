# Model improvement experiments

Status: first pass complete; multi-hot observation implemented — no v2 model trained  
Updated: 2026-10-26

This is the implementation-facing companion to the game repo's
`docs/future/research-ideas.md`. It records the model-quality experiments that
are worth trying now that the 58-observation student clears all 16 campaign
levels, but is still slow on route-heavy maps.

## First pass (2026-10-26)

The metric contract is implemented in the game and in the eval paths:

- `PlayerAITrainingController.get_info()` now reports `finished`, `deaths`,
  `efficiency`, and `flow` alongside `run_time`.
- `10_ppo_pilot.py` reports finish rate, best/median/mean time, time CV,
  efficiency/flow means, and deaths.
- `--reward-profile=time` is a training-only game reward profile. The shipped
  default reward is unchanged.
- `--eval-every` / `--patience` stop PPO on a deterministic median-time plateau
  and select the best eval checkpoint.
- `13_ppo_retrain_all.py` drives the all-teacher retrain.
- `14_select_teachers.py` selects per-level champions with speed primary and
  efficiency/flow tie-breakers inside a 1% time tolerance.
- `16_ppo_seed_restarts.py` runs multi-seed PPO restarts after a plateau; it is
  multi-start optimization, not random-state training.
- `12_ppo_distill.py` now supports initialization seeds, training-only dropout,
  DAgger rollout temperature, and the metric-rich eval output.
- `15_recovery_test.py` evaluates live teacher takeover from seeded random
  student mid-run states.

Quick teacher/student pass (100 agents, deterministic, `action_repeat=4`,
`speedup=8`):

```text
all 16 teachers and the student: 100/100 finishes, 0 deaths
weakest teacher times: 1-11 9.53 s, 1-7 7.75 s, 1-10 4.60 s
student is within 0.07-0.22 s of its teacher on the weak levels
```

The 1-11 smoke test with the time profile + patience (25k additional steps,
one eval step) improved the teacher from **9.53 s to 8.05 s**, still 100/100
with zero deaths. Efficiency/flow fell from 0.673/0.911 to 0.485/0.861, which
is the expected trade: the policy found a faster route rather than a cleaner
one. This confirms that the finish-biased reward was leaving time on the table.

The full 16-level retrain is running with 50k additional steps, eval every 25k,
and patience 2. Early results: 1-7 improved 7.75 s → 7.62 s, 1-8 improved 5.43 s → 3.22 s,
1-10 improved 4.60 s → 4.47 s, and 1-11 improved 9.53 s → 8.25 s; 1-1 through
1-6 and 1-9 kept the old teacher because deterministic time did not improve.
The separate 1-11 25k smoke run reached 8.05 s, slightly better than the
50k all-level run; that is exactly why seed/plateau selection matters. Champion selection compares each new teacher against the
old one; the old teacher remains the fallback.

The 1-6 run is a useful warning: stochastic training episodes reached ~3.05 s,
but the deterministic 100-agent eval stayed at 5.05 s. Training-return or
sampled-policy improvements are not deployment improvements. The champion
selector and patience callback are therefore intentionally based on
deterministic eval time, not training reward.

A recovery smoke test on 1-11 (student drives for a seeded random 20–80
decisions, then the improved teacher takes over the same live player) finished
**100/100 after takeover**. Late takeovers at 80–120 and 120–145 also finished
100/100. The old teacher also recovered, but at 9.53 s. A 1-8 recovery run with
the 3.22 s champion finished 100/100 after takeover. These are the first direct
“teacher gets itself out of student states” results; the game-side
state-snapshot API is not required for this live-takeover version.

### Selected student

The all-16 champion collection (263,519 samples) trained
`student_obs58_256_time` to 16/16 and a 79.133 s campaign sum. A targeted
DAgger round on 1-3, 1-7, and 1-11 (63,680 labels) plus a second initialization
seed produced `student_obs58_256_time_dagger`:

```text
16/16 levels, 100/100 finishes, 0 deaths
campaign best-time sum: 78.583 s (old bundled student: 83.117 s)
```

It fixed the 1-3 regression (3.43 → 3.23 s), improved 1-7 (7.97 → 7.62 s) and
1-12 (6.23 → 5.97 s), and kept the 1-8 (3.22 s) and 1-11 (8.25 s) gains. A
further 1-11 DAgger round using the 8.05 s smoke teacher produced
`student_obs58_256_time_dagger2`; it improved 1-11 to 8.05 s but regressed 1-3
to 3.77 s, raising the sum to 78.917 s, so it was rejected.

The selected JSON is `student_obs58_256_time_dagger.json`, exported and
verified, then installed as the game's `student_obs58_256.json`. In-game
headless bench: all 16 finish, zero deaths, ~78.8 s sum.

## Multi-hot observation v2 (implemented 2026-10-26)

The game now supports a second observation contract alongside the shipped
58-value vector:

```text
legacy:     58 values
multi_hot: 754 values
  7 x 7 x 15 semantic grid
  + 19 player/route values
```

The grid stores multi-hot type bits (solid, hazard, exit, bounce, dissolve,
powerup, pickup, secret), powerup kind, dissolve state/timer, and raw bounce
orientation. The player tail stores scaled velocity (magnitude preserved),
time-to-exit + gradient, exit direction, floor/wall/jump phase, and powerup
inventory. Empty cells are all zeros; stacked symbols set multiple type bits.
There is no `dx/dy`, no level ID, and no full 15-way tile one-hot.

The schema is mirrored in `src/obs_spec.py`; the game's `AIObservationSpec`
is the source of truth. Run commands with:

```text
--obs-encoding multi_hot
```

on `10_ppo_pilot.py`, `12_ppo_distill.py collect/train/dagger`, and the retrain
/ seed-restart drivers. `12_ppo_distill.py eval` and `15_recovery_test.py` read
the encoding from the student checkpoint. `export` writes spec 2 JSON with
`obs_encoding`, `observation_size`, and the training `action_repeat`.

**Teachers and students must share the encoding and cadence.** The existing
teacher champions and `student_obs58_256_time_dagger` are legacy 58-value,
`action_repeat=4`; they cannot label or drive a multi-hot environment. A v2
run needs new multi-hot PPO teachers first, then collection/DAgger. The bundled
JSON remains legacy until that model is trained, exported, and benched.

## Current baseline

The selected student artifact is `student_obs58_256_time_dagger` (runtime
model name `student_obs58_256`):

```text
student_obs58_256_time_dagger
58 -> 256 -> 256 -> 2
ReLU hidden layers, argmax over idle/jump
action_repeat = 4
level_conditioning = none
```

It clears all 16 campaign levels at 100/100 deterministic agents, with zero
deaths in the game-side headless bench. The sum of best times is **78.583 s**.

Using **gold = par x 1.25** from the game's level data, the weak spots are:

| Level | Student | Gold | Gap |
|---|---:|---:|---:|
| 1-1 | 2.33 s | 2.87 s | faster than gold |
| 1-6 | 5.05 s | 4.63 s | +9% |
| 1-7 | 7.62 s | 6.22 s | +23% |
| 1-10 | 4.47 s | 3.79 s | +18% |
| 1-11 | 8.25 s | 4.27 s | +93% |

The easy levels show the model can move quickly when the route is simple. The
route-heavy levels suggest the bottleneck is not raw network size alone. It is
more likely a mix of:

- the policy's route information,
- the reward's finish bias,
- the lack of explicit exploration after PPO converges,
- the teacher's own route quality, and
- the fixed `action_repeat=4` control cadence.

The student matches the champion on the weak levels: `1-11` is 8.25 s student
vs 8.25 s teacher, `1-7` is 7.62 vs 7.62, and `1-10` is 4.47 vs 4.47. DAgger
and distillation cannot exceed the teacher. **Improve the teacher on the hard
levels first; otherwise a bigger student just compresses the same slow route.**

## Observation/interface diagnosis (2026-10-26)

The selected student is input-limited before it is capacity-limited. The
58-value observation is a local 7×7 tile window plus a scalar flow distance; at
16 px tiles and 220 px/s max speed the window is only ~0.22 s ahead, while a
full jump lasts ~0.6 s. The vector also normalizes velocity (direction only, no
speed magnitude), omits the flow gradient, dissolve-block timer, powerup
type/count, bounce-pad facing, and previous action, and decides at 15 Hz
(`action_repeat=4`). A recurrent model cannot recover information that was
never observed; add explicit state and test frame stacking before GRU/LSTM or
attention. The teacher shares the same interface, so DAgger fixes state
distribution, not the teacher ceiling. The 100-agent eval is deterministic; it
is a regression test, not a robustness test. The game-side full analysis and
candidate experiment order are in
`../dragon-jump-remaster/docs/future/research-ideas.md` § Policy diagnosis.

## Current constraints

### Game runtime

`MLPPolicy` is intentionally narrow:

- exactly 58 observations;
- exactly two ReLU hidden layers;
- exactly two outputs (`idle`, `jump`);
- argmax inference;
- `action_repeat` read from the JSON;
- no level ID;
- no dropout;
- no arbitrary topology, skip connections, or recurrence.

Widths can vary within the two hidden layers, but the loader rejects a
three-layer model. Any layer-count or topology change requires a game-side
runtime change. Any model exported to the game must pass the JSON forward-pass
parity check.

### Training cadence

The bundled student is tagged `action_repeat=4`. PLaiGROUND's PPO default is
also 4 for the student path, but the NEAT config and some historical teacher
runs used other values. **Match the cadence used for export when comparing
policies.** A policy trained at 8 and exported at 4 is a different control
problem.

### Reward

The game reward lives in `PlayerAITrainingController.get_reward()`:

```text
progress potential  + flow distance, discounted by 0.99
efficiency          - 0.02 * wasted_air_frames
time pressure       - 0.05 per decision
finish bonus        + 100 + 50 * previous_best / run_time
death penalty       - 50
```

At `action_repeat=4`, the time term is only `-0.75/s`. A 10-second run pays
about `-7.5`, while finishing pays roughly `+100` to `+150`. The finish base
dominates, and the finish ratio rewards relative improvement rather than
absolute speed. This is a good "learn to finish" reward, not a strong
"minimize time" reward.

Reward changes are game-side changes. PLaiGROUND should consume the resulting
reward and log its terms; it should not silently fork the game reward.

## Experiment tracks

### 1. PPO evaluation patience

The current PPO pilot saves checkpoints and runs a final evaluation, but it does
not stop on an evaluation plateau. `NEAT` already has a useful model in
`07_neat_agent.py`: minimum generations, patience, minimum finish rate, and
minimum finish streak.

Proposed PPO loop:

- evaluate the deterministic policy every N timesteps (for example 25k);
- evaluate on the target level and a fixed regression set;
- track best/median clear time and finish rate;
- stop when there is no meaningful time improvement for `patience` evaluations;
- keep the **best evaluation checkpoint**, not merely the final model;
- keep a hard max-step cap as a guardrail.

Do not use the training return as the patience metric. A policy can improve its
expected return while its deterministic argmax time gets worse.

Suggested defaults to test:

```text
eval_every = 25_000 timesteps
patience = 3-5 evaluations
min_improvement = 1-2% on best time
min_finish_rate = 0.90-1.00
```

Implementation point: add a callback around `evaluate_policy()` /
`evaluate_levels()` in `10_ppo_pilot.py`, and reuse the same idea in
`11_ppo_lineage.py` per stage. The callback should save the evaluated best
checkpoint separately from the periodic training checkpoints.

### 2. Time-focused reward A/B

This is the highest-leverage experiment after the evaluation harness. It needs
a game-side reward branch or flag so the current baseline can still be
reproduced.

Candidate changes:

- lower the finish base during a fine-tune stage;
- add a fixed-target time bonus, e.g. `target_time / run_time`, uncapped or
  capped above 1.0;
- strengthen the time pressure;
- keep efficiency as a small tie-breaker;
- lower the death penalty only in the time-focused stage, with a finish-rate
  gate;
- replace the grid-flow potential with a physics-aware `time_to_exit` potential
  (falling back to flow when `INF`), or blend the two.

Do **not** add the player's exact practice speed axis if it is capped at gold.
The student already beats gold on several levels, so that axis saturates.

Do **not** add the practice flow ratio (`distinct_cells / cell_changes`) as a
primary reward. It rewards clean routes, not necessarily the fastest route; a
fast run may legitimately backtrack.

Do **not** add consistency. It is a cross-run FOCUS streak, not a per-episode
learning signal.

Recommended A/B target: `1-11`, with `1-1`, `1-7`, and `1-10` as regression
levels. Fine-tune from the current post-fix teacher or student, not from a
pre-fix reward run.

### 3. Per-level specialists

The global student averages over 16 levels. A specialist can take risks and
find a route that the global policy will not.

Experiment:

1. Pick one hard level, initially `1-11`.
2. Fine-tune the existing teacher or a new PPO policy with the time-focused
   reward and evaluation patience.
3. Compare best time, median time, finish rate, and the regression set.
4. If the specialist is materially better, distill its rollouts into the global
   student rather than replacing the student with a level-specific model.

The shipped model must remain level-agnostic. Specialists are teachers, not a
reason to add a campaign-level one-hot input.

### 4. Fixed topology, NEAT, and architecture search

The current NEAT config in `src/neat_1_1_config.txt` is **not** the student
architecture:

```text
58 inputs -> 1 output
sigmoid
initial_connection = full
structural mutations enabled
```

It evolves compact networks and uses the game reward as fitness. It is useful
as an independent teacher, but it is not a drop-in fine-tuner for the
58 -> 256 -> 256 -> 2 student.

There are three separate ideas here:

- **Fixed-topology weight evolution:** keep the student's structure and evolve
  only weights/biases. This is an evolutionary strategy, not really NEAT. It is
  the most practical way to seed a population from the current model.
- **Topology search:** let NEAT add/remove nodes and connections. This can
  discover a different teacher, but the result is not directly exportable to
  `MLPPolicy`. Distill it into the fixed student.
- **Architecture search:** decide layer count and widths. The current game
  runtime rejects anything except two hidden layers, so layer-count search is
  teacher-only unless the runtime is changed.

Recommendation:

- For a first fine-tuning experiment, use a **fixed-topology evolutionary
  strategy in PyTorch** on the student weights, or add a custom fixed-topology
  NEAT genome. Do not start by changing the game runtime.
- If topology search is interesting, run it as a teacher experiment and distill
  the best policy. Constrain the exported student to two hidden layers.
- Width-only search (two hidden layers, 64-512 per layer) is exportable with the
  current JSON format and can be tested without a runtime change.

### 5. Dropout

Dropout is a training regularizer, not a required shipped parameter.

- The game runtime does not implement dropout; export would run with it off.
- Dropout at inference makes the AI stochastic and the leaderboard noisy.
- For PPO, dropout adds noise to advantage estimates.
- For DAgger, **MC dropout during student rollouts** is useful: it creates
  diverse student-visited states to label with the teacher. Turn it off for
  export.

If dropout is evolved as a genome parameter, evaluate it with a fixed seed and
do not compare a stochastic policy directly against a deterministic one.

### 6. Ensemble DAgger and initialization diversity

This is a strong direction and fits the existing `12_ppo_distill.py` pipeline.

Current DAgger already does the important part:

- student rollouts supply the states;
- the level teacher supplies the action probabilities;
- the student is retrained on the aggregated dataset.

What is missing is diversity and coverage. Proposed variants:

- train `N` students with different initialization seeds;
- have each student collect rollouts;
- aggregate all student-visited states;
- label them with the teacher;
- retrain one student;
- repeat for several DAgger rounds;
- use disagreement between students to prioritize states.

Additional exploration knobs for rollout collection:

- sample actions from the student distribution instead of always taking argmax
  (this is already how the current DAgger collector runs);
- add action noise or a temperature parameter;
- use MC dropout during rollouts;
- include the current student's states as replay data;
- keep a balanced per-level mix so one hard level does not dominate.

Important limits:

- The teacher is the student's ceiling. A better student needs a better
  teacher, especially on `1-11`.
- Multiple initializations do not fix missing observation information.
- If the global student has to average incompatible per-level strategies,
  distillation can regress. Evaluate the full 16-level set, not just the target
  level.
- Soft teacher probabilities are already used; keep them rather than collapsing
  to hard argmax labels.

The existing `docs/ppo-distillation.md` multi-teacher proposal is compatible
with this. Ensemble DAgger is the data-collection side; multi-teacher
aggregation is the label side.

### 7. Shared player/AI metrics

The player-facing radar and the AI reward should share raw telemetry, not the
same formula. The game-side research document records the contract; the short
version for PLaiGROUND is:

| Signal | Player use | AI use |
|---|---|---|
| Clear time vs gold | speed axis, capped at 1.0 | fixed-target time bonus, uncapped or loosely capped |
| Useful air / total air | efficiency axis | small reward tie-breaker / evaluation |
| Distinct cells / cell changes | flow axis | evaluation, or a tiny regularizer only |
| Finish streak | consistency axis | finish rate/stability, not per-episode reward |

Use the four axes as the evaluation dashboard for both human and AI runs. Do
not scalarize all four into the RL reward and expect optimal time.

For distillation, the metrics can weight demonstrations: a fast, efficient,
clean teacher run gets more sample weight than a slow sloppy clear. That is a
better use of the player metrics than adding them all to the reward.

## Metrics as reward vs selection

Do not use the four player axes as a single weighted reward. Speed is the primary
deployment objective; efficiency is a small reward tie-breaker; flow is an
evaluation/tie-breaker; consistency is a robustness gate.

- **Reward:** time + a small efficiency term. A tiny flow term is an optional
  experiment; it must not be allowed to trade away time.
- **Selection:** speed first, then efficiency/flow only when candidates are
  within about 1–2% time. A slower clean run is not the best teacher.
- **Consistency:** finish rate across canonical/perturbed starts, time variance,
  and takeover recovery rate. Keep it as a gate, not a per-episode reward.

## Seeds vs random starts

Two different experiments:

- **PPO seed restart:** new exploration/initialization from the current
  champion. This can escape a local optimum. The level is deterministic, so the
  start state is unchanged.
- **Random start/takeover:** the student drives for a seeded random number of
  decisions and the teacher continues. This is state-distribution/robustness
  training. Use it for recovery evaluation, and for DAgger-style labels; do not
  feed an off-policy student prefix directly into a PPO update without treating
  it as a reset/context.

## Execution order (current)

The current pass deliberately **does not use NEAT or architecture search**. It
is PPO + reward + patience + distillation only.

1. **Metric report card** — shared telemetry and 100-agent eval output.
2. **Quick teacher pass** — all 16 post-fix teachers and the student; identify
   the speed gaps.
3. **Retrain all 16 teachers** — start from each post-fix teacher, time profile,
   eval every 25k steps, patience 2, 50k additional-step cap. The 1-11 smoke
   test improved 9.53 s → 8.05 s.
4. **Champion selection** — keep the old teacher if the new one fails the
   finish gate or regresses; speed first, efficiency/flow only within a 1%
   time tie. The old teacher is the fallback.
5. **Multi-seed restarts** — after a canonical plateau, rerun PPO from the
   champion with 3 seeds; keep the best deterministic eval. This is multi-start
   exploration, not random-state training.
6. **Distill champions** — fresh 16-level collection, train the 58→256→256→2
   student, ensemble DAgger / initialization diversity, full 16-level eval,
   JSON export + in-game bench.
7. **Recovery test** — live teacher takeover from seeded random student states;
   use the states for recovery evaluation, and DAgger labels if the student is
   still weak. A game-side state-snapshot API is not required for the live
   takeover version.

### Commands

```bash
# 16-teacher retrain (time profile + patience)
.venv/bin/python src/13_ppo_retrain_all.py \
  --levels all --timesteps 50000 --patience 2 --eval-every 25000

# Resume without retraining completed levels
.venv/bin/python src/13_ppo_retrain_all.py \
  --levels all --timesteps 50000 --patience 2 --eval-every 25000 \
  --skip-existing

# Keep old teacher when the new one is not better; metric tie-breaks within 1%
.venv/bin/python src/14_select_teachers.py --levels all

# Multi-seed restarts after a canonical plateau (hard levels first)
.venv/bin/python src/16_ppo_seed_restarts.py \
  --levels 1-7,1-10,1-11 --seeds 42,43,44 --timesteps 25000

# Live takeover recovery test
.venv/bin/python src/15_recovery_test.py \
  --level 1-11 --teacher runs/teacher_champions/1-11/ppo_pilot.zip \
  --min-takeover-step 20 --max-takeover-step 80 --seed 42
```

### V2 / nice-to-have

- fixed-topology ES or a custom NEAT genome;
- topology/width search as a teacher-only experiment;
- dropout as an evolved parameter;
- observation upgrade (time-to-exit or flow-direction features);
- `action_repeat=2` control experiment.

## Evaluation contract

Every candidate should report:

- 100-agent deterministic finish count per level;
- best and median clear time per level;
- sum of best times;
- deaths;
- gold gap;
- regression levels (`1-1`, `1-3`, `1-7`, `1-10`, `1-11` at minimum);
- JSON parity check against PyTorch;
- in-game headless bench for the exported JSON.

A candidate is promotable only if it improves the target without breaking the
rest of the campaign. "16/16" remains the minimum competence gate; it is not
enough to prove the model is better.

## Open decisions

- Is the goal a better demo model, or genuinely optimal route times? The
  training budget and acceptable regression risk differ.
- Should the time-focused reward remain a training-only branch, or become a
  permanent alternate training profile?
- Does the V1 Dragon Brain plan need the same student trunk, or can the offline
  teacher improve independently and be re-distilled later?
- Do we add the training-only state snapshot/reset API for direct recovery
  tests, or keep the random-delay proxy for this pass?
