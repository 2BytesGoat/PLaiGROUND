# PPO policy distillation

Status: active experiment  
Updated: 2026-10-26

> **Implemented 2026-10-26:** DAgger now supports rollout temperature,
> training-only dropout, and an explicit initialization seed. Metric-rich eval
> reports efficiency/flow as well as time. Multiple initialization runs are the
> current ensemble path; weighted demonstrations remain a follow-up. See
> [model-improvement-experiments.md](model-improvement-experiments.md) § 6.

> **Pre-fix data:** the teacher snapshots and student results below were
> collected while the game training reward had a run-tracking bug: after the
> first started run, run time and jump-arc efficiency were disabled and the
> finish bonus fell back to +100. The game-side fix now also makes the agent
> own the first jump and uses the discounted flow potential; all 16 post-fix
> teachers clear their levels 100/100 without a forced start. The current
> student is observation-only (58 inputs, 256 hidden) and clears all 16 levels
> at 100/100 with no level token. The earlier 74-input level-one-hot run is a
> retired benchmark, not the current design.

## Goal

Merge the per-level PPO snapshots into **one student policy** that can play
multiple levels. This is supervised learning, not PPO: the student learns from
teacher actions and probabilities instead of from the game reward.

The student input is:

```text
58 game observations
```

There is no campaign-level ID or one-hot input. The policy must infer what to
do from terrain, exit direction, flow distance, velocity, and player state
alone, so the same weights can be tried on unseen maps.

The output is the same binary jump distribution as PPO:

```text
2 outputs: no-jump / jump
```

## Teachers

The current post-fix dataset uses all 16 teachers:

```text
1-1 through 1-16
```

The earlier 12-teacher subset (missing 1-4, 1-7, 1-13, and 1-15) is historical;
those four levels now have competent post-fix teachers and are included in the
262,778-sample dataset.

## Collection

```bash
.venv/bin/python src/12_ppo_distill.py collect \
  --episodes 2 \
  --agents 100 \
  --output-dir runs/ppo_distill
```

For every teacher level the collector runs the teacher with sampled actions
(so the 100 parallel agents diverge), records the observation, teacher
probabilities, sampled action, level id, episode, step, and agent, then writes
`dataset/<level>.npz`.

The agent owns the first action. Training starts each teacher episode idle,
the teacher must jump to start, and the collector records that start decision
without a forced override.

All game processes use private `HOME`/`XDG` trees under
`runs/ppo_distill/userdata`; real saves and settings are not touched.

## Student training

```bash
.venv/bin/python src/12_ppo_distill.py train \
  --epochs 25 \
  --hidden 128 \
  --output-dir runs/ppo_distill
```

Student:

```text
58 -> 128 -> 128 -> 2
softmax over 2 actions
```

The input is the raw observation vector returned by `PlayerAITrainingController`.
Level IDs remain dataset metadata for teacher selection and per-level
evaluation; they never enter the network.

Loss:

```text
-(teacher_probabilities * log_softmax(student_logits)).sum(axis=1).mean()
```

Optimizer: Adam, learning rate 1e-3, weight decay 1e-4, batch size 512.
Outputs:

```text
runs/ppo_distill/student.pt
runs/ppo_distill/train_student.csv
```

Use `--hidden 256` for a larger student. Each model writes its own
`train_<model>.csv`.

## Evaluation

```bash
.venv/bin/python src/12_ppo_distill.py eval \
  --levels all \
  --agents 100 \
  --output-dir runs/ppo_distill
```

DAgger labels student-visited states with the teacher:

```bash
.venv/bin/python src/12_ppo_distill.py dagger --levels 1-2
.venv/bin/python src/12_ppo_distill.py train
.venv/bin/python src/12_ppo_distill.py eval --levels all
```

Evaluation is deterministic (argmax), feeds the 58 observations directly,
lets the policy choose the first jump, and writes:

```text
runs/ppo_distill/eval/<model-name>/summary.csv
runs/ppo_distill/eval/<model-name>/godot-<level>.log
```

For example, `--model student_obs58.pt --eval-name obs58` writes under
`runs/ppo_distill/eval/obs58/`. The student chooses the first action; no forced
start is applied.

## Export

The 58-observation student exports to a versioned JSON MLP for the Godot
runtime:

```bash
.venv/bin/python src/12_ppo_distill.py export \
  --model student_obs58_256.pt \
  --output runs/ppo_distill_post/export/student_obs58_256.json \
  --output-dir runs/ppo_distill_post
```

The exporter:

- accepts only 58-input, shared-head checkpoints; the retired 74-input
  level-conditioned checkpoints are rejected;
- flattens the three weight matrices and biases in `[out, in]` order;
- writes `spec_version`, `observation_size`, `input_size`, `hidden_sizes`,
  `activation`, `action_names`, `action_repeat=4`, and
  `level_conditioning="none"`;
- verifies the JSON forward pass against PyTorch on 256 random observations
  before writing (max abs error ~6e-6, argmax agreement 100%).

Exports produced:

```text
runs/ppo_distill_post/export/student_obs58.json        (128 hidden)
runs/ppo_distill_post/export/student_obs58_dagger.json (128 hidden)
runs/ppo_distill_post/export/student_obs58_256.json    (256 hidden, selected)
```

The selected JSON is copied into the game at
`godot/src/resources/ai/student_obs58_256.json`. The game export presets include
`*.json` non-resource files so the model is packed into desktop and Web builds.
The game-side runtime is now live: `MLPPolicy` loads the JSON and
`PlayerAIModelController` runs it through the canonical 58-observation path,
with the heuristic as fallback. In-game headless verification finished all 16
campaign levels with zero deaths at 8× (1-1 in 2.35 s). This repository still
owns the training/export pipeline; see the game's
`docs/technical/ai-model-runtime.md`.

## Observation-only student (2026-10-26)

All 16 post-fix teachers were collected with the agent-owned first jump and no
forced opening action:

```text
teachers: 16
episodes per level: 2
agents: 100
samples: 262,778
```

The selected student is trained on the 58 observations only:

```text
student_obs58_256: 58 -> 256 -> 256 -> 2
25 epochs, batch 512, hidden 256
validation action agreement: 98.1%
```

100-agent deterministic evaluation, no forced first jump:

```text
student_obs58_256: 16/16 levels at 100/100, sum of best times 83.117 s
```

Per-level best times:

```text
1-1   2.333 s    1-9   2.900 s
1-2   3.017 s    1-10  4.600 s
1-3   3.383 s    1-11  9.600 s
1-4   4.250 s    1-12  6.233 s
1-5   2.967 s    1-13  6.267 s
1-6   5.050 s    1-14  6.800 s
1-7   7.967 s    1-15  6.117 s
1-8   5.433 s    1-16  6.200 s
```

The model is `runs/ppo_distill_post/student_obs58_256.pt`, also copied to
`student.pt`. Its evaluation is
`runs/ppo_distill_post/eval/obs58_256/summary.csv`; the model comparison is
`runs/ppo_distill_post/model_comparison.csv`.

### Time-profile DAgger student (selected 2026-10-26)

After the all-16 time-profile teacher retrain and champion selection, a fresh
collection produced 263,519 samples. The base student cleared 16/16 at a
79.133 s campaign sum. A targeted DAgger round on 1-3, 1-7, and 1-11 (63,680
student-visited labels) plus a second initialization seed produced the selected
model:

```text
student_obs58_256_time_dagger
58 -> 256 -> 256 -> 2
16/16 levels at 100/100, zero deaths
campaign best-time sum: 78.583 s
```

Per-level best times:

```text
1-1   2.333 s    1-9   2.900 s
1-2   3.017 s    1-10  4.467 s
1-3   3.233 s    1-11  8.250 s
1-4   4.250 s    1-12  5.967 s
1-5   2.967 s    1-13  6.267 s
1-6   5.050 s    1-14  6.800 s
1-7   7.617 s    1-15  6.117 s
1-8   3.217 s    1-16  6.133 s
```

Artifacts:

```text
runs/ppo_distill_time/student_obs58_256_time_dagger.pt
runs/ppo_distill_time/eval/time_dagger/summary.csv
runs/ppo_distill_time/export/student_obs58_256_time_dagger.json
```

The JSON export was installed as the game's bundled
`student_obs58_256.json`; in-game headless bench: 16/16, zero deaths, ~78.8 s
sum. A second DAgger variant using the 8.05 s 1-11 smoke teacher
(`student_obs58_256_time_dagger2`) improved 1-11 to 8.05 s but regressed 1-3 to
3.77 s, raising the sum to 78.917 s, so it was rejected.

The distillation path also supports the game's **multi-hot observation v2**
(`--obs-encoding multi_hot`, 754 values; see `src/obs_spec.py`). Teachers,
student, DAgger, eval, and export must all use the same encoding and
`action_repeat`. The existing 58-value champions are legacy and cannot be mixed
with a multi-hot run; train new multi-hot PPO teachers first.

The 128-hidden observation-only baseline cleared 15/16; only 1-10 failed
(0/100, 0 deaths, 100 timeouts). A targeted DAgger round on 1-10 labelled
17,013 student-visited states with the 1-10 teacher and retrained on 279,791
samples. It fixed 1-10 (100/100, best 4.600 s) but regressed 1-11 and 1-12 to
0/100, so that model is 14/16 and is not selected. This is the regression case
that motivates the multi-teacher experiment below.

### Retired 74-input level-conditioned benchmark

The earlier 74-input variants added a 16-way campaign-level one-hot:

```text
shared128: 74 -> 128 -> 128 -> 2, shared head
shared256: 74 -> 256 -> 256 -> 2, shared head
level128:  74 -> 128 -> 128 -> [16 x 2], level-conditioned heads
```

`shared128` and `shared256` cleared 16/16 at 100/100, and `level128` cleared
15/16 (1-11 failed). Those results are retained as a benchmark only. The
one-hot was a level-conditioning shortcut: it made the 16-level benchmark work
but gave the policy no valid input for an unseen map. The current student and
all new runs use 58 observations with no level token.

## Result (2026-10-25) — legacy pre-fix 74-input run

This run is historical: it used the pre-fix reward and the retired 74-input
level-conditioned student. It is not the current comparison.

Collectors:

```text
teachers used: 12
samples:       192,885 state/probability pairs
missing:       1-4, 1-7, 1-13, 1-15
```

Student:

```text
74 -> 128 -> 128 -> 2
25 epochs, batch 512
best validation accuracy: 98.2%
```

100-agent deterministic evaluation, initial behaviour cloning:

```text
passed: 1-1, 1-3, 1-5, 1-6, 1-8, 1-9,
        1-10, 1-11, 1-12, 1-14, 1-16

failed: 1-2 (despite a passing teacher)
       1-4, 1-7, 1-13, 1-15 (no teacher)

one student result: 11/16 levels
```

The useful result is that one shared student did absorb 11 of the 12
teacher policies. It is not yet the full campaign policy.

### DAgger attempt

DAgger was run on 1-2 only:

```text
student-visited states labelled by teacher: 20,672
retrained student: 213,557 samples
1-2 result after DAgger: 0/100
```

DAgger did not fix 1-2. That level likely needs a larger student or more
targeted labels; the other 11 teacher levels were retained.

## Proposed experiment — multi-teacher DAgger (not run)

The current student learns from the teacher for the level being collected. That
is a hard, per-level router: one level's labels can fix that level and still
shift the shared policy on another (the 1-10 DAgger round fixed 1-10 but
regressed 1-11 and 1-12). A multi-teacher variant would let the student visit
its own states and ask a committee of teachers for labels:

```text
student rollout
  -> query all teachers at each state
  -> aggregate their action probabilities / choose the best teacher
  -> add the aggregate labels to the dataset
  -> retrain one 58-input student
```

Candidate aggregation rules:

- **Per-level teacher** — current baseline; the teacher for the level being
  collected. Strong specialization, no cross-level smoothing.
- **Uniform ensemble** — average all teachers' action probabilities. Simple,
  but disagreements can blur into an indecisive policy.
- **Best-value teacher** — pick the teacher with the highest value/Q at the
  state. Requires comparable value estimates across teachers.
- **Competence-weighted gate** — learn which teacher is reliable from
  observations or map features, not from a campaign-level ID. This is the
  "this AI is good for these kinds of levels" idea in its general form.

The student stays one 58-input policy; a gate can be distilled back into the
single MLP if it starts as a mixture of experts.

Caveats: the teacher committee cannot exceed its own coverage; per-level PPO
values are not automatically comparable across levels; averaging conflicting
strategies can be worse than selecting one; and student capacity limits how
much teacher knowledge can be compressed. The first evaluation should compare
the current per-level labels against uniform and competence-weighted labels on
the 16 campaign levels plus a held-out map, and watch for regressions rather
than only total pass count.

## Limits

- Teacher quality is the student's ceiling.
- The four missing teacher levels in the historical run would not work until
  they were trained; the current dataset has all 16.
- Imitation can suffer from distribution shift; one DAgger round on 1-2 was not
  enough. More rounds or more data are the next experiments.
- The current student has no level input. It is one policy over the 58
  observations, so unseen maps are a generalization test rather than a
  missing-token error.
