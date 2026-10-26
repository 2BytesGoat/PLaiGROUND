# PPO policy distillation

Status: active experiment  
Updated: 2026-10-26

> **Pre-fix data:** the teacher snapshots and student results below were
> collected while the game training reward had a run-tracking bug: after the
> first started run, run time and jump-arc efficiency were disabled and the
> finish bonus fell back to +100. The game-side fix now also makes the agent
> own the first jump and uses the discounted flow potential; all 16 post-fix
> teachers clear their levels 100/100 without a forced start. The current
> student is observation-only (58 inputs); the earlier 74-input level-one-hot
> run is a retired benchmark, not the current design.

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

## Observation-only student (2026-10-26)

All 16 post-fix teachers were collected with the agent-owned first jump and no
forced opening action:

```text
teachers: 16
episodes per level: 2
agents: 100
samples: 262,778
```

The current student is trained on the 58 observations only:

```text
student_obs58: 58 -> 128 -> 128 -> 2
25 epochs, batch 512, hidden 128
validation action agreement: 97.8%
```

The 100-agent deterministic evaluation, no forced first jump, is running; the
per-level pass counts and best times will replace this paragraph when complete.
The model is `runs/ppo_distill_post/student_obs58.pt`; evaluation writes to
`runs/ppo_distill_post/eval/obs58/`.

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

## Limits

- Teacher quality is the student's ceiling.
- The four missing teacher levels in the historical run would not work until
  they were trained; the current dataset has all 16.
- Imitation can suffer from distribution shift; one DAgger round on 1-2 was not
  enough. More rounds or more data are the next experiments.
- The current student has no level input. It is one policy over the 58
  observations, so unseen maps are a generalization test rather than a
  missing-token error.
