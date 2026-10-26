# PPO pilot

Status: active experiment  
Updated: 2026-10-26

> **Implemented 2026-10-26:** evaluation-based patience and best-checkpoint
> selection are available via `--eval-every` / `--patience`. The metric-rich
> eval reports finish rate, best/median/mean time, time CV, efficiency, and
> flow. See [model-improvement-experiments.md](model-improvement-experiments.md)
> § 1 and § 2.

> **Reward/start correction (2026-10-26):** PPO teacher results below the
> post-fix section are **pre-fix**. The game training controller stopped tracking run time and
> jump-arc efficiency after the first started run, and evaluation forced the
> first jump. The game-side fix starts every training episode idle (the agent
> owns the first jump), restores the run clock/efficiency terms, corrects the
> record-bonus order, and uses the discounted flow potential. Do not compare
> these runs against post-fix runs without re-baselining.

## Goal

Test whether one PPO policy can learn Dragon Jump faster than the sequential
NEAT lineage. The first target is **1-3**, the first level that failed the
NEAT finish-rate gate. If PPO shows a signal there, the next step is one policy
across levels rather than one independent PPO policy per level.

## Why one process per world

The existing multi-agent training wrapper is useful for population inference
and 100-agent evaluation, but it is not a standard SB3 vector environment:
the Python wrapper does not yet auto-reset individual agents.

`10_ppo_pilot.py` therefore runs **one isolated Godot training world per
process** and wraps each one as a normal Gymnasium environment. SB3's
`DummyVecEnv`/`SubprocVecEnv` handles per-agent reset, truncation, and episode
bookkeeping.

The pilot uses 8 worlds. That is enough to establish whether PPO learns the
level; it is not the 100-agent evaluation population. **8 is the pilot default,
not a measured optimum.** The game now supports optional per-agent reset
indices (`GodotEnv.reset(agents=...)`), but the pilot stays 8×1 until a
multi-agent VecEnv exists, so the completed 512 width A/B remained comparable.
See [model-improvement-experiments.md](model-improvement-experiments.md) §
Training topology.

## Isolation

Each Godot process gets its own `HOME`/`XDG` tree under:

```text
runs/<run>/userdata/
```

That keeps training away from the real user save, settings, ghosts, medals,
unlocks, arcade lives, and leaderboard data. Godot's raw stdout/stderr goes to:

```text
runs/<run>/godot/train_<n>.log
```

The console stays limited to PPO episode summaries and the final evaluation.
No production menu or export wiring is added.

## Command

```bash
.venv/bin/python src/10_ppo_pilot.py \
  --level 1-3 \
  --n-envs 8 \
  --timesteps 100000 \
  --max-decisions 150 \
  --action-repeat 4 \
  --speedup 8 \
  --n-steps 150 \
  --batch-size 120 \
  --epochs 10 \
  --net-arch 64 64 \
  --output-dir runs/ppo_1-3
```

`max_decisions=150` with `action-repeat=4` is **600 physics ticks = 10 s**.
The game's reward is used unchanged: progress, efficiency, time, finish, and
survival. No second fitness function is added. Add `--obs-encoding multi_hot`
for the 754-value semantic-grid observation; a resumed model and its environment
must use the same encoding and control cadence.

`--init-student <student.pt>` warm-starts the PPO policy/action head from a
distillation checkpoint while leaving the value head random. This is used for
multi-hot teachers that start from a behavior clone of the legacy champions.
`--baseline-eval` (implied by `--init-student`) evaluates that starting policy
and seeds eval patience with it, so a bad PPO run cannot replace a good clone.
`src/17_multi_hot_teachers.py` wraps the per-level warm-start loop.

The agent owns the first jump. Every training episode resets to idle and the
policy must start the run; evaluation uses the policy's first action too.
`--force-start-jump` remains available as a diagnostic override, but it is not
part of the competence gate. This keeps the start decision in the same
distribution as training.

## Outputs

```text
runs/<run>/monitor.csv          VecMonitor episode summary
runs/<run>/episodes.csv         one row per episode, including finish/timeout
runs/<run>/checkpoints/         PPO checkpoints
runs/<run>/ppo_pilot.zip        final policy
runs/<run>/eval/results-*.csv   deterministic 100-agent evaluation
```

Evaluate a saved policy without training:

```bash
.venv/bin/python src/10_ppo_pilot.py \
  --level 1-3 \
  --eval-only \
  --resume runs/ppo_1-3/ppo_pilot.zip \
  --max-decisions 150 \
  --action-repeat 4 \
  --eval-agents 100 \
  --show
```

Add `--eval-stochastic` to evaluate the sampled policy. The deterministic
mode is the default; the first start action is chosen by the policy.

## Baseline

The comparison is the existing NEAT lineage on 1-3:

```text
repeat 4, 600 decisions: 6/100 training finishes, best 3.567 s
repeat 1, 900 decisions: 10/100 training finishes, best 3.017 s
```

PPO is judged by whether it beats those finish rates and reaches the 30%
gate faster. A failed pilot is still useful: it separates optimizer failure
from control-cadence and reward failure.

## Sequential all-level lineage

`11_ppo_lineage.py` is the PPO counterpart to the NEAT lineage. It keeps one
policy and trains one level at a time:

1. zero-shot evaluate the incoming policy on the next level with 100 agents
2. train that same model on the new level
3. evaluate the post-stage policy
4. save `snapshots/<level>/ppo_pilot.zip`
5. continue to the next level

It does **not** run one environment per level in parallel. Parallel level
environments feed different level gradients into every PPO update; the lineage
runner measures transfer from one level to the next instead.

```bash
.venv/bin/python src/11_ppo_lineage.py \
  --levels all \
  --start-level 1-1 \
  --output-dir runs/ppo_lineage \
  --n-envs 8 \
  --chunk-timesteps 100000 \
  --max-timesteps 200000 \
  --min-finish-rate 0.30 \
  --continue-on-fail \
  --final-eval
```

The gate is checked on deterministic 100-agent evaluation after each chunk.
`--continue-on-fail` keeps moving through the campaign while marking failed
gates in `summary.csv`; without it the lineage stops at the first failed gate.
Each stage has its own raw log and `eval/summary.csv`.

### Sequential result

The fixed sequential pass trained 1-4 through 1-16 from the 1-1..1-3
snapshots. Stage-time results:

```text
pass: 1-1, 1-2, 1-3, 1-5, 1-6, 1-8, 1-9,
      1-10, 1-11, 1-12, 1-14, 1-16
fail: 1-4, 1-7, 1-13, 1-15
```

However, the final policy catastrophically forgot every earlier level:

```text
final 100-agent eval:
  1-1 .. 1-15: 0/100 finishes
  1-16:       100/100 finishes, best 6.200 s
```

Each stage was genuinely competent immediately after training that level, but
later PPO updates replaced those weights. Naive sequential PPO does not retain
an all-level policy. The next viable direction is rehearsal/regularization or
distilling the per-level snapshots into one student policy, not another pass of
the same sequential training.

### Teacher repair (2026-10-26, pre-fix)

After the sequential failures, four missing teachers were retried from their
failed snapshots with 8 worlds and 400k extra steps:

```text
1-4: 400k -> 100/100 deterministic, best 7.98 s
1-7: 400k -> 100/100 deterministic, best 7.97 s
1-15: stopped at ~330k after the reward audit; no final eval
1-13: stochastic 70/100, deterministic 0/100; 200k low-entropy sharpen did not fix it
```

These runs used the pre-fix reward tracking described above. The 1-4/1-7
snapshots were competent but had not learned the start action; the post-fix
runs below replace them. The old 1-13 deterministic failure was later fixed by
the reward/start correction, so it is no longer an open teacher.

### Post-fix validation (2026-10-26)

An intermediate 5k-step 1-1 smoke (`runs/reward_fix_smoke`) validated the run
clock and finish-ratio repair before the later start-ownership/discounted-flow
changes:

```text
245/245 training episodes finished
episode_length=23 -> run_time=1.633 s (step fallback would be 1.533 s)
episode_length=36 -> run_time=2.367 s (step fallback would be 2.400 s)
finish return ≈ 160.2
```

The current contract is validated by the post-fix runs below, which start idle
and use the discounted flow potential:

```text
1-1,  50k fine-tune:            100/100, best 2.333 s
1-2,  50k fine-tune:            100/100, best 3.017 s
1-3, 100k from scratch:         100/100, best 3.233 s
1-4, 100k fine-tune:            100/100, best 4.250 s
1-5,  50k fine-tune:            100/100, best 2.967 s
1-6,  50k fine-tune:            100/100, best 5.050 s
1-7, 100k fine-tune:            100/100, best 7.750 s
1-8,  50k fine-tune:            100/100, best 5.433 s
1-9,  50k fine-tune:            100/100, best 2.900 s
1-10, 50k fine-tune:            100/100, best 4.600 s
1-11, 50k fine-tune:            100/100, best 9.533 s
1-12, 50k fine-tune:            100/100, best 6.100 s
1-13, 400k from failed:         100/100, best 6.267 s
1-14, 100k fine-tune:           100/100, best 6.817 s
1-15, 400k from failed:         100/100, best 6.117 s
1-16, 50k fine-tune:            100/100, best 6.133 s
```

All 16 use the fixed run clock, finish ratio, and `C(s)-0.99*C(s')`
progress. 1-13 and 1-15 were the blockers before the reward/start fix; both now
reach the gate deterministically without a forced first jump. 1-14 needed an
extra 100k after its first 50k fine-tune stalled at 0/100. Post-fix models are
under `runs/teacher_repair/final/` (hard levels) and
`runs/teacher_repair_post/<level>/`; they are the teachers for the next
distillation pass.

## Pilot result (2026-10-25)

PPO, 8 worlds, 100,000 timesteps, about 2.5 minutes:

```text
training, last 100 episodes: 99/100 finishes
deterministic 100-agent eval: 100/100 finishes, best 3.333 s
stochastic 100-agent eval:     96/100 finishes, best 3.183 s
```

This pilot is pre-fix: its deterministic policy needs the old forced start
jump; without it, its first argmax action is idle and all 100 agents time out.
That is exactly why the training/eval start contract was changed and why this
run is now a legacy baseline.

Deterministic 100-agent checkpoint sweep:

```text
25k:   0/100 finishes
50k:   0/100 finishes
75k:   0/100 finishes
100k: 100/100 finishes
```

The sampled policy was already finishing regularly before 100k; the argmax
policy became reliable only at the final checkpoint. Keep checkpoint
evaluation in future runs instead of assuming every saved policy is deployable.

Zero-shot on 1-4 was 0/100. The next experiment was one shared policy with
mixed-level/curriculum training, not 14 independent PPO policies. That line of
work is now resolved by distillation. The current observation-only 58-input,
256-hidden student clears all 16 levels at 100/100; the retired 74-input
level-conditioned model also cleared 16/16 but required a campaign ID. See
[ppo-distillation.md](ppo-distillation.md).
