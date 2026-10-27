# Model improvement experiments

Status: semantic 64-d MAE + fresh 64×64 head matches 3.7500 s at step 0; legacy AE ties 3.8667 s; one-hot AE worse
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

## Training topology (analysis 2026-10-26)

The current PPO runs use **8 Godot processes × 1 agent**. This is the pilot
default, not a measured optimum. The development machine has 10 cores; the
100-agent evaluation runs 100 worlds in one process because it is one episode
with no resets, not because one process is the best training topology. All
workers feed one central PPO policy; there is no per-worker weight copy and no
weight merging.

The game now supports optional per-agent reset indices on the `reset` message
(`agents: [1, 3]`), and `GodotEnv.reset(agents=...)` exposes it. PLaiGROUND now
has a multi-agent `VecEnv` with per-agent auto-reset, and `10_ppo_pilot.py`
accepts `--agents-per-process`. The completed 512 width A/B stayed on 8×1 for
comparability. The 8×8 legacy rerun uses `--agents-per-process 8 --n-envs 8`
(64 agents) and 2.4M total timesteps so each agent gets the old 37.5k-decision
budget (300k / 8).

Measured environment throughput (1-11, multi-hot 754, random actions,
`action_repeat=4`, `speedup=8`, 500 decisions/agent, 2 repeats):

| Agents/process | Aggregate decisions/s | Per-agent |
|---:|---:|---:|
| 1 | 116.0 | 116.0 |
| 2 | 225.7 | 112.9 |
| 4 | 434.7 | 108.7 |
| **8** | **791.7** | **99.0** |
| 16 | 753.6 | 47.1 |
| 32 | 626.7 | 19.6 |
| 64 | 476.6 | 7.4 |
| 100 | 405.1 | 4.1 |

**8 agents/process is the knee.** More agents than that divide the process’s
capacity instead of increasing it. The next measurement is the process-count
sweep at 8 agents/process (1, 2, 4, 8). Game-side throughput work is tracked in
the game repo’s A26: profiling, training-only no-visuals, packed binary/reset-only
observations, cached multi-hot cell features, and a conditional Rapier/threaded
physics spike only if profiling shows physics is dominant.

Do not mix topology or throughput changes into the width comparison.

### 2026-10-27 follow-up: process sweep + binary observations

A controlled 8-agents/process environment sweep (`--action-seed`) found 8×7
(~1,912 decisions/s) and 8×8 (~1,930) effectively tied, with 8×1 at ~795. The
environment-only knee remains around 8 agents/process; more processes scale
sublinearly because the 10-core box saturates.

The game then moved step/reset observations from JSON numbers to length-prefixed
float32 frames; info remains complete/columnar. Short 30k PPO smokes (1-11,
multi-hot, 64×64, `--skip-eval`) favored **16 agents/session**:

| Topology | Actual timesteps | Wall clock | Approx. decisions/s |
|---|---:|---:|---:|
| 8×8 | 38,400 | 66.7 s | ~575 |
| 16×1 | 30,960 | 45.1 s | ~686 |
| 16×2 | 33,600 | 38.4 / 56.2 s | ~874 / ~598 |
| 16×3 | 36,000 | 47.7 s | ~754 |
| 16×4 | 38,400 | 49.2 / 64.6 s | ~780 / ~594 |
| 16×8 | 38,400 | 67.9 s | ~565 |
| 24×2 | 36,000 | 56.3 s | ~640 |

These are noisy, action-stream-uncontrolled, non-learning smokes; repeat variance
was often >30%. The promising candidate is **16 agents/session on 2–4 processes**,
not 16×8. A longer wall-clock-to-target run with interleaved topologies is still
required. Reset-only observations were tried and reverted after a regression.

### 2026-10-27 current-code topology + cosmetics

Fresh current-code runs (1-11, multi-hot, 64×64, CPU, `action_repeat=4`,
`speedup=8`) supersede the older cross-code comparisons:

- One-process environment-only: 1 agent 115/s, 2 224/s, 4 427/s, 8 790/s,
  **16 1,106/s**, 24 963/s, 32 838/s. The per-instance knee is 16.
- Environment-only process scaling: **8×8 ~2,301/s**, 8×6 ~2,259/s, 8×4 ~1,834/s,
  16×3 ~1,549/s, 16×4 ~1,318/s.
- Fresh 30k PPO: 8×8 64.1 s (~468/s), **16×2 36.7 s (~817/s)**, 16×3 44.9 s
  (~669/s), 16×4 45.7 s (~656/s). PPO flips the winner to 16×2; 8×8 spent
  136.7 s of system time vs 43.7 s.
- Synchronizer timers at 16 agents: physics ~0.57 ms/decision, observations
  ~0.07, binary packing ~0.06, wait ~0.02, info ~0.01.
- New `--training-no-cosmetics`/`--no-cosmetics` A/B skips effect scenes, player
  particles/SFX, and player animations. Environment-only: 16 agents/process
  954→1,230/s (+29%), 16×2 1,391→1,644/s (+18%). A four-run 30k 16×2 PPO A/B was
  neutral: baseline median 39.71 s vs 39.65 s no-cosmetics (means 39.81 vs
  40.55). The flag is experimental and off by default; the current speedup pass
  is closed.

These are still noisy, short, no-eval throughput smokes; use a longer
wall-clock-to-target run before locking the training topology.

### PPO-first robustness plan (2026-10-27)

Speed experiments are closed. The current exported policy clears canonical 16/16 but stalls on
unseen obstacles (e.g. a block two cells ahead); the block is inside the 7×7 window, so this is
route/recovery overfit rather than observation blindness. The first proof is **1-1 only**:

1. Export the 1-1 level analysis; enumerate clean standable/reachable spawn cells.
2. Run the current 1.7000 s 512×512 expert, record its route, and hold out the right/long route
   cells. Build a train spawn pool from the remaining cells and a holdout pool from the long route.
3. Baseline the current expert from default, train-pool, and holdout spawns; record finish rate,
   time, and route regret.
4. Train PPO from the expert at **16 agents × 2 processes** on multi-hot 754 with canonical +
   randomized spawns. Start with a canonical/near-route curriculum and exclude holdout cells.
5. Evaluate default-spawn best time, holdout-route finish/time and route choice, random-spawn
   recovery, and canonical regression against the original expert.

Then scale to 1-11 and all 16 levels. Only after a PPO baseline exists, A/B DDQN/off-policy with
level-balanced replay. Do not use all-unique-state replay or strict sequential carry.

**Result (2026-10-27):** both the 16×2 and 8×8 from-scratch runs failed the canonical gate, so the
from-scratch branch is closed. DDQN/off-policy remains deferred. The convention baseline is the
frozen 58/256 student; the only remaining model work is a bounded same-contract DAgger pass after
the demo gate (1.7 multi-hot 1-1 teacher → 58 student, train spawns, all-level replay). MAE remains
a frozen representation experiment until a v1.1 spec-3/ONNX runtime is chosen; the flat JSON
runtime cannot run the encoder.

### 2026-10-27 from-scratch 1-1 probe (negative; branch closed)

A fresh 512×512 multi-hot 16×2 PPO run was trained on canonical 1-1 starts only for ~1M steps
(300k + 700k continuation), with the right/long route held out from evaluation. Canonical time
plateaued at **2.333 s** after ~350k and stayed there; it never reached the existing **1.7000 s**
expert. Holdout recovery was **0/10 baseline**, briefly **4/10 at ~400k**, then fell back to
**0–1/10**. No checkpoint met the canonical **≤1.75 s** gate while improving holdout.

This is a topology/learning result, not a spawn-sampling bug: the 16×2 throughput winner does not
learn the 1-1 1.7 s route from scratch within 1M steps. The checkpoint selection originally used
the baseline time as its target and briefly accepted the 2.333 s plateau; the absolute 1.75 s gate
is now fixed. Do not treat 2.333 s as a deployable result.

The 8×8 follow-up (64 agents, 2.4M steps, 512×512, canonical-only, seed 47) also failed the
absolute gate. The best canonical checkpoint was **2.133 s at 1.5M**; the final 2.4M fresh-env
canonical run failed while training episodes hovered **1.8–2.15 s**. Holdout was **6/10 final**
(7/10 at 1.5M). No checkpoint was promotable. Artifacts: `runs/spawn_1_1/scratch_8x8_2400k/`
(log, checkpoints, episodes) and `runs/spawn_1_1/pool.json` (21/10 split). **From-scratch PPO is
closed**; the convention baseline is the bundled 58/256 DAgger student (16/16, zero deaths,
1-1 2.35 s). One bounded
same-contract baseline-v2 pass may run only after the demo gate: cross-encoding DAgger from the
1.7 multi-hot 1-1 teacher into the 58 student, train spawns only, all-level replay, canonical
anchor, and a 16/16 + canonical ≤2.0 s + holdout ≥ baseline gate. No PPO seed/topology chase,
DDQN, MAE, multi-hot install, or teacher sweep before the convention.

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

### Warm-start requirement observed

A fresh multi-hot PPO run from random weights on 1-11 (`300k` cap, 8 envs,
`reward-profile=time`, eval every 25k, patience 3) stopped at 75k because every
100-agent eval was **0/100 finishes**. That run is kept under
`runs/teacher_multi_hot_scratch/1-11` as a negative result: the larger semantic
observation does not remove the need for a movement/finish curriculum.

The practical path is therefore:

1. Collect multi-hot observations while the legacy 58-value champions drive
   (`--emit_legacy_obs=true` exposes both vectors).
2. Train a 754-input behavior-cloned student from those labels.
3. Warm-start per-level PPO from that checkpoint (`10_ppo_pilot.py
   --init-student`), with `--baseline-eval` seeding eval patience from the
   cloned policy so a bad PPO run cannot replace it.
4. Fine-tune with `src/17_multi_hot_teachers.py`; compare each result against
   the legacy champion and keep the faster eligible teacher per level.

This is an engineering response to the observed plateau, not evidence yet that
multi-hot is better.

### Teacher result (negative, 2026-10-26)

The old 8.05 s teacher was not a ceiling. A controlled 1-11 A/B used 300k
steps, 8 envs, 64×64, the existing `time` reward, `action_repeat=4`, seed 42,
and deterministic 100-agent evals:

| Arm | Result |
|---|---:|
| fresh legacy 58, 64×64 | **3.8667 s, 100/100** |
| legacy 58, 64×64, 8×8 (64 agents) | **3.8667 s** at 600k; tied, then plateaued |
| multi-hot 754, 64×64 | **0/100**, training reward stuck at timeout |
| multi-hot 754, 256×256 | 7.0833 s at 300k; **4.4667 s** after extension to 775k |
| multi-hot 754, 512×512 | **0/100** at 300k; one stochastic finish at ~17.5k, then only timeouts/deaths; no best checkpoint |
| multi-hot 754, 512×512, 8×8 | **3.7500 s, 100/100** at 2.0M; final 2.4M = 3.7500 |
| old legacy 1-11 champion | 8.05 s |

The 58 run first cleared at 125k and reached 3.8667 at 200k. The 256 multi-hot
run needed 275k for its first 7.08 clear, then extended to 775k and settled at
4.4667; its training reward rose to ~125 and training median clear time was
~4.75 s. It never matched the 58 baseline. The 512×512 from-scratch arm failed
at every eval (0/100 from 25k through 300k); training had one stochastic finish
at ~17.5k (8.98 s), then no further finishes (1,960 timeouts / 60 deaths;
timeouts returned 37.15), and no best checkpoint. It was not extended. Width is
non-monotonic: 64 and 512 fail, 256 learns; 350×350 is the remaining untested
width. The earlier 512-wide behavior clone result (stuck at 9.65 s) is a
different experiment.

The legacy **8×8 topology** (64 agents, 2.4M total = 37.5k decisions/agent)
reached **3.8667 s, 100/100** at 600k (~9.4k/agent), then plateaued. Training
mean return sat at ~115–118 and median training clear at ~4.75–4.85 s; later
evals were 4.00, 5.08, 5.08, 5.22, 4.48, 4.55. It was stopped at ~1.286M after
six no-improvement evals. It ties the 8×1 best rather than beating it, but
reaches that time with much less per-agent experience.

The **512×512 multi-hot 8×8** arm then broke the ceiling: **3.7500 s, 100/100**
at 2.0M (0 deaths/timeouts, efficiency/flow 1.0), 2.1M tied, and the final 2.4M
eval also returned 3.7500. That is 0.1167 s faster than the 3.8667 s legacy
ceiling. The per-level specialist sweep (`src/20_train_all_levels_8x8.py`) is
paused at 1-5; it uses tiered caps/patience (1.2M/5, 1.8M/8, 2.4M/13) and logs
`no_clear`/`regression` instead of aborting. Completed: **1-1 1.7000 s** (old
2.35), **1-2 3.0167 s** (tied), **1-3 3.0000 s** (old 3.23); **1-4 regressed to
5.7167 s** vs 4.25 and needs a retry.

The continual-learning alternative (A29) is paused:
`src/21_curriculum_8x8.py` starts from the 1-1 best, fine-tunes the next level for
300k, evaluates every seen level with 100 agents, and replays a forgotten level
for 50k before advancing. It accepts only when all seen levels clear; otherwise
it retries that level rather than restarting the campaign.

A faster legacy-input encoder experiment is now running (A30): keep the 58-value
contract, but parse the 49 tile IDs as a 7×7 grid with a tile embedding + 2-layer
CNN, then append the 9 scalar tail values. `--grid-encoder legacy_cnn` selects
`LegacyGridExtractor`; the baseline is the 8×8 legacy MLP at **3.8667 s**. This is
teacher-only until a spec-3/ONNX runtime exists.

**Frozen grid autoencoder arm (A30 follow-up, 2026-10-26).** The joint CNN arm
was stopped at 200k after four deterministic evals stayed **0/100**; training
episodes were reaching 4–6 s but the deployed argmax policy never cleared.
The follow-up separates representation learning from control:

- **Static level windows.** `src/23_generate_level_windows.py` parses the 16
  campaign level codes, reproduces the game's border fill and stacked-cell
  behavior, and slides a 7×7 window over every cell. This produced **8,857
  unique static grids**. Every one of the **3,724** real agent grids is present
  in that static set. A clean all-real holdout trained on the **5,133 static
  grids that are not real states** reaches **96.55% cell / 30.1% exact**; this
  is the generalization caveat, not the final encoder result.
- **Autoencoder.** The 17-token grid autoencoder (tile IDs 0–15 plus `-1`/void
  = 16) uses a 64-d latent and a convolutional decoder. The final encoder is
  trained on all **8,857** corrected static grids. On its random static
  validation split it reaches **99.66% cell / 87.0% exact**; on the real agent
  grids (which are in training) it reaches **99.95% cell / 97.8% exact**. The
  64-d encoder is frozen.
- **Behavior-cloning head.** The frozen latent is concatenated with the nine
  exact tail values and fed to a **64×64 tanh MLP**. Training uses **126,152
  rows / 23,854 unique 1-11 states**, including 13,970 fresh on-policy samples
  from the **3.8667 s** teacher; labels are the teacher probabilities. Validation
  action agreement is **99.50%**.
- **SB3 extractor reload bug.** SB3 re-initializes a custom feature extractor
  after policy construction. The first frozen-AE PPO runs therefore used a
  random encoder and are **invalid** for this route. `FrozenLegacyAutoencoderExtractor.reload_encoder()`
  restores the checkpoint after construction/load; verification shows a
  **0.0** weight difference. With the corrected extractor, the fixed-AE BC head
  alone evaluates at **100/100, 3.8667 s, 0 deaths/timeouts, efficiency/flow 1.0**.
- **PPO fine-tune.** The earlier 3.8667 s head was fine-tuned for 100k: the 50k
  eval tied **3.8667 s, 100/100**, then the 100k eval regressed to **0/100** as
  PPO moved away from the behavior-cloned policy. The run was stopped and the
  baseline checkpoint kept. The corrected-AE head also matches **3.8667 s at
  step 0**; the final comparison is a tie with the original 8×8 MLP, reached
  without PPO exploration. The original MLP first cleared at ~125k.
- **One-hot ablation (negative, 2026-10-26).** A `7×7×17` one-hot conv
  autoencoder with the same 64-d latent, conv widths, data split, and class
  loss is worse than the learned tile embedding. On the clean all-real holdout
  it reaches **95.44% cell / 21.75% exact**, versus **96.55% / 30.1%** for the
  embedding AE. Trained on all 8,857 static grids, it reaches **99.52% cell /
  80.18% exact** on real states, versus **99.95% / 97.8%** for the embedding AE.
  Explicit one-hot channels do not preserve more information through the
  bottleneck; the learned embedding is retained and no downstream one-hot run
  is scheduled.

Implementation: `src/23_generate_level_windows.py`, `src/22_train_legacy_autoencoder.py`,
`src/wrappers/legacy_autoencoder.py`, `--grid-encoder legacy_autoencoder`, and
`--init-head`. Teacher-only; no runtime or installation change.

```bash
# 1. Static 7x7 windows + real agent grids
.venv/bin/python src/23_generate_level_windows.py \
  --center-mode all \
  --output-dir runs/legacy_autoencoder/static_windows_fixed_s42

# 2. Clean all-real holdout: train on static grids that are not real states
.venv/bin/python - <<'PY'
import numpy as np
from pathlib import Path
base = Path("runs/legacy_autoencoder/static_windows_fixed_s42")
static = np.load(base / "static_grids.npy")
real = np.load(base / "real_grids.npy")
real_keys = {row.tobytes() for row in real}
train = static[[row.tobytes() not in real_keys for row in static]]
np.save(base / "static_train_grids.npy", train)
PY
.venv/bin/python src/22_train_legacy_autoencoder.py \
  --device cpu \
  --grids-path runs/legacy_autoencoder/static_windows_fixed_s42/static_train_grids.npy \
  --holdout-grids-path runs/legacy_autoencoder/static_windows_fixed_s42/real_grids.npy \
  --output-dir runs/legacy_autoencoder/static_ae64_fixed_holdout_s42 \
  --latent-dim 64 --epochs 40 --autoencoder-patience 5 --skip-bc

# 3. Final autoencoder on all corrected static grids
.venv/bin/python src/22_train_legacy_autoencoder.py \
  --device cpu \
  --grids-path runs/legacy_autoencoder/static_windows_fixed_s42/static_grids.npy \
  --output-dir runs/legacy_autoencoder/static_ae64_fixed_s42 \
  --latent-dim 64 --epochs 40 --autoencoder-patience 5 --skip-bc

# 4. On-policy teacher states + frozen-encoder BC head
.venv/bin/python src/12_ppo_distill.py collect \
  --levels 1-11 --episodes 2 --agents 100 --teacher-root runs/teacher_best \
  --output-dir runs/teacher_best_collect
.venv/bin/python src/22_train_legacy_autoencoder.py \
  --device cpu \
  --grids-path runs/legacy_autoencoder/static_windows_fixed_s42/static_grids.npy \
  --load-autoencoder runs/legacy_autoencoder/static_ae64_fixed_s42/autoencoder.pt \
  --data-root runs --teacher-model runs/ab_1-11/legacy_w64_s42/best_model.zip \
  --bc-level 1-11 --bc-epochs 150 --head-patience 20 \
  --output-dir runs/legacy_autoencoder/static_ae64_fixed_teacher_head_s42
```



### Semantic multi-hot MAE (2026-10-26)

The semantic 754-value contract is the better shared-encoder input. The first
MAE uses the **7×7×15 grid** (735 values) and bypasses the **19 exact tail
values**. Random cells are masked during training; the encoder gets a mask
channel.

Data: **442,816 rows**, deduplicated to **7,412 unique semantic grids** across
all 16 levels.

Held-out-level test — train on the other 15 levels, validate on 1-11:

- train grids: **4,532**
- held-out 1-11 grids: **2,880**
- bit accuracy: **99.40%**
- exact-grid (all 12 binary channels, all 49 cells): **8.47%**
- continuous-channel MAE: **0.049**

The strict exact-grid metric is harsh: one rare bit mismatch fails the whole
grid. Per-bit information is high, but rare dynamic bits are not perfectly
preserved.

Full encoder — train on all 16 levels:

- **99.81% bit accuracy**
- **44.4% exact-grid** on the random holdout
- **0.027** continuous MAE

A 128-d latent was worse on the held-out test (**99.32% bit / 4.83% exact**).
The 64-d latent is retained.

**Reuse test.** Freeze the 64-d semantic encoder, concatenate the exact 19-value
tail, and train a fresh **64×64** head on 18,197 unique 1-11 states, relabelled
from the **3.7500 s** 512×512 teacher:

- head action agreement: **95.38%**
- deterministic 100-agent eval: **100/100, 3.7500 s**
- 0 deaths, 0 timeouts, efficiency/flow 1.0

This matches the best known 1-11 time with a **64-d latent + 64×64 head**, at
**step 0**, versus the 512×512 teacher's 2.4M PPO steps. The comparison is still
behavior cloning, not from-scratch RL, but it demonstrates that the semantic
MAE is a viable reusable state encoder.

Artifacts:

```text
runs/legacy_autoencoder/semantic_mae64_holdout_1-11_s42/autoencoder.pt
runs/legacy_autoencoder/semantic_mae64_all_s42/autoencoder.pt
runs/legacy_autoencoder/semantic_mae64_head_1-11_s42/policy_head.pt
runs/legacy_autoencoder/semantic_mae64_head_1-11_s42/head_only.zip
```

```bash
# Held-out-level semantic MAE
.venv/bin/python src/24_train_semantic_autoencoder.py \
  --device cpu --holdout-levels 1-11 --latent-dim 64 \
  --output-dir runs/legacy_autoencoder/semantic_mae64_holdout_1-11_s42

# Final all-level semantic MAE
.venv/bin/python src/24_train_semantic_autoencoder.py \
  --device cpu --holdout-levels "" --latent-dim 64 \
  --output-dir runs/legacy_autoencoder/semantic_mae64_all_s42

# Frozen semantic latent + 19 tail -> 64x64 head, labels from the 3.75 teacher
.venv/bin/python src/25_train_semantic_head.py \
  --device cpu \
  --autoencoder-path runs/legacy_autoencoder/semantic_mae64_all_s42/autoencoder.pt \
  --teacher-model runs/topology/multi_hot_8x8_w512_s42/best_model.zip \
  --output-dir runs/legacy_autoencoder/semantic_mae64_head_1-11_s42
```

No VAE was needed for this result. A VAE would model the observed latent
distribution, not create missing rare states; the masked autoencoder plus exact
tail bypass already preserves enough control information. A VAE remains a
possible later generative/uncertainty experiment, not the first shared encoder.

**1-2 accepted:** 1-1 stayed **1.7000 s**, and 1-2 reached **3.0000 s**
(specialist 3.0167) after one replay round. 1-3 is running from that snapshot.
This is the first evidence that the sequential policy can add a level without
losing the previous one.

**Comparison discipline:** use the 64×64 legacy **3.8667 s** as the performance
ceiling only, not as a learning-speed baseline. The 512×512 model has a larger
input and network, so compare its learning curve only with another 512 run
(especially 512 8×8 vs 512 8×1). Cross-model comparisons happen only at the
final deterministic 100-agent eval after both models clear.

Reward-shaping next steps are in the game repo's **A27**: do not use a flat
+500 finish bonus; test time-scaled terminal weights/exponents and per-step
pressure one variable at a time before any CMA-ES/GA-style weight search.



The behavior-clone route also plateaued: global 256 clone failed 1-2/1-11/1-14/
1-15; 1-11-only clones stayed 9.65 through DAgger and 512 width; PPO from the
9.65 clone collapsed to 0/100 at the 25k eval. The 512 width in that clone did
not help. The student therefore used the legacy teacher set.

A temporary **sparse-time reward** (time pressure + finish-time + death,
no progress/efficiency) was also tested on the legacy 58, 64×64 setup. It
stayed at the timeout return (~-15) and never cleared 1-11; the profile and code
were removed. This confirms the progress/efficiency shaping is required for
learning.

### Student result (negative, 2026-10-26)

The final 754-input student was a 256×256 DAgger distillation trained on
375,556 samples (two episodes per level from the legacy champions, plus 84,164
DAgger states from 1-2, 1-3, 1-4, 1-11, 1-14, and 1-15):

```text
13/16 levels clear at 100/100, 0 deaths
failures: 1-11, 1-14, 1-15
regressions vs legacy student: 1-3 3.50 vs 3.233; 1-10 6.217 vs 4.467;
                              1-12 6.10 vs 5.967
candidate: runs/multi_hot_base/student_multi_hot_dagger.pt
JSON:      runs/multi_hot_base/export/student_multi_hot_dagger.json
           spec 2 / 754 / action_repeat 4 / 5.8 MB
```

The candidate does **not** clear the 16-level gate and was not installed. The
256-wide global clone alone cleared 12/16 (failed 1-2, 1-11, 1-14, 1-15); a
512-wide clone fixed 1-2 but regressed 1-8 to 8.82 s and 1-10 to 6.22 s. Width
alone is not the fix.

The bundled game model remains `student_obs58_256.json` (legacy 58-value
DAgger, 16/16, 78.583 s campaign sum). The multi-hot branch and spec-2 loader
remain available for the next experiment.

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

### 0. Policy seed vs environment seed (proposed)

`--seed` currently feeds both the PPO/policy seed and the base environment
seed; the 8 training workers get sequential env seeds. Before calling env-seed
variation “more scenarios,” freeze one policy and evaluate it across several
env seeds with the 100-agent protocol. If the times vary materially, add an
env-seed robustness/fine-tune pass with the policy seed fixed. If they do not,
use randomized starts or takeover/DAgger states instead. This is not
implemented and must not change fixed-seed A/B comparisons.

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

### Route checkpoints (first 1-1 checkpoint eval 2026-10-27)

The game exports its real level analysis and accepts clean standable reset
spawns. The first 1-1 checkpoint evaluation is run with **21 train / 10
holdout** clean standable cells (right/long route as holdout). The 1.7 s expert
gets **10/21 train and 5–6/10 holdout**; expert-initialized 16×2 reaches
**7–8/10 holdout** but regresses canonical to 2.4 s; from-scratch 16×2 and 8×8
fail canonical. No checkpoint is promotable. Route-choice testing and the 1-11
speedrun holdout remain unrun; the holdout is a spawn checkpoint, not route
choice with every route open.

Game side:

```bash
python3 tools/export_level_analysis.py 1-1
python3 tools/validate_route_spawns.py \
  runs/level_analysis/1-1.json path/to/routes-1-1.json
python3 tools/visualize_level_analysis.py --route path/to/routes-1-1.json \
  runs/level_analysis/1-1.json
```

Python side:

```python
env.reset(spawns=[{"agent": 0, "cell": [10, 14], "facing": 1}])
env.reset(agents=[0], spawns=[{"agent": 0, "cell": [20, 14], "facing": 1}])
# MultiAgentGodotVecEnv: map global env index -> {"cell": [x, y], "facing": 1}
vec_env.set_spawns({0: {"cell": [10, 14], "facing": 1}})
```

`spawns` only restores cell and facing; velocity, powerups, dissolve timers,
and elapsed run time are not snapshotted. Missing/invalid entries fall back to
the canonical level start. Route-forced level variants/blockers, arbitrary
state snapshots, and PPO route-start training are deferred.

The first 1-1 short/long route holdout is run; next is a 1-11 speedrun holdout
using the existing model zoo (3.75, 4.4667, 4.6167, 8.05, 9.53). Measure finish
rate, segment time, and route-specific cell coverage; a checkpoint tests
execution from an unseen route state, not route choice with all routes open. See
the game repo's `docs/technical/training.md` § 1-1 spawn evaluation and A31/A32.

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
