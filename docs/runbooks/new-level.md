# New-level pipeline

Status: active  
Last updated: 2026-10-30

This is the living procedure for adding one campaign level N after level N-1.
It covers the whole path: NEAT route discovery, strict/near-clean route
selection, start pools, a single-level DDQN expert, evaluation, repair, and
selection.

The old
[`../audit/2026-10-29-level-transfer-playbook.md`](../audit/2026-10-29-level-transfer-playbook.md)
is still useful for the NEAT discovery branches, but it stops at the start
pool. Follow this runbook for the expert half.

## Scope

- One expert per level. An expert is trained on level N only; do not mix old
  levels into the run.
- Experts are teachers for the eventual shared student A. They are not the
  final one-checkpoint product.
- A route archive is level-specific. Transfer the NEAT population, never the
  previous level's route archive.
- The accepted gate is based on the best available clean tier:
  strict-clean first, near-clean only when strict-clean is unavailable.
- All-success routes are a diagnostic sensitivity set, not the accepted gate.

## Pipeline

```text
0. preflight
1. NEAT transfer on N
2. route tiers + filtered routes + train/holdout/eval pools
3. expert training from N-1's selected expert
4. checkpoint sweep on canonical + accepted eval pool
5. repair loop only if the gate fails
6. select, record, hand off to the next level
```

Each stage produces a go/no-go artifact. Do not run the next stage on a failed
gate.

## 0. Preflight

Check:

```text
game repo branch/commit/dirty state
level file and par time
level analysis exists
previous NEAT checkpoint exists
previous selected expert exists
ports are free
runs/ has room
```

Export the level analysis if it is missing (from the game repo):

```bash
cd ../dragon-jump-remaster
python3 tools/export_level_analysis.py 1-9
```

The Python pipeline reads the analysis from
`../dragon-jump-remaster/runs/level_analysis/<level>.json`.

## 1. NEAT transfer

Create an experiment manifest under
`experiments/<id>_neat_transfer_<level>_from_<prev>/` and copy the previous
level's final population, not its route archive.

The standard search settings are:

```text
population:        384
parallel:          8
action repeat:     4
max steps:         300
time fitness:      20
generations:       50
clean-route stop:  25
checkpoint every:  1 generation
rebalance every:   2 generations
observation:       legacy_stick 61
action:            reassert_held_jump
```

Run:

```bash
caffeinate -dimsu .venv/bin/python scripts/genetic/01_neat.py \
  --level <N> --obs-encoding legacy_stick --reassert-held-jump \
  --resume <N-1 final checkpoint> \
  --output-dir runs/neat_transfer_<N>_from_<prev> \
  --archive-dir runs/neat_transfer_<N>_from_<prev>/archive \
  --generations 50 --parallel 8 --action-repeat 4 --speedup 12 \
  --max-steps 300 --record-paths --checkpoint-interval 1 \
  --rebalance-every 2 --rebalance-final --min-clean-routes 25 \
  --time-fitness-weight 20 --port <free port>
```

Go/no-go:

| Result | Next |
|---|---|
| `finishers == 0` | Fresh high-mutation discovery; if that also has zero, change representation/reward/curriculum. |
| `finishers > 0`, strict-clean `== 0` | Extend the seeded search or run a high-mutation cleanup from the best route clusters. Do not build a pool from zero clean routes. |
| strict-clean `>= target` | Go to stage 2. |
| strict-clean low but near-clean available | Use the near-clean fallback explicitly and record the exception. |

## 2. Route tiers and pools

The repeatable post-NEAT step is:

```bash
.venv/bin/python tools/level.py data \
  --level <N> \
  --run-dir runs/neat_transfer_<N>_from_<prev> \
  --generation <G> \
  --analysis ../dragon-jump-remaster/runs/level_analysis/<N>.json \
  --quality auto \
  --min-routes 25 \
  --seed 0 \
  --canonical-weight 0.50 --state-weight 0.50
```

Makefile shortcuts:

```bash
make level-status LEVEL=1-9
make level-data LEVEL=1-9 LEVEL_RUN=runs/neat_transfer_1_9_from_1_8 LEVEL_GEN=<G>
```

`--output-root` writes the derived files elsewhere without modifying the NEAT
archive or its checkpoints. `tools/level.py data` writes:

```text
strict_clean_routes_g<G>.json
near_clean_routes_g<G>.json
all_success_routes_g<G>.json
route_quality_g<G>.json
filtered_<tag>_g<G>/
pool_<tag>_g<G>/
  train_start_pool.json
  holdout_start_pool.json
  eval_<tag>_start_pool.json
level_data_summary.json
```

Quality tiers:

| Tier | Rule | Use |
|---|---|---|
| strict-clean | `revisit_ratio <= 0.10`, `backtracks <= 2`, action flips `<= 0.75`, path length `>= 5` | Accepted gate when available. |
| near-clean | `revisit_ratio <= 0.20`, same backtrack/flip/path rules | Accepted gate only when strict-clean is unavailable. Record the exception. |
| all-success | every de-duplicated finisher | Diagnostic sensitivity set. |

`--quality auto` chooses strict-clean when it meets `--min-routes`, otherwise
near-clean. It never selects all-success automatically.

The eval pool is the deduplicated union of the train and holdout state pools:

```text
unique_eval_states = train_states + holdout_states - shared_state_keys
```

Check the summary for this warning:

```text
fastest cluster c000 (...) is holdout-only; expert training will not see its actions
```

That is not automatically a bug, but it is a decision. On 1-8 the fastest
cluster `c000` was holdout-only, so the expert reproduced the fastest **train**
route (4.4667 s), not the fastest overall NEAT route (4.4167 s). If speed is
the goal, adjust the split or run a separate speed branch; do not describe the
coverage expert as the fastest route.

Use `tools/level.py status <N>` to see the current experiments and summaries.

## 3. Expert training

Create `experiments/<id>_ddqn_<level>_feasible_starts/config.toml` from the
selected run. The essential fields are:

```text
levels:             8 x <N> (single level only)
action repeat:      4
observation:        lean_stick 191 -> 957
resume_from:        <N-1 selected expert>
train_pool:         pool_<tag>_g<G>/train_start_pool.json
holdout_pool:       pool_<tag>_g<G>/holdout_start_pool.json
expert_routes:      filtered_<tag>_g<G>/train_clean_routes.json
expert_data:        runs/<expert-run>/expert_<N>/expert_routes.npz
expert_buffer:      runs/<expert-run>/expert_<N>/stable_expert_buffer.pkl
sets:               pool_<tag>_g<G>/eval_<tag>_start_pool.json
old anchor:         none
```

Replay the train routes in the 191 `lean_stick` environment and build the
stable expert buffer:

```bash
.venv/bin/python tools/collect_expert_routes.py \
  --config experiments/<expert-id>_ddqn_<N>_feasible_starts

.venv/bin/python tools/build_expert_buffer.py \
  --config experiments/<expert-id>_ddqn_<N>_feasible_starts \
  --max-replay-gain 0.25
```

Train:

```bash
.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/<expert-id>_ddqn_<N>_feasible_starts
```

The standard budget is 100k steps with checkpoints every 12.5k. Do not select
`dqn_final.zip` by name; evaluate every checkpoint.

## 4. Evaluation and selection

For every checkpoint:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/<eval-id>_eval_ddqn_<N>_feasible_starts \
  --model runs/<expert-run>/dqn/checkpoints/<checkpoint>.zip \
  --tag <checkpoint>
```

Gate:

```text
canonical:   finish, <= par + 0.30 s unless the user sets a different target
accepted:    every state in eval_<tag>_start_pool.json finishes
diagnostic:  all-success pool is reported separately
safety:      report deaths, timeouts, and spawn_ok separately
```

Selection rule:

1. Reject any checkpoint with a canonical death/timeout or a spawn failure.
2. Prefer the checkpoints that cover the accepted eval pool completely.
3. Among full-coverage checkpoints, choose the fastest canonical time.
4. Repeat the selected checkpoint's canonical run and accepted pool exactly.
5. If the fastest full-coverage checkpoint is one frame above the gate, use a
   short low-LR canonical nudge before accepting a near-miss or retraining.
6. Never use `dqn_final.zip` unless it was evaluated and won.

## 5. Repair loop

Only repair after a full checkpoint sweep. Diagnose first:

| Failure | Repair |
|---|---|
| Canonical only, coverage intact | Canonical-heavy pool or a short low-LR canonical nudge. |
| One known start missing | Targeted suffix buffer from that exact start (`tools/build_start_expert_buffer.py --suffix`). |
| Repair fixes one start but breaks another | Stop stacking repairs; revert to the last full-coverage checkpoint and nudge canonical only. |
| Canonical fast but coverage drops | Prefer coverage first, then canonical nudge. |
| Targeted repair over-corrects (e.g. 33/40) | Reject it; do not extend the same repair. |

The 1-8 sequence is the worked example:

```text
0109 canonical-heavy repair -> 40/40, canonical 4.5167
0112 canonical-only fine-tune -> 4.4667, 39/40
0114 targeted [4,21]/-1 suffix -> 40/40 at 4.5167; 4.4667 at 39/40
0116 targeted [3,21]/-1 suffix -> over-corrected to 33/40; rejected
0118 500-step low-LR canonical nudge -> 4.4667 and 40/40; selected
```

## 6. Record and hand off

Record:

- experiment README and `config.toml`;
- selected checkpoint path, canonical time, accepted coverage, all-success
  diagnostic, deaths/timeouts, spawn_ok;
- final target or "not set";
- game branch/commit/dirty state;
- the next level's seed: this level's final NEAT checkpoint and selected
  expert.

Update `docs/reference/level-ledger.md`, `docs/status.md`, and
`docs/reference/run-artifacts.md` when the run becomes reference material.

## Current next level: 1-10

```text
level:             1-10 (DOUBLE_SHORT)
par:               3.0336 s
provisional gate:  3.3336 s (par + 0.30)
level analysis:    ../dragon-jump-remaster/runs/level_analysis/1-10.json (export first)
NEAT seed:         runs/neat_transfer_1_9_from_1_8/neat-checkpoint-89
expert base:       runs/ddqn_1_9_single_route/dqn/checkpoints/ddqn_2060360_steps.zip
```

1-9 is complete as a **single-route** expert: canonical 2.9500 s, selected
`ddqn_2060360_steps.zip`. Its 11-route pool is diagnostic only. The 1-9
retention check is recorded in the ledger; do not expect old-level coverage
from that checkpoint.

## Related docs

- [Level-transfer playbook](../audit/2026-10-29-level-transfer-playbook.md) —
  NEAT discovery branches.
- [Level ledger](../reference/level-ledger.md) — what is selected for each
  level.
- [Known issues](../known-issues.md) — checkpoint selection, near-clean
  exception, and 1-8 speed/coverage caveat.
- [Run artifacts](../reference/run-artifacts.md) — what each run directory
  contains.
- [Expert portfolio decision](../audit/2026-10-29-expert-portfolio-and-student-decision.md)
  — why experts are teachers for one shared student.
