# Run artifacts

`runs/` is generated and gitignored. This page is the committed index of what
each active run directory is for and where to look first. Add a row when a run
becomes important; do not commit the run itself.

Sizes are approximate from 2026-10-28.

## Active / reference runs

| Directory | Size | Status | What it is | Start here |
|---|---:|---|---|---|
| `runs/dqn_route_spawns_1_1/` | 6 MB | **incomplete** | DQN route-spawn extract attempt for 1-1. Routes copied; `spawns/` empty; no `dqn/`, no `eval/`. Design pivoted to the start-pool path; this dir is the fallback. | `godot/route_replay.log`; then [`../audit/2026-10-28-dqn-pipeline-review.md`](../audit/2026-10-28-dqn-pipeline-review.md) and [`../audit/2026-10-28-start-pool-clock-bias.md`](../audit/2026-10-28-start-pool-clock-bias.md) |
| `runs/neat_clean_450/` | ~61 MB | active/reference | 1-1 8×24 NEAT collection stopped at gen 25 on the 450-route target. Best **1.6667 s**; **452 clean routes**, 1,486 checkpoint files; filtered to 95 clusters / 216 routes; pools in `pool/`. | `train.console.log`, `filtered/summary.json`, `pool/split.json` |
| `runs/ppo_clean_start_1_1/` | TBD | **running** | Lean 188→954 PPO, 1M timesteps, 8 envs, canonical 0.70 / 13 train-state 0.30. Checkpoints every 100k; canonical + train/holdout pool eval at the end. | `train.console.log`, `preprocessing.json`, `checkpoints/`, `eval/` |
| `runs/neat_1_1_8x24_archive/` | 22 MB | reference | 8×24 NEAT 1-1 archive. Best **2.15 s** at gen 9; 4,243 finishers; 460 clean routes; coverage, strata, PPO start pools. | `best_time.json`, `clean/clean_summary.json`, `strata/strata_summary.json`, `coverage/coverage_summary.json` |
| `runs/neat_1_2_transfer_8x24/` | 12 MB | reference | 1-2 NEAT transfer (resumed 1-1 population at checkpoint 61). Best **3.0167 s**; 3,052 finishers; 12 clean routes; 71/72 starts. | `best_clean.json`, `clean/clean_summary.json`, `strata/strata_summary.json` |
| `runs/neat_1_3_transfer_8x24/` | 15 MB | reference | 1-3 NEAT transfer (resumed 1-2 at checkpoint 97). Best **3.5667 s**; 3,580 finishers; 26 clean routes; 62/86 starts. | `best_clean.json`, `clean/clean_summary.json`, `strata/strata_summary.json` |
| `runs/ppo_multilevel_1_1_1_2_1_3/` | 289 MB | reference | 754 multi-hot PPO across 1-1/1-2/1-3. 200k checkpoint selected: canonical **1.6667 / 2.9667 / 5.2333 s**, full 242-start coverage. 300k regressed. | `best_multilevel_3maps.zip`, `eval_final200k_*_fixed/summary.json`, `eval_final300k_*_fixed/summary.json` |
| `runs/ppo_fresh_1_1_sensor_794/` | 468 MB | reference | 794 sensor PPO 1-1 lineage. Best canonical **2.1333 s** (600k / ar2 final); full strata coverage. Many eval/ablation dirs. | `eval_ar2_final/summary.json`, `eval_600k/summary.json` |
| `runs/ppo_sensor_1_2_from900k/` | 94 MB | reference | 794 sensor 1-2 transfer from the 1-1 900k checkpoint. Canonical **3.0667 s**; full 1-2 strata. | `eval_1_2/summary.json`, `eval_1_1/summary.json` |
| `runs/ppo_sensor_1_3_from1_2/` | 95 MB | reference/negative | 794 sensor 1-3 transfer. Canonical **0/1**; medium 25/46, hard 5/27 at the base eval. Checkpoint evals at 1000k–1100k. | `eval_1_3/summary.json`; compare `eval_1_3_*/summary.json` |
| `runs/distill_1_3_merged_available/` | 19 MB | reference | Distilled 1-3 candidate from the 794 teacher checkpoint `ppo-1075823`; 46,582 observations, BC accuracy 93.29%. Eval: canonical **3.9 s**, medium 43/46, hard 25/27. Not installed. | `merged/summary.json`, `eval_1_3/summary.json` |

## What the files mean

| File | Meaning |
|---|---|
| `best_time.json` / `best_clean.json` | Best NEAT result metadata (time, generation, level, observation, action repeat). |
| `clean/clean_routes.json` | Deduped successful routes with `path_cells`, `actions`, `run_time`, `route_signature`. |
| `clean/clean_summary.json` | Route counts, top genomes, source finisher count. |
| `coverage*/coverage_summary.json` | Start-cell coverage from the top genomes: union finishes, per-stratum counts, canonical. |
| `strata/start_strata.json` | Every standable start with `stratum`, `distance`, `facing`. |
| `strata/strata_summary.json` | Strata counts and coverage/time summaries. |
| `strata/sampling_plan.json` | Proposed canonical/medium/hard mix. |
| `ppo/start_pool.json` | Weighted PPO reset pool (canonical/cell/prefix entries). |
| `ppo/start_pool_summary.json` | Pool counts, weights, focus/heldout starts. |
| `filtered/summary.json` | Route-filter counts: input/unique routes, clusters, kept train/holdout routes. |
| `pool/train_start_pool.json`, `pool/holdout_start_pool.json` | Clean-start state pools for PPO; the holdout file is evaluation-only. |
| `pool/split.json` | Start-pool counts, weights, checkpoint rejection stats, and train/holdout key overlap. |
| `preprocessing.pkl`, `preprocessing.json` | Fitted lean one-hot pipeline and its input/output metadata; load these with the model. |
| `monitor.csv` | SB3 `VecMonitor` episode summary. |
| `episodes.csv` | PPO per-episode rows: start source/kind, finish, timeout, run time, efficiency, flow. |
| `checkpoints/` | SB3 checkpoints (`.zip`). |
| `eval*/summary.json` | Per-stratum aggregate: `starts`, `finished`, `spawn_ok`, `best`, `median`, `worst`, `failures`. |
| `eval*/results.json` | Per-start rows. |
| `godot/*.log` | Raw Godot stdout/stderr for a worker/eval process. |
| `userdata/` | Isolated `HOME`/`XDG` tree for a process. Never read a real save from here. |
| `tensorboard/` | SB3 logs. |

## Naming conventions in run dirs

- `eval_<step>` / `eval_<step>_zero` — checkpoint eval / zero-sensor ablation.
- `eval_<step>_ar2` — action-repeat 2 variant.
- `eval_*_fixed` — eval after the start-jump terminal accounting fix; prefer
  these over the unfixed versions.
- `run_<step>` / `run_<variant>` — training segment.
- `watch_*` — visible playback of a checkpoint.
- `checkpoints/ppo-<steps>.zip` — PPO checkpoint naming.

## Archive

`archive/2026-10-27-cleanup/` contains:

- 20+ historical scripts (linear/decision-tree/random-forest tutorials, old
  PPO lineage/teacher/encoder/topology drivers, old distillation drivers,
  `playground.py`);
- 146 historical run directories (under `archive/runs/`, gitignored and large);
- the historical `docs/model-improvement-experiments.md` experiment log.

Start at `archive/2026-10-27-cleanup/README.md`. `archive/runs/` is intentionally
not tracked; do not try to add it to git.

## Adding a new run

1. Use a descriptive `runs/<experiment>_<level>_<variant>/` name.
2. Keep the standard subdirs (`checkpoints/`, `eval/`, `godot/`, `userdata/`).
3. Record the command, branch/commit, game dirty state, and key result in
   `docs/status.md` or a dated audit.
4. Add a row to this page when the run becomes part of the reference set.
