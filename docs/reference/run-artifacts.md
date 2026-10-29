# Run artifacts

`runs/` is generated and gitignored. This page is the committed index of what
each active run directory is for and where to look first. Add a row when a run
becomes important; do not commit the run itself.

Sizes are approximate from 2026-10-28.

## Active / reference runs

| Directory | Size | Status | What it is | Start here |
|---|---:|---|---|---|
| `runs/neat_clean_450/` | 61 MB | **active/reference** | Latest 1-1 8×24 NEAT collection. Best **1.6667 s**; 452 clean routes; filtered to 95 clusters / 216 routes. Standability-validated pool: **12 train / 8 holdout** starts. | `train.console.log`, `filtered/summary.json`, `pool/split.json` |
| `runs/neat_clean_450_1_2_transfer/` | 20 MB | **complete; first 1-2 pool** | Final 192-genome 1-1 population transferred to 1-2 and evolved for six generations. Best **3.3667 s**; 25 clean / 79 unique routes; 576 checkpoints. Permissive filter: 7 clusters / 19 routes (16 train / 3 holdout), but one cluster dominates 22/32 train starts. Recommended diverse filter: 12 clusters, 9/3 routes, pool **31 train entries (30 state + canonical) / 20 holdout states**. | `evolve.console.log`, `archive/routes.json`, `filtered_j07_cap1/summary.json`, `pool_j07_cap1/split.json` |
| `runs/ppo_clean_start_1_1_fresh/` | 369 MB | **complete; 500k baseline** | Fresh 500k lean PPO from scratch. Canonical **1/1 at 2.15 s**, train **12/12**, holdout **8/8**. | `train.console.log`, `checkpoints/ppo_500000_steps.zip`, `ppo_pilot.zip`, `eval/ppo_500000_steps/summary.json` |
| `runs/ppo_clean_start_1_1_resume_1m/` | 369 MB | **complete; 1M diagnostic** | Continuation from 500k to 1M. 1M: canonical **1/1 at 2.15 s**, train 12/12 median **3.4167 s**, holdout 8/8 median **4.2750 s**. Worse pools than 600k. | `train.console.log`, `ppo_pilot_1m.zip`, `eval/ppo_1000000_steps/summary.json` |
| `runs/ppo_stability_*` | 263 MB | **complete; 8×1/600/480 won sweep** | Five 50k continuations from 500k. Large rollouts fixed the control regression; 8×1/600/480 selected for 100k confirmation. | `train.console.log`, `ppo_pilot_550k.zip`, `eval/ppo_550k/summary.json` |
| `runs/ppo_clean_start_1_1_best_100k/` | 88 MB | **complete; 600k PPO baseline** | 100k confirmation from 500k with 8×1/600/480. Canonical **1/1 at 2.15 s**, train 12/12 median **3.3250 s**, holdout 8/8 median **3.8583 s**; training canonical ≤2.2 s **97.7%**, timeouts **0.04%**. Superseded by DQN 0013. | `train.console.log`, `ppo_pilot_600k.zip`, `eval/ppo_600k/summary.json` |
| `runs/ppo_time_trial_1_1_225k_100k/` | 88 MB | **complete; no speed gain** | 100k canonical-only time-trial from 225k with the new reward. Canonical stays **2.15 s**; train 12/12 median **3.2917 s**; holdout 8/8 median **3.8583 s**. | `train.console.log`, `ppo_pilot_325k.zip`, `eval/ppo_325k_full_train/summary.json` |
| `runs/ppo_time_trial_continuous_1_1_225k_100k/` | 88 MB | **complete; reward ranking fixed, route not learned** | A/B of 0008 with continuous under-gold (`25/s`, cap 50). 1.633 s scores 556.8 vs 2.133 s 546.4, but only one fast episode in 3,114. Canonical 2.15 s; train 3.5583 s; holdout 4.9500 s. | `train.console.log`, `ppo_pilot_325k.zip`, `eval/ppo_325k_full_train/summary.json` |
| `runs/ppo_time_trial_continuous_8x8_225k_100k/` | 88 MB | **complete; no speed gain** | 8×8 (64 envs), 9,600-decision rollout from 225k with continuous under-gold. Canonical 2.1500 s; train 12/12 median 3.4167 s; holdout 8/8 median 4.2750 s. | `train.console.log`, `ppo_pilot_325k.zip`, `eval/ppo_325k_full_train/summary.json` |
| `runs/dqn_clean_start_1_1/` | 76 MB | **complete; negative at 1M** | Fair lean-input 8×8 DQN: 64 envs, 188 → 954, 1M. Canonical timeout; train 8/12 median 2.9750 s; holdout 7/8 median 5.2333 s. | `train.console.log`, `dqn/dqn_final.zip`, `eval/dqn_1m_8x8/summary.json` |
| `runs/dqn_clean_start_1_1_8x1_1m/` | 1.6 GB | **complete; 1.8667 s canonical** | Lean 8×1 DQN, 1M, 3-step returns, 200k replay. Includes saved `replay_buffer_200k.pkl` with 41 canonical 1.633 s episodes. | `train.console.log`, `dqn/dqn_final.zip`, `dqn/replay_buffer_200k.pkl` |
| `runs/dqn_clean_start_1_1_8x1_buffer_finetune/` | 420 MB | **complete; 1.6167 s selected** | Continuation from the 1M DQN with the saved 200k buffer, stopped at 1.853M. Selected 1.85M: canonical **1.6167 s**, train 12/12 median **3.2500 s**, holdout 8/8 median **3.7917 s**. The first 100k stayed at 1.8667 s; the fast route became deterministic by 1.8M. | `train.console.log`, `dqn/dqn_final.zip`, `eval/dqn_1850000_steps/summary.json` |
| `runs/dqn_zero_shot_1_2/` | 1 MB | **complete; zero-shot baseline** | Selected 1.85M 1-1 DQN on 1-2. Canonical **9.1167 s**; train **17/31** finishes median 5.5000 s; holdout **15/20** median 5.9500 s. | `eval/zero_shot_1_2_pools/summary.json` |
| `runs/dqn_multilevel_1_1_1_2/` | 59 MB | **complete pilot; candidate** | 4×1-1 + 4×1-2 fine-tune from the 1.85M DQN, 100k additional steps. 1-1 retained exactly (canonical **1.6167 s**, train 12/12 3.2500 s, holdout 8/8 3.8000 s). 1-2 improved to canonical **3.7500 s**, train **31/31** median 4.3833 s, holdout **20/20** median 4.0500 s. | `train.console.log`, `dqn/dqn_final.zip`, `monitor.csv` |
| `runs/dqn_multilevel_1_1_1_2_100k/` | 59 MB | **complete; 1-2 candidate** | +100k continuation of 0016 at 2.05M. 1-1 still **1.6167 s** canonical, train 12/12 3.2667 s, holdout 8/8 3.8000 s. 1-2 canonical **3.7833 s**, train **31/31** median 4.0167 s, holdout **20/20** median 3.3500 s. | `train.console.log`, `dqn/dqn_final.zip`, `monitor.csv` |
| `runs/dqn_multilevel_1_1_1_2_ddqn_300k/` | 59 MB | **complete; 2.9667 s candidate** | Double-DQN continuation from 2.05M. Final 2.35M failed the 1-1 holdout gate; selected **2.15M**: 1-1 **1.6167 s** canonical, train 12/12 3.2750 s, holdout 8/8 3.7917 s; 1-2 **2.9667 s** canonical, train 31/31 3.9167 s, holdout 20/20 3.1417 s. | `train.console.log`, `dqn/checkpoints/ddqn_2150000_steps.zip`, `dqn/monitor.csv` |
| `runs/dqn_multilevel_1_1_1_2_1_3/` | 20 MB | **stopped; canonical-only fallback** | 0019: 4×1-1 + 2×1-2 + 2×1-3, stopped at 2.274M because 1-3 had only canonical starts. Do not use as a candidate. | `train.console.log`, `dqn/checkpoints/` |
| `runs/neat_clean_450_1_3_from_1_2/` | 16 MB | **complete; zero finishers** | 0020: 1-2 NEAT population evolved on 1-3, 192 population, 300-step horizon, 20 generations. Best fitness 74.67 plateaued; **0/192 finishes**, 0 clean routes. | `evolve.console.log`, `genome_stats.csv` |
| `runs/neat_explore_1_3/` | 64 MB | **complete; 12 finisher routes** | 0021: fresh 384-genome, high-mutation 1-3 search, 30 generations. Best **4.2000 s**, 113 finishers, 12 unique routes / 113 checkpoints, 0 strict clean. | `evolve.console.log`, `archive/routes.json`, `best_genome.pkl` |
| `runs/neat_explore_1_3_long/` | 384 MB | **complete; strong portfolio** | 0022: 40 more generations on 1-3. 220/384 finishes, 304 unique routes, 6,361 checkpoints, 53 clean routes. Best 4.2333 s (unclean); best clean 5.4667 s, clean median 5.9833 s; 13 clusters at Jaccard 0.5. | `evolve.console.log`, `archive/routes.json`, `best_genome.pkl` |

All older PPO/DQN/NEAT-transfer/smoke run directories were removed on
2026-10-28. Their committed experiment manifests remain historical records.

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
| `monitor.csv` | SB3 `VecMonitor` episode summary: `r`, `l`, `t`, plus start/level/run metadata. New runs use this instead of `episodes.csv`. |
| `episodes.csv` | Historical PPO per-episode rows (pre-2026-10-28 harness). New runs use `monitor.csv`. |
| `checkpoints/` | SB3 `CheckpointCallback` files. New PPO runs use `ppo_<steps>_steps.zip`; historical runs use `ppo-<steps>.zip`. |
| `eval/<tag>/summary.json` | One evaluation run: canonical per-level summaries and per-set summaries (`agents`, `finishes`, `spawn_ok`, `best_time`, `median_time`, `timeouts`). |
| `eval/<tag>/*.csv` | Per-start rows for canonical and each configured evaluation set. |
| `eval*/summary.json`, `eval*/results.json` | Historical evaluator output. |
| `eval*/start-pool-*.json` | Historical start-pool summaries. |
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
- `checkpoints/ppo-<steps>.zip` — historical PPO checkpoint naming.
- `checkpoints/ppo_<steps>_steps.zip` — new SB3 `CheckpointCallback` naming.
- `eval/<tag>/` — evaluation output for one model/tag; the tag defaults to the
  model stem.

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
