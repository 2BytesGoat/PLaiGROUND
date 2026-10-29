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
| `runs/ddqn_anchor_1_1_1_2/` | 4.3 GB | **complete; protected anchor** | 0030: 200k state-aware 1-1/1-2 transitions from the 2.15M model, 3,675 state buckets, return-to-go and Q_old gaps. 141,249 positive gaps above the 0.90 threshold (0.12649). Includes smoke buffers. | `replay_buffer_200k.pkl`, `replay_buffer_200k.json`, `collect_gap.console.log` |
| `runs/ddqn_protected_smoke/` | 12 MB | **smoke passed** | 2k-step protected-buffer smoke test with 200k anchor and 400k online buffer; loaded successfully. | `train.console.log`, `dqn/dqn_final.zip` |
| `runs/ddqn_protected_1_1_1_2_1_3/` | 156 MB | **complete; selected 2.225M** | 0031: 300k continuation from 2.15M with 25% state/Q-gap anchor. Selected **2.225M**: 1-1 1.6167 / 12/12 3.2667 / 8/8 3.7917; 1-2 **2.9667** / 31/31 4.3500 / 20/20 3.1333; 1-3 3.2500 / 45/45 5.2667 / 23/23 5.3333. Final 2.45M not selected; 2.30/2.325/2.35 failed 1-1 holdout. | `train.console.log`, `train.interrupted_sampler_cache.log`, `dqn/checkpoints/ddqn_2225000_steps.zip`, `eval_logs/`, `eval/*/summary.json` |
| `runs/dqn_multilevel_1_1_1_2_1_3_ddqn_300k/` | 155 MB | **complete; prior 1-3 candidate** | 0025: 300k additional steps from 2.15M with 4×1-3 + 2×1-2 + 2×1-1 and an 88-start combined pool. Selected **2.30M**: 1-1 1.6167 / 12/12 3.2750 / 8/8 3.8000; 1-2 3.8667 / 31/31 4.0000 / 20/20 3.1667; 1-3 3.1000 / 45/45 5.2000 / 23/23 5.4500. Final 2.45M failed 1-1 holdout. | `train.console.log`, `dqn/checkpoints/ddqn_2300000_steps.zip`, `eval_*.log`, `eval/*/summary.json` |
| `runs/neat_frequent_rebalance_1_3/` | 203 MB | **complete; 1-4 seed** | 0024: live rebalancing every 2 generations from fixed checkpoint 66; merged 0022+0023 archives. Final 384/384 finishes, 3 speed / 78 median / 227 slow / 76 exploration; 311 deployable genomes; 655 routes / 105 clean / 60 clusters; archive best clean 4.7667 s. Final checkpoint 76; separate seed evaluation 384/384. | `history.csv`, `genome_stats.csv`, `rebalance-*.json`, `archive/routes.json` |
| `runs/neat_frequent_rebalance_1_3_seed_eval/` | 27 MB | **reference; seed gate** | Independent evaluation of checkpoint 76: **384/384 finishes**, best 4.2333 s, best clean 5.4667 s, bins 3 speed / 78 median / 228 slow / 75 >10 s. The 4.7667 s clean route remains in the archive. | `genome_stats.csv`, `best_*.json` |
| `runs/neat_transfer_1_4_from_1_3/` | 12 MB | **complete; failed transfer** | 0035: checkpoint 76 on 1-4. Zero-shot 1/384, 10.5333 s. Twenty generations ended 6/384 finishes, best 10.5333 s, 4 routes, **0 clean routes**. No 1-4 pool built. | `zero_shot.console.log`, `train.console.log`, `archive/routes.json`, `neat-checkpoint-96` |
| `runs/neat_explore_1_4_high_mutation/` | 3.0 MB | **complete; no finishers** | 0036: fresh high-mutation 384-genome search on 1-4, 30 generations, `--record-paths`. **0/384** finishes, 0 routes, 0 clean routes; best genome timed out at fitness 134.25. No pool built. | `train.console.log`, `neat-checkpoint-30`, `history.csv`, `best_genome.pkl` |
| `runs/neat_seeded_1_4_from_0035/` | 12 MB | **stopped; seeded discovery** | 0037: four rough 0035 finisher genomes seeded as elites into 384 high-mutation genomes. Generation 1: **40/384** finishes, **5.9167 s**. Stopped at generation 25: 17 routes, 0 clean, live collapsed to 2/384 because no rebalancing. Superseded by 0038. | `seed-checkpoint-0`, `train.console.log`, `archive/routes.json`, `neat-checkpoint-24` |
| `runs/neat_seeded_1_4_stick/` | 14 MB | **complete; clean 1-4 pool** | 0038: top four 0037 routes seeded into **legacy_stick** (61 values), `reassert_held_jump`, rebalance every 2 generations. Generation 1: **173/384** finishes, **5.7667 s**, **10 clean routes** / 4 clusters. Filtered to 5 train / 5 holdout routes; pool 8 train / 3 holdout states; all 11 starts reset-smoked. Best 5.7667 s is holdout. | `seed-checkpoint-0`, `train.console.log`, `archive/routes.json`, `filtered/summary.json`, `pool/split.json`, `pool/multilevel_train_start_pool.json` |
| `runs/ddqn_protected_1_1_1_2_1_3_1_4/` | 156 MB | **complete; 1-4 negative** | 0039: 300k protected DDQN from selected 2.225M. Final 1-4 **0/1 canonical, 1/9 train, 1/3 holdout** at 6.7833 s; old levels mostly retained (1-1 full, 1-2 30/31 + 19/20, 1-3 full). No 1-4 checkpoint selected. | `train.console.log`, `eval_logs/`, `dqn/checkpoints/`, `dqn/monitor.csv` |
| `runs/neat_seeded_1_4_stick_full/` | 0.9 MB | **stopped; diagnostic** | 0044: first live-finisher rebalance attempt. Generation 1: 173/384 finishes, 10 clean routes; rebalance saw 173 live finishers but only 26 survived reproduction. Stopped after generation 3; generation 2 reached 246/384. | `train.console.log`, `rebalance-1.json`, `neat-checkpoint-1` |
| `runs/neat_seeded_1_4_stick_full_finishers/` | **stopped; diagnostic** | 0045: merged evaluated finishers after reproduction, but selected only 123/173 because >10 s finishers were classified as exploration. Peak finish rate 380/384 at generations 6 and 8. | `train.console.log`, `rebalance-*.json`, `archive/` |
| `runs/neat_seeded_1_4_stick_full_population/` | **complete; 1-4 seed** | 0046: preserved every finisher including >10 s routes. Final 380/384, peak 383/384, best 4.9833 s, 75 clean routes; independent checkpoint-30 eval 382/384. | `train.console.log`, `rebalance-*.json`, `history.csv`, `archive/`, `filtered/`, `pool/` |
| `runs/neat_transfer_1_5_from_1_4/` | **complete; diagnostic** | 0047: 1-4 → 1-5 probe 6/384; seeded high-mutation evolution 384/384 at generation 20, best 4.6833 s, 9 clean routes. Not carried into DDQN. | `zero_shot.console.log`, `train.console.log`, `neat-checkpoint-20`, `archive/routes.json` |
| `runs/ddqn_protected_1_1_1_2_1_3_1_4_full/` | **stopped diagnostic** | 0048: 188-input DDQN on the 0046 pool. 2.525M: **0/1 canonical, 1/11 train, 0/2 holdout**; stopped at ~2.626M. | `train.console.log`, `dqn/checkpoints/`, `eval/0048_2525/` |
| `runs/ddqn_protected_1_1_1_2_1_3_1_4_lean_stick/` | **stopped; 100k probe** | 0053: lean_stick 191 (188 + 3 flags), 957 model inputs, 188 weights expanded. 100k: **2/11 train, 0/1 canonical, 0/2 holdout**; 1-3 regressed (41/45, 19/23). **1-4 eval used legacy action semantics; rerun before treating as final.** | `train.console.log`, `dqn/checkpoints/ddqn_100000_steps.zip`, `eval/0053_100k_matrix.json` |
| `runs/ddqn_fresh_1_1_lean_stick/` | **complete; 1M selected** | 0058: fresh lean_stick 191 DDQN on 1-1 only, 8×1, 1M steps, discovered 12/8 starts, no 188 weights. Corrected eval: **1.5667 canonical, 12/12 train 3.2583, 8/8 holdout 3.8000**. | `train.console.log`, `preprocessing.pkl`, `dqn/dqn_final.zip`, `dqn/replay_buffer_200k.pkl`, `eval/0058_1m_reassert/summary.json` |
| `runs/ddqn_fresh_1_1_1_2_lean_stick/` | **complete; 1.3M selected** | 0060: 4×1-2 + 4×1-1 from 0058, 191 inputs, 200k 1-1 buffer, 300k additional steps. Selected 1.3M: **1-2 3.0333/31/31/20/20; 1-1 1.5667/12/12/8/8**. 1-2 state median 4.4333; final model's holdout was slower than the checkpoint. | `train.console.log`, `dqn/checkpoints/ddqn_1300000_steps.zip`, `eval/0060_1300000_1_2/`, `eval/0060_1300000_1_1/` |
| `runs/neat_rebalanced_1_3/` | 144 MB | **complete; mixed** | 0023: 10-generation continuation from rebalanced checkpoint 66. 238/384 final finishes (62%), 364 routes / 60 clean / 54 clusters. Archive improved, but live speed tail collapsed to 0 speed / 7 median / 224 slow / 153 exploration. Last checkpoint 72. | `history.csv`, `genome_stats.csv`, `archive/routes.json` |
| `runs/neat_rebalanced_1_3_final4/` | 54 MB | **diagnostic; separate trajectory** | 4-generation replay from checkpoint 72 to save checkpoint 76. 219 finishes, 1 speed / 9 median / 205 slow / 169 exploration; not identical to the original final generation. | `neat-checkpoint-76`, `history.csv`, `archive/routes.json` |
| `runs/neat_explore_1_3_long/rebalanced_fixed/` | 508 KB | **seed; innovation tracker fixed** | Rebalanced checkpoint 66 with tracker advanced to 5691; use instead of the original rebalanced checkpoint. | `rebalanced-66`, `rebalanced-.portfolio.json` |
| `runs/benchmarks/2026-10-29_macbook-air-m5_clean/` | 1 MB | **reference; battery sweep** | NEAT throughput sweep on Apple M5. Battery winner: **8×12 at 42.11 s/generation**; 12 processes oversubscribed. | `results.md`, `results.json` |
| `runs/benchmarks/2026-10-29_macbook-air-m5_ac/` | 1 MB | **reference; AC winner** | AC-power rerun: **8×12 at 28.81 s/generation**, 2.64 GB peak. | `results.md`, `results.json` |

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
