# Run artifacts

`runs/` is generated and gitignored. This page is the committed index of what
each active run directory is for and where to look first. Add a row when a run
becomes important; do not commit the run itself.

> For a live summary of one run, use `make run RUN=<name>`; this page is the
> curated narrative inventory, not the live status.

Sizes are approximate from 2026-10-30.

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
| `runs/ddqn_expert_replay_1_4_lean_stick/` | 186 MB | **complete; 1.555M selected** | 0073/0079: 1-4 expert replay, 1.555M chain. 1-1..1-4 canonical **1.5667 / 3.1500 / 3.2667 / 4.4667**, all finish, no deaths. | `dqn/checkpoints/ddqn_1555000_steps.zip`, `eval/` |
| `runs/chain_check_1_1_to_1_4_1555k/` | 108 KB | **complete; chain check** | 0079: windowed 1-1..1-4 replay of the 1.555M checkpoint; times match 0073 exactly. | `eval/watch_1555k_1x/summary.json` |
| `runs/neat_speed_1_5_from_0080/` | 26 MB | **stopped; best 4.4000 s** | 0081: time-penalized 2-frame search, stopped at generation 95. 569 routes / 130 clean; 2-frame subset 32 train / 4 holdout; speed plateaued above the 3.41 s gate. | `clean_2frame_routes.json`, `filtered_2frame/`, `pool_2frame/` |
| `runs/ddqn_expert_replay_1_5_lean_stick/` | 138 MB | **stopped; 2-frame negative** | 0082: 2-frame 4.40 s expert, 4,191 transitions, 2-frame anchor. 1.58M deterministic: **1-5 0/1, 1-4 timeout, 1-1 2.9333 s**. | `expert_buffer_1_5_2frame.pkl`, `eval/eval_1580000/summary.json` |
| `runs/ddqn_anchor_1555_2frame/` | 1.4 GB | **complete; degraded 2-frame anchor** | 0083: 200k 2-frame transitions from the 4-frame 1.555M model. 2,229 episodes, median 3.25 s, 409 timeouts; do not mix with 4-frame training. | `rare/replay_buffer_200k.pkl`, `replay_buffer_200k.json` |
| `runs/ddqn_expert_replay_1_5_4frame/` | 145 MB | **complete; 1-5 3.1667 selected** | 0085: 16 clean 4-frame routes / 1,510 expert transitions; final **1-5 3.1667 s** (twice), 1-1/1-3 retained, 1-4/1-2 not. | `dqn/dqn_final.zip`, `eval/eval_final/summary.json` |
| `runs/ddqn_anchor_1555_all_levels_4frame/` | 1.4 GB | **complete; retention anchor** | 0087: 200k all-level 4-frame anchor from 1.555M. 5,105 episodes, median 3.00 s, best 1.5667 s, 6 timeouts. | `rare/replay_buffer_200k.pkl` |
| `runs/ddqn_retention_1_5_4frame/` | 80 MB | **complete; single-model negative** | 0088: 25k retention fine-tune. Final **1-5 3.5000 s**, 1-4 timeout, 1-3 3.2000, 1-2 3.0167, 1-1 1.5667. Not selected. | `eval/eval_final/summary.json` |
| `runs/neat_transfer_1_6_from_1_5/` | 3.8 GB | **complete; 1-6 seed + speed archive** | 0090/0093: 4-frame 1-6 search. Initial 4.5500 s / 25 clean; speed run best **4.5000 s**, 376 routes / 263 clean; 0093 finished in 602 s. | `history.csv`, `archive/`, `best_clean.json` |
| `runs/neat_transfer_1_6_from_1_5/pool_state/` | 128 KB | **historical; small state pool** | The first 1-6 analysis pool: 11 train / 6 holdout state starts (5 shared keys), canonical weight 0.70. The full 263-route expansion is in `runs/neat_speed_1_6_from_0090/pool_state_*`. | `train_start_pool.json`, `holdout_start_pool.json`, `split.json` |
| `runs/neat_speed_1_6_from_0090/` | 17 MB | **complete; 1-6 speed** | 0093: 30-generation time-penalized continuation. Best 4.5000 s at gen 68; target 3.98 not reached. `filtered/` holds 133 clean / 33 clusters / 104 train / 29 holdout; `filtered_all/` holds all 263 clean / 227 train / 36 holdout. | `history.csv`, `filtered/summary.json`, `filtered_all/summary.json` |
| `runs/neat_speed_1_6_from_0090/pool_state_50/` | 160 KB | **complete; any-start pool** | All 263 clean routes expanded: 31 train states + canonical, 16 holdout states (15 shared), 0.5/0.5 weights. `eval_all_start_pool.json` is the 32-state coverage gate. | `train_start_pool.json`, `holdout_start_pool.json`, `eval_all_start_pool.json`, `split.json` |
| `runs/neat_speed_1_6_from_0090/pool_state_70/` | 160 KB | **complete; conservative pool** | Same 31/32 states with 0.7 canonical / 0.3 state weights. Used by 0099. | `train_start_pool.json`, `holdout_start_pool.json`, `eval_all_start_pool.json` |
| `runs/ddqn_expert_replay_1_6_lean_stick/` | 148 MB | **complete; 1-6 4.57 canonical** | 0091: 1-6 DDQN from the 1-5 model. Best checkpoint **1.705M: 4.5667 s**; 1.68M 4.6833; 1.73M 8.60; 1.755M 8.3167; final 7.25; old levels fail. Full 32-state pool: **11/32**, 20 deaths, 1 timeout, median 6.1167 (see 0097 baseline). | `dqn/checkpoints/ddqn_1705000_steps.zip`, `eval/selected_default_check/summary.json` |
| `runs/ddqn_anchor_1_6_1680/` | 1.4 GB | **complete; 1-6-only anchor** | 0094: 200k rare transitions from the 1.68M model. 3,224 episodes, 1,697 finishes / 1,412 deaths / 115 timeouts; median 4.6833; stochastic best finish 3.0667; reported 2.4333 is a **death**. | `rare/replay_buffer_200k.pkl`, `rare/collect_episodes_200k.csv` |
| `runs/ddqn_1_6_retrain/` | 489 MB | **complete; 1-6-only negative** | 0095: 8 x 1-6 + 1-6 anchor + 4.07 s replay. Final **6.3833 s**; intermediate 8.50/timeouts. The 4.07 replay is not reproducible in isolation; see the 0097 audit. | `eval/eval_final/summary.json`, `expert_1_6/expert_routes.npz` |
| `runs/ddqn_1_6_feasible_starts/` | 150 MB | **complete; any-start base** | 0097: 100k from 1.705M, 8 x 1-6, 32 state starts, 85 stable expert routes. First 32/32 at **1.7675M** (canonical 4.65); 1.805M 4.55 but 31/32. | `dqn/checkpoints/ddqn_1767480_steps.zip`, `eval/check_*/summary.json`, `expert_1_6/stable_expert_buffer.pkl` |
| `runs/ddqn_1_6_start_repair/` | 80 MB | **complete; selected 1-6 expert** | 0099: targeted suffix repair from 1.805M; selected **1.815M**. Canonical **4.6000 s**; **32/32** states, 0 deaths/timeouts, median 6.0417, spawn 32/32; repeat-stable. Speed gate 4.28 open. | `dqn/checkpoints/ddqn_1814952_steps.zip`, `eval/selected_repeat/summary.json`, `expert_1_6/repair_suffix_buffer.pkl` |
| `runs/neat_transfer_1_7_from_1_6/` | 1.5 GB | **stopped; 1-7 transfer** | 0101: 4-frame 1-6 population transferred to 1-7. Generations 81-92: 240 routes / 10 clusters / **0 strict clean**, best 7.9667 s. | `neat-checkpoint-92`, `archive/routes.json`, `history.csv` |
| `runs/neat_clean_1_7_from_0101_attempt2/` | 4.3 GB | **complete; 1-7 route archive** | 0102: high-mutation cleanup from six 1-7 cluster representatives. 902 routes / 37 clusters / **0 strict clean**, best **7.0333 s** (gen 9, revisit 0.107). 20-state all-success pool and 10-state near-clean pool. | `neat-checkpoint-30`, `archive/routes.json`, `filtered_all_success_g30/`, `pool_all_success_g30/`, `pool_near_clean_g30/` |
| `runs/ddqn_1_7_feasible_starts/` | 347 MB | **complete; selected 1-7 expert** | 0103: 100k from 1.815M, 8 x 1-7, 20-start all-success pool, 2,495 stable expert transitions. Selected **1.877432M**: canonical **5.2333 s**; near-clean **10/10**, 0 deaths/timeouts, median 7.9583; all-success 18/20 (2 diagnostic timeouts). | `dqn/checkpoints/ddqn_1877432_steps.zip`, `eval/selected_repeat/summary.json`, `expert_1_7/stable_expert_buffer.pkl` |
| `runs/neat_transfer_1_8_from_1_7/` | 144 MB | **complete; 1-8 clean archive** | 0105: final 1-7 population transferred to 1-8 (DOUBLE_JUMP). 161 finishers / **26 strict-clean routes / 11 clusters**, best **4.4167 s**, stopped in 16 generations. `pool_clean_g46/` has 37 train / 14 holdout states (40 unique eval); `pool_all_success_g46/` has 102 train / 63 holdout (107 unique eval). | `neat-checkpoint-46`, `archive/routes.json`, `pool_clean_g46/`, `pool_all_success_g46/` |
| `runs/ddqn_1_8_feasible_starts/` | 279 MB | **complete; 1-8 first expert** | 0106: 100k from 1.877432M, 8 x 1-8, clean pool. 16/16 expert routes replayed; stable buffer 1,800 transitions. Best early checkpoint **1.939912M**: canonical 4.5167, clean 39/40, all-success 106/107. | `dqn/checkpoints/ddqn_1939912_steps.zip`, `eval/`, `expert_1_8/stable_expert_buffer.pkl` |
| `runs/ddqn_1_8_canonical_repair/` | 177 MB | **complete; coverage repair** | 0109: canonical-heavy 25k from 1.939912M. 1.949912M reaches **40/40** clean states, canonical 4.5167, 0 deaths/timeouts. | `dqn/checkpoints/ddqn_1949912_steps.zip`, `eval/ddqn_1949912_steps/summary.json` |
| `runs/ddqn_1_8_canonical_route/` | 40 MB | **complete; canonical fix** | 0112: canonical-only 5k from 1.949912M. 1.952408M canonical **4.4667 s** but clean 39/40. | `dqn/checkpoints/ddqn_1952408_steps.zip`, `eval/ddqn_1952408_steps/summary.json` |
| `runs/ddqn_1_8_start_repair/` | 77 MB | **complete; start repair** | 0114: targeted `[4,21]/-1` suffix from 1.952408M. 1.959896M first **40/40** model at canonical 4.5167; 1.962392M canonical 4.4667 but 39/40. | `dqn/checkpoints/ddqn_1959896_steps.zip`, `dqn/checkpoints/ddqn_1962392_steps.zip`, `eval/` |
| `runs/ddqn_1_8_last_start_repair/` | 45 MB | **complete; negative** | 0116: targeted `[3,21]/-1` suffix from 1.962392M over-corrected; best 33/40 clean. Not selected. | `eval/ddqn_1964888_steps/summary.json` |
| `runs/ddqn_1_8_canonical_nudge/` | 37 MB | **complete; selected 1-8 expert** | 0118: 500-step low-LR canonical nudge from 1.959896M. Selected **1.960392M**: canonical **4.4667 s**, clean **40/40**, 0 deaths/timeouts, median 9.8833; all-success 106/107 (one high-revisit timeout). | `dqn/checkpoints/ddqn_1960392_steps.zip`, `eval/selected_repeat/summary.json`, `eval/selected_all_success/summary.json` |
| `runs/neat_transfer_1_9_from_1_8/` | 31 MB | **stopped; 1-9 single-route archive** | 0121: final 1-8 population transferred to 1-9 (CHAIN_JUMPS), stopped at generation 89. **11 strict-clean routes / 6 clusters**, best **2.9500 s**. `level.py data` gives 9 train / 2 holdout routes, 6/8 state starts, 11 eval states. The fastest route (`926883...`) is the selected single-route source. | `neat-checkpoint-89`, `archive/routes.json`, `strict_clean_routes_g89.json`, `pool_clean_g89/`, `single_route/` |
| `runs/ddqn_1_9_single_route/` | 12 MB | **complete; selected 1-9 single-route expert** | 0122: 100k canonical-only DDQN from 1.960392M, one-route buffer (43 transitions). All 8 checkpoints finish canonical **2.9500 s**, 0 deaths/timeouts. Selected **2.060360M**. Retention 1-1..1-9: 1-1/1-3/1-9 finish; only 1-1/1-9 pass gate. | `dqn/checkpoints/ddqn_2060360_steps.zip`, `eval/final_1_1_to_1_9/summary.json`, `expert_1_9/stable_expert_buffer.pkl` |
| `runs/neat_transfer_1_10_from_1_9/` | 1.3 GB | **stopped; 1-10 route archive** | 0125: final 1-9 population transferred to 1-10 (DOUBLE_SHORT). First segment stopped at generation 91 (30 clean, best 4.9500); resumed to generation 131. **387 finishers / 311 strict clean / 22 clusters**; filtered 109 routes, pool 24 train / 6 holdout, 25 eval states. Best clean **4.7000 s**; gate 3.3336 open. | `neat-checkpoint-131`, `archive/routes.json`, `pool_clean_g131/`, `level_data_summary.json` |
| `runs/ddqn_1_10_feasible_starts/` | 767 MB | **complete; selected 1-10 coverage expert** | 0126/0127: 100k from the 1-9 2.060360M model, 6,834 stable expert transitions / 78 routes. Selected **2.122840M**: canonical **5.3667 s**; eval **22/25**, 3 deaths, 0 timeouts, spawn 25/25. Three hard starts `[8,25]`, `[9,25]`, `[26,20]`. Speed gate open. | `dqn/checkpoints/ddqn_2122840_steps.zip`, `eval/selected_repeat/summary.json`, `expert_1_10/stable_expert_buffer.pkl` |
| `runs/ddqn_1_10_feasible_starts_continue/` | 1.3 GB | **complete; no coverage gain** | 0128: 200k low-LR continuation from 2.122840M. Best canonical **5.3000 s** but coverage **19-21/25**; late training decays coverage. Not selected. | `dqn/checkpoints/ddqn_2222808_steps.zip`, `train.console.log` |
| `runs/ddqn_1_10_start_repair/` | 87 MB | **complete; speed alternative** | 0129: 10k targeted suffix repair from 2.122840M for `[8,25]`, `[9,25]`, `[26,20]`. Reached NEAT-best **4.7000 s** canonical but coverage **17-20/25**; the three target starts still fail. Not selected. | `dqn/checkpoints/ddqn_2127832_steps.zip`, `expert_1_10/repair_suffix_buffer.pkl` |

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
| `strict_clean_routes_g<G>.json`, `near_clean_routes_g<G>.json`, `all_success_routes_g<G>.json` | Re-computed route tiers from a NEAT archive; produced by `tools/level.py data`. |
| `route_quality_g<G>.json` | Tier counts, best times, and the thresholds used. |
| `filtered/summary.json` | Route-filter counts: input/unique routes, clusters, kept train/holdout routes. |
| `pool/train_start_pool.json`, `pool/holdout_start_pool.json` | Clean-start state pools; the holdout file is evaluation-only. |
| `pool/eval_<tag>_start_pool.json` | Deduplicated train+holdout union for the accepted evaluation gate. |
| `pool/split.json` | Start-pool counts, weights, checkpoint rejection stats, train/holdout key overlap, and `unique_eval_states`. |
| `level_data_summary.json` | One level's tier counts, selected quality, filtered/pool paths, eval union, fastest cluster, and warnings. |
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
3. After NEAT, run `tools/level.py data` to create the route tiers and pools.
4. Record the command, branch/commit, game dirty state, and key result in
   `docs/status.md`, `docs/reference/level-ledger.md`, or a dated audit.
5. Add a row to this page when the run becomes part of the reference set.
