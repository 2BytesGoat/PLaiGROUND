# PLaiGROUND — docs index

PLaiGROUND is the external Python/RL lab for **Dragon Jump**. It connects to the
game's training mode over a local TCP socket, trains policies with NEAT, PPO,
supervised distillation, or DQN, and evaluates them on campaign levels. It is a
development/research repo, not part of the shipped game.

> **AI agents:** read [`../AGENTS.md`](../AGENTS.md) first; it contains the
> working rules and environment footguns.

## Start here

| If you want to… | Read |
|---|---|
| See the current branch, active experiment, and verified results | [`status.md`](status.md) |
| Watch the selected 1-1 → 1-4 policy in the game | `make watch` (see [`../README.md`](../README.md) § Watch the selected policy) |
| Find code, symbols, experiments, or runs | `make context q="<task>"` → [`code-map.md`](code-map.md) |
| Find a file, script, or run artifact | this page → [`../src/README.md`](../src/README.md) → [`reference/run-artifacts.md`](reference/run-artifacts.md) |
| Understand the game ↔ Python architecture | [`architecture.md`](architecture.md) |
| Know what is broken, stale, or confusing | [`known-issues.md`](known-issues.md) |
| Read the detailed DQN review from 2026-10-28 | [`audit/2026-10-28-dqn-pipeline-review.md`](audit/2026-10-28-dqn-pipeline-review.md) |
| Understand the start-pool clock bias and the PPO hypothesis | [`audit/2026-10-28-start-pool-clock-bias.md`](audit/2026-10-28-start-pool-clock-bias.md) |
| Understand the raw-state observation decision | [`audit/2026-10-28-raw-state-observation-decision.md`](audit/2026-10-28-raw-state-observation-decision.md) |
| Understand the clean-start state/world-state decisions | [`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md) |
| Understand the 450-route reservoir and path-filter decision | [`audit/2026-10-28-clean-route-reservoir.md`](audit/2026-10-28-clean-route-reservoir.md) |
| Review the applied route filter, start pools, and lean PPO launch | [`audit/2026-10-28-clean-route-filter-and-lean-ppo.md`](audit/2026-10-28-clean-route-filter-and-lean-ppo.md) |
| Review the repository restructure | [`audit/2026-10-28-repository-restructure.md`](audit/2026-10-28-repository-restructure.md) |
| Review PPO/DQN duplication, dead branches, and the composition plan | [`audit/2026-10-28-training-harness-composition.md`](audit/2026-10-28-training-harness-composition.md) |
| Understand the PPO timeout and start-pool root cause | [`audit/2026-10-28-ppo-timeout-root-cause.md`](audit/2026-10-28-ppo-timeout-root-cause.md) |
| Understand the lean 188-value state decision | [`audit/2026-10-28-lean-state-encoding.md`](audit/2026-10-28-lean-state-encoding.md) |
| Follow the full new-level pipeline (NEAT → expert → evaluate → repair) | [`runbooks/new-level.md`](runbooks/new-level.md) |
| See the selected expert and gate for each level | [`reference/level-ledger.md`](reference/level-ledger.md) |
| Follow the NEAT-only discovery branches | [`audit/2026-10-29-level-transfer-playbook.md`](audit/2026-10-29-level-transfer-playbook.md) |
| Understand the NEAT fitness / reward bias finding | [`audit/2026-10-29-reward-fitness-audit.md`](audit/2026-10-29-reward-fitness-audit.md) |
| Understand the 1-6 any-start expert and the non-reproducible 4.07 replay | [`audit/2026-10-29-1-6-ddqn-investigation.md`](audit/2026-10-29-1-6-ddqn-investigation.md) |
| Read the 1-8 (DOUBLE_JUMP) transfer and any-start expert sequence | [`audit/2026-10-30-1-8-expert.md`](audit/2026-10-30-1-8-expert.md) |
| Understand the expert-teacher / one-student architecture | [`audit/2026-10-29-expert-portfolio-and-student-decision.md`](audit/2026-10-29-expert-portfolio-and-student-decision.md) |
| Understand run reliability, checkpoint selection, and 1-6 fast modes | [`audit/2026-10-29-run-reliability.md`](audit/2026-10-29-run-reliability.md) |
| Set up the environment | [`../README.md`](../README.md) § Setup |
| Read a term you don't recognize | [`reference/glossary.md`](reference/glossary.md) |
| Check the deployed game model | [`reference/deployed-model-58.md`](reference/deployed-model-58.md) |

## Search and context

`make context q="<task>"` is the first read-only call for a task. It ranks
matching Python symbols, experiments, docs, and local runs, then suggests the
next files to read. `make index` regenerates [`code-map.md`](code-map.md) and
the local `.cache/repo-index.json`; `make index-check` fails if the committed
map is stale.

## Experiments

| Doc | Status | One-line |
|---|---|---|
| [`experiments/dqn-route-spawns.md`](experiments/dqn-route-spawns.md) | **active, legacy path** | Double DQN on 1-1 NEAT start pools; 794 sensor + 512×512 prefix baseline stopped at ~129.5k. Not the clean-start target; no model promotion. |
| [`experiments/dqn-clean-start.md`](experiments/dqn-clean-start.md) | **core implemented; validation pending** | Level-agnostic clean-start contract: raw state, source-checkpoint time, held powerups, dynamic `world_state`, NEAT validation. 1-1 smoke passed. |
| [`experiments/ppo-pilot.md`](experiments/ppo-pilot.md) | active/reference | Shared-policy PPO, multi-level, start pools, 754/794 observation variants. |
| [`experiments/ppo-clean-start.md`](experiments/ppo-clean-start.md) | historical plan; 225k baseline complete | 1-1 PPO on the lean 188-value packet with Python categorical one-hot, 1M timesteps, NEAT state starts with route-cluster holdout. |
| [`../experiments/0000_neat_clean_450/`](../experiments/0000_neat_clean_450/) | **complete manifest** | NEAT clean-route collection at game commit `adb9278`. |
| [`../experiments/0001_ppo_clean_start_1_1/`](../experiments/0001_ppo_clean_start_1_1/) | **complete manifest** | Exact config, game commit, and result record for the lean PPO run; 225k selected at 2.3167 s canonical. |
| [`../experiments/0002_ppo_clean_start_1_1_horizon_1m/`](../experiments/0002_ppo_clean_start_1_1_horizon_1m/) | **stopped; 400k selected** | 1M-horizon continuation; found 2.15 s at 400k, then deterministic timeout windows. Stopped at 972.6k for investigation; no retrain. |
| [`../experiments/0003_dqn_clean_start_1_1/`](../experiments/0003_dqn_clean_start_1_1/) | **planned manifest** | Config-driven raw-state DQN defaults for the 1-1 clean-start path. |
| [`../experiments/0004_ppo_clean_start_1_1_fresh/`](../experiments/0004_ppo_clean_start_1_1_fresh/) | **complete; 500k selected** | Fresh 500k PPO from scratch on the corrected pool; canonical 2.15 s, train 12/12, holdout 8/8. |
| [`../experiments/0038_neat_seeded_1_4_stick/`](../experiments/0038_neat_seeded_1_4_stick/) | **complete; clean 1-4 pool** | Hold-aware 1-4 discovery: legacy_stick 61-value observation, held-jump reassertion, seeded from 0037. Generation 1 reached 173/384 finishes and 10 clean routes; 5 train / 5 holdout routes. |
| [`../experiments/0039_ddqn_protected_1_1_1_2_1_3_1_4/`](../experiments/0039_ddqn_protected_1_1_1_2_1_3_1_4/) | **complete; 1-4 negative** | First 1-4 DDQN stage: 300k protected replay from selected 2.225M. Final 1-4 **0/1 canonical, 1/9 train, 1/3 holdout**; old levels mostly retained. No 1-4 checkpoint selected. |
| [`../experiments/0044_neat_full_finisher_1_4/`](../experiments/0044_neat_full_finisher_1_4/) | **stopped; diagnostic** | First finisher-preserving replay attempt. Live finish metadata reached rebalancing, but reproduction had already removed most evaluated finishers; only 26 of 173 remained. Superseded by 0045. |
| [`../experiments/0045_neat_full_finisher_1_4_retry/`](../experiments/0045_neat_full_finisher_1_4_retry/) | **stopped; diagnostic** | Retry with evaluated finishers merged back; reached 380/384 at generations 6/8 but still dropped the 50 finishers slower than 10 s (123 of 173 preserved). |
| [`../experiments/0046_neat_full_population_1_4/`](../experiments/0046_neat_full_population_1_4/) | **complete; 1-4 seed** | Final preservation pass: 380/384 at gen 30, peak 383/384, best 4.9833 s, 75 clean routes; independent eval 382/384. |
| [`../experiments/0047_neat_transfer_1_5_from_1_4/`](../experiments/0047_neat_transfer_1_5_from_1_4/) | **complete; diagnostic** | 1-4 → 1-5 transfer: zero-shot 6/384; seeded evolution finished 384/384, best 4.6833 s, 9 clean routes. Not pursued further. |
| [`../experiments/0048_ddqn_protected_1_4_full_pool/`](../experiments/0048_ddqn_protected_1_4_full_pool/) | **stopped diagnostic** | New 1-4 DDQN on the 0046 pool but lean 188 (no hold signals): at 2.525M, **0/1 canonical, 1/11 train, 0/2 holdout**. |
| [`../experiments/0053_ddqn_protected_1_4_lean_stick/`](../experiments/0053_ddqn_protected_1_4_lean_stick/) | **stopped; 100k probe** | lean_stick 191 (188 + on_ceiling/jump_held/stick_active), 188 weights expanded to 957 inputs. At 100k: 1-4 **2/11 train**, 0/1 canonical; 1-3 regressed. |
| [`../experiments/0058_ddqn_fresh_1_1_lean_stick/`](../experiments/0058_ddqn_fresh_1_1_lean_stick/) | **complete; 1M selected** | Fresh 191 backbone, 1-1 only, 1M steps. Corrected eval: **1.5667 s canonical, 12/12 train, 8/8 holdout**. The first eval omitted held-jump semantics. |
| [`../experiments/0059_ddqn_fresh_1_1_lean_stick_continue/`](../experiments/0059_ddqn_fresh_1_1_lean_stick_continue/) | **not needed** | The 1.85M continuation was unnecessary once 0058 passed the gate at 1M. |
| [`../experiments/0060_ddqn_fresh_1_1_1_2_lean_stick/`](../experiments/0060_ddqn_fresh_1_1_1_2_lean_stick/) | **complete; 1.3M selected** | 1-2 transfer from 0058: 4×1-2 + 4×1-1, 191 inputs, 200k 1-1 buffer. Selected 1.3M: **1-2 3.0333 canonical, 31/31 train, 20/20 holdout; 1-1 1.5667, 12/12, 8/8**. Remaining weakness: 1-2 state-start speed. |
| [`../experiments/0073_ddqn_expert_replay_1_4_lean_stick/`](../experiments/0073_ddqn_expert_replay_1_4_lean_stick/) | **complete; 1.555M selected** | One-step DDQN with 21 successful 1-4 NEAT routes (2,231 transitions) in an expert buffer plus an action-margin loss. 1.555M: **1-4 1/1 canonical @ 4.4667 s, 11/11 train, 2/2 holdout**; old levels full coverage (1-2 3.1500, 1-3 3.2667). 1-4 fixed; 1-2/1-3 canonical slightly slower than 1.53M. |
| [`../experiments/0080_neat_extend_1_5_from_0047/`](../experiments/0080_neat_extend_1_5_from_0047/) | **complete; 1-5 archive** | Extended the 0047 1-5 population to generation 50: 284 routes, 16 clean, best 4.6833 s. Raw reward ranked a 7.1167 s finisher above the 4.6833 s route. |
| [`../experiments/0081_neat_speed_1_5_from_0080/`](../experiments/0081_neat_speed_1_5_from_0080/) | **stopped; best 4.4000 s** | Time-penalized 2-frame search from generation 50 to 95: 569 routes / 130 clean; speed bin 0 → ~88–100%; plateaued above the 3.41 s gate. |
| [`../experiments/0082_ddqn_expert_replay_1_5_lean_stick/`](../experiments/0082_ddqn_expert_replay_1_5_lean_stick/) | **stopped; 2-frame negative** | 2-frame 4.40 s expert (4,191 transitions) with a 2-frame anchor. Deterministic 1.58M: 1-5 0/1, 1-4 timeout, 1-1 2.9333 s. Control-rate change rejected. |
| [`../experiments/0085_ddqn_expert_replay_1_5_4frame/`](../experiments/0085_ddqn_expert_replay_1_5_4frame/) | **complete; 1-5 3.1667 s** | 4-frame DDQN on all 16 clean 1-5 routes (1,510 expert transitions, 4.68 s cluster included). Final **1-5 3.1667 s** twice; 1-1/1-3 retained, 1-4/1-2 not. Selected 1-5 model. |
| [`../experiments/0088_ddqn_retention_1_5_4frame/`](../experiments/0088_ddqn_retention_1_5_4frame/) | **complete; single-model negative** | 25k retention fine-tune from 1.655M with 2 x 1-4/1-2 and a 1.555M all-level anchor. Final: 1-5 3.5000 s, 1-4 timeout, 1-3 3.2000, 1-2 3.0167. Two-model chain selected. |
| [`../experiments/0090_neat_transfer_1_6_from_1_5/`](../experiments/0090_neat_transfer_1_6_from_1_5/) | **complete; 1-6 clean seed** | Seeded from the 4-frame 1-5 checkpoint-50. Generation 51 produced 31 routes / 25 clean, best **4.5500 s**; stopped at the clean target. Filtered 24 clean / 9 clusters / 22 train / 2 holdout. |
| [`../experiments/0091_ddqn_expert_replay_1_6_lean_stick/`](../experiments/0091_ddqn_expert_replay_1_6_lean_stick/) | **complete; 1-6 negative** | 20/22 expert routes replayed (1,696 transitions). Best checkpoint **1.705M: 1-6 4.5667 s canonical** (1.68M 4.6833); final 7.25 s. Full 32-state pool: **11/32**, 20 deaths. Lost 1-5/1-4; not a chain member. |
| [`../experiments/0093_neat_speed_1_6_from_0090/`](../experiments/0093_neat_speed_1_6_from_0090/) | **complete; 1-6 4.5000 s** | 30-generation time-penalized search: best **4.5000 s** at gen 68; archive 376 routes / 263 clean; still +0.22 over the 4.28 gate. |
| [`../experiments/0094_ddqn_anchor_1_6_1680/`](../experiments/0094_ddqn_anchor_1_6_1680/) | **complete; 1-6 anchor** | 200k rare transitions from 1.68M; 3,224 episodes, 1,697 finishes / 1,412 deaths / 115 timeouts; median 4.6833; real stochastic finishes at 3.0667/3.2/3.3333 (155 under 4.28); 2.4333 is a death. |
| [`../experiments/0095_ddqn_1_6_retrain_1680/`](../experiments/0095_ddqn_1_6_retrain_1680/) | **complete; 1-6-only negative** | 8 x 1-6 from 1.68M, 1-6 anchor, 4.07 s replay. Final **6.3833 s**; intermediate 8.50/timeouts. The 4.07 s replay was later found not reproducible in isolation. Best canonical model is 1.705M 4.5667. |
| [`../experiments/0096_eval_ddqn_1_6_retrain_1680/`](../experiments/0096_eval_ddqn_1_6_retrain_1680/) | **complete; 1-6 eval** | Deterministic 1-6 eval for 0095 checkpoints: final 6.3833; intermediate timeout/8.50/death. |
| [`../experiments/0097_ddqn_1_6_feasible_starts/`](../experiments/0097_ddqn_1_6_feasible_starts/) | **complete; any-start base** | 8 x 1-6 from 1.705M, no old-level mixing, 32 NEAT state starts, 85 stable expert routes. First 32/32 at **1.7675M** (canonical 4.65); 1.805M canonical 4.55 but 31/32. |
| [`../experiments/0098_eval_ddqn_1_6_feasible_starts/`](../experiments/0098_eval_ddqn_1_6_feasible_starts/) | **complete; 32-state eval** | Canonical + all 32 NEAT feasible starts; baseline 1.705M 11/32; 0097 checkpoint sweep. |
| [`../experiments/0099_ddqn_1_6_start_repair/`](../experiments/0099_ddqn_1_6_start_repair/) | **complete; selected 1-6 expert** | Targeted suffix repair of `[21,16]/-1`. Selected **1.815M: canonical 4.6000 s; 32/32 states, 0 deaths/timeouts, median 6.0417 s**; spawn 32/32; repeat-stable. Speed gate 4.28 still open. |
| [`../experiments/0100_eval_ddqn_1_6_start_repair/`](../experiments/0100_eval_ddqn_1_6_start_repair/) | **complete; selected eval** | Deterministic repeat of 0099 1.815M: 4.6000 s canonical, 32/32 states, 0 deaths/timeouts. |
| [`../experiments/0101_neat_transfer_1_7_from_1_6/`](../experiments/0101_neat_transfer_1_7_from_1_6/) | **stopped; 1-7 transfer** | Final 4-frame 1-6 population transferred to 1-7; 240 routes / 10 clusters / **0 strict clean** at gen 92, best 7.9667 s. Level analysis exported. |
| [`../experiments/0102_neat_clean_1_7_from_0101/`](../experiments/0102_neat_clean_1_7_from_0101/) | **complete; 1-7 cleanup** | High-mutation population seeded from six 1-7 route-cluster representatives; 902 routes / 37 clusters / **0 strict clean**, best 7.0333 s; 20 grounded feasible starts; stable expert 2,495 transitions. |
| [`../experiments/0103_ddqn_1_7_feasible_starts/`](../experiments/0103_ddqn_1_7_feasible_starts/) | **complete; selected 1-7 expert** | 8 x 1-7 from 1.815M. Selected **1.877432M: canonical 5.2333 s; near-clean 10/10, 0 deaths/timeouts, median 7.9583; all-success 18/20 (2 diagnostic timeouts)**. Provisional 5.2728 gate passes by 0.0395 s. |
| [`../experiments/0104_eval_ddqn_1_7_feasible_starts/`](../experiments/0104_eval_ddqn_1_7_feasible_starts/) | **complete; selected 1-7 eval** | Deterministic repeat of 1.877432M: canonical 5.2333, near-clean 10/10, all-success 18/20, spawn 20/20. |
| [`../experiments/0105_neat_transfer_1_8_from_1_7/`](../experiments/0105_neat_transfer_1_8_from_1_7/) | **complete; 1-8 clean seed** | Final 1-7 population transferred to 1-8 (DOUBLE_JUMP); 26 strict-clean routes / 11 clusters, best **4.4167 s**, 161 finishers, stopped at the clean target in 16 generations. |
| [`../experiments/0106_ddqn_1_8_feasible_starts/`](../experiments/0106_ddqn_1_8_feasible_starts/) | **complete; 1-8 first expert** | 16/16 clean train routes replayed (1,800 stable transitions). First full-coverage candidate **1.939912M**: canonical 4.5167, clean 39/40, all-success 106/107. |
| [`../experiments/0107_eval_ddqn_1_8_feasible_starts/`](../experiments/0107_eval_ddqn_1_8_feasible_starts/) | **complete; 1-8 sweep** | All eight 0106 checkpoints on 40 strict-clean states. 1.939912M selected: 39/40, 0 deaths, 1 timeout; 1.927416M has 40/40 but canonical 10.0667. |
| [`../experiments/0108_eval_ddqn_1_8_all_success/`](../experiments/0108_eval_ddqn_1_8_all_success/) | **complete; all-success diagnostic** | 1.939912M and 1.927416M cover 106/107 all-success starts; 1.952408M covers 102/107 with 5 timeouts. |
| [`../experiments/0109_ddqn_1_8_canonical_repair/`](../experiments/0109_ddqn_1_8_canonical_repair/) | **complete; 1-8 coverage repair** | Canonical-heavy 25k repair. 1.949912M reaches **40/40** clean states, canonical 4.5167; no deaths/timeouts. |
| [`../experiments/0112_ddqn_1_8_canonical_route/`](../experiments/0112_ddqn_1_8_canonical_route/) | **complete; 1-8 canonical fix** | Canonical-only suffix fine-tune against the 4.4667 s route. 1.952408M reaches canonical **4.4667 s** but drops to 39/40 clean states. |
| [`../experiments/0114_ddqn_1_8_start_repair/`](../experiments/0114_ddqn_1_8_start_repair/) | **complete; 1-8 start repair** | Targeted `[4,21]/-1` suffix buffer. 1.959896M becomes the first **40/40** clean model; 1.962392M keeps canonical 4.4667 but is 39/40. |
| [`../experiments/0116_ddqn_1_8_last_start_repair/`](../experiments/0116_ddqn_1_8_last_start_repair/) | **complete; negative** | Targeted `[3,21]/-1` suffix over-corrected: 1.964888M canonical 4.4667 but only 33/40 clean. Not selected. |
| [`../experiments/0118_ddqn_1_8_canonical_nudge/`](../experiments/0118_ddqn_1_8_canonical_nudge/) | **complete; selected 1-8 expert** | 500-step low-LR canonical nudge from 1.959896M. Selected **1.960392M: canonical 4.4667 s; clean 40/40, 0 deaths/timeouts, median 9.8833; all-success 106/107**. |
| [`../experiments/0120_eval_ddqn_1_8_final_all_success/`](../experiments/0120_eval_ddqn_1_8_final_all_success/) | **complete; final 1-8 diagnostic** | Selected 1.960392M: canonical 4.4667, clean 40/40, all-success 106/107 (one high-revisit timeout), spawn 107/107. |
| [`experiments/compact-policy-deployment.md`](experiments/compact-policy-deployment.md) | **proposed** | Compact flat/embedding student options and the distillation plan for shipping a JSON policy. |
| [`experiments/neat-1-1.md`](experiments/neat-1-1.md) | historical baseline | Early 32-agent NEAT 1-1 run (2.35 s). The 8×24 archive improved this to 2.15 s. |
| [`experiments/neat-lineage.md`](experiments/neat-lineage.md) | available/historical | One population carried through the campaign in order. |
| [`experiments/ppo-distillation.md`](experiments/ppo-distillation.md) | deployed/reference | Per-level PPO teachers → one level-agnostic 58-value student; the model installed in the game. |

## Reference

| Doc | What it covers |
|---|---|
| [`reference/observation-contracts.md`](reference/observation-contracts.md) | 58 / 61 (legacy_stick) / 754 / 794 / 188 / raw observation layouts, action cadence, reset forms. |
| [`reference/reward-contract.md`](reference/reward-contract.md) | Shipped five-term training reward, `gold_ratio`, raw telemetry, force-start, terminal death. |
| [`reference/glossary.md`](reference/glossary.md) | Teacher, student, specialist, lineage, prefix replay, start-jump, canonical, clean route, etc. |
| [`reference/run-artifacts.md`](reference/run-artifacts.md) | What each `runs/` directory contains and how to read its summaries. |
| [`reference/deployed-model-58.md`](reference/deployed-model-58.md) | The frozen 58→256→256→2 JSON student shipped in the game. |
| [`reference/ddqn-1-5-chain.md`](reference/ddqn-1-5-chain.md) | Selected two-model 1-5 DDQN chain: 1-5 3.1667 s, old levels 1.5667 / 3.1500 / 3.2667 / 4.4667; rejected single-model attempt. |
| [`audit/2026-10-29-1-6-ddqn-investigation.md`](audit/2026-10-29-1-6-ddqn-investigation.md) | 1-6 investigation: 0099 1.815M any-start 32/32 at 4.60 s; 1.705M 4.5667 canonical-only; non-reproducible 4.07 replay; 3.07/3.2 stochastic modes; false 2.43/2.7 death times. |
| [`audit/2026-10-29-run-reliability.md`](audit/2026-10-29-run-reliability.md) | Run reliability audit: 1.705M checkpoint discovery, death/timeout metric fix, socket/process cleanup, strict spawns, 1-6 state pool, checkpoint-selection checklist. |
| [`audit/2026-10-29-expert-portfolio-and-student-decision.md`](audit/2026-10-29-expert-portfolio-and-student-decision.md) | **Decision:** NEAT-discovered feasible starts; one expert per level; experts are teachers; one shared student A with no level ID or per-level heads; single-level expert runs separate from the multi-level student phase. |

## Repo map

| Path | Purpose |
|---|---|
| `src/plaiground/` | Importable training/data/evaluation package. See [`../src/README.md`](../src/README.md). |
| `scripts/` | Thin baseline, genetic, supervised, reinforcement, and evaluation entry points. |
| `tools/` | Route filtering, start-pool building, observation checks, reporting, and repo search utilities. |
| `experiments/` | Committed experiment manifests and result notes. |
| `experiments/0000_neat_clean_450/` | NEAT clean-route collection manifest. |
| `experiments/0001_ppo_clean_start_1_1/` | Lean PPO clean-start manifest. |
| `experiments/0002_ppo_clean_start_1_1_horizon_1m/` | 1M-horizon continuation manifest. |
| `src/plaiground/envs/` | `GodotEnv`, Gymnasium/SB3 adapters, shared env factory, vector env, legacy wrapper. |
| `src/plaiground/observations/spec.py` | Python mirror of the game's `AIObservationSpec`. |
| `src/plaiground/observations/lean_io.py` | Lean preprocessing load/create. |
| `src/plaiground/evaluation/rollout.py` | Shared canonical/start-set evaluator. |
| `src/plaiground/experiments/runner.py` | Config -> run paths and common CLI. |
| `tools/level.py` | Route-tier classification, filtered pools, eval union, and per-level status. |
| `tools/evaluate.py` | Shared PPO/DQN evaluation entry point. |
| `tests/` | Protocol, preprocessing, model, runner, evaluation, route, pool, and experiment tests. |
| `runs/` | Generated artifacts (gitignored): checkpoints, eval summaries, logs, per-run userdata. |
| `archive/2026-10-27-cleanup/` | Historical scripts/runs/docs. Start at its `README.md`. |
| `docs/code-map.md` | Generated Python/experiment/doc map; do not hand-edit. |
| `docs/runbooks/` | Living procedures, including the new-level pipeline. |
| `docs/reference/level-ledger.md` | Selected expert, canonical time, and gate per level. |
| `docs/` | Architecture, contracts, audits, and historical experiment deep-dives. |
| `notebooks/` | Tutorial notebooks; not the active training path. |
| `data/cheatsheet/` | Small tutorial dataset. |
| `environments/` | Game binaries / macOS source shim; contents are gitignored. |

## Other

- [`blog-growing-a-hand.md`](blog-growing-a-hand.md) — long-form blog draft about
  the project; not a technical reference.
- [`DragonJump/`](DragonJump/) — game screenshots used by the blog/README.
- [`../archive/2026-10-27-cleanup/README.md`](../archive/2026-10-27-cleanup/README.md)
  — historical scripts/runs/docs index.

## Conventions

- **Experiment manifests** go in `experiments/<id>_<name>/` with `README.md`,
  `config.toml`, and `results.md`. Deep technical dives may still live in
  `docs/experiments/`.
- **Status fields** at the top of active docs: `Status: active | historical | deployed | negative`.
- **Dates** are ISO (`2026-10-28`) and refer to local dev time.
- **Run artifacts are not committed.** Link them by path and record the result in
  docs; `runs/` is gitignored.
- **The code map is generated.** `docs/code-map.md` comes from `make index`;
  update the source docs/experiments instead of hand-editing the map.
- **Cross-repo claims** should name the game branch/commit when they depend on
  game code. The game vault is the source of truth for game behavior.
- **Investigation notes** go in `docs/audit/`; living issues go in
  `docs/known-issues.md`; current state goes in `docs/status.md`.
