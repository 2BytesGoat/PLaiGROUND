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
| Understand the lean 188-value state decision | [`audit/2026-10-28-lean-state-encoding.md`](audit/2026-10-28-lean-state-encoding.md) |
| Set up the environment | [`../README.md`](../README.md) § Setup |
| Read a term you don't recognize | [`reference/glossary.md`](reference/glossary.md) |
| Check the deployed game model | [`reference/deployed-model-58.md`](reference/deployed-model-58.md) |

## Experiments

| Doc | Status | One-line |
|---|---|---|
| [`experiments/dqn-route-spawns.md`](experiments/dqn-route-spawns.md) | **active, legacy path** | Double DQN on 1-1 NEAT start pools; 794 sensor + 512×512 prefix baseline stopped at ~129.5k. Not the clean-start target; no model promotion. |
| [`experiments/dqn-clean-start.md`](experiments/dqn-clean-start.md) | **core implemented; validation pending** | Level-agnostic clean-start contract: raw state, source-checkpoint time, held powerups, dynamic `world_state`, NEAT validation. 1-1 smoke passed. |
| [`experiments/ppo-pilot.md`](experiments/ppo-pilot.md) | active/reference | Shared-policy PPO, multi-level, start pools, 754/794 observation variants. |
| [`experiments/ppo-clean-start.md`](experiments/ppo-clean-start.md) | **active; 1M run started** | 1-1 PPO on the lean 188-value packet with Python categorical one-hot, 1M timesteps, NEAT state starts with route-cluster holdout. |
| [`../experiments/0000_neat_clean_450/`](../experiments/0000_neat_clean_450/) | **complete manifest** | NEAT clean-route collection at game commit `adb9278`. |
| [`../experiments/0001_ppo_clean_start_1_1/`](../experiments/0001_ppo_clean_start_1_1/) | **complete manifest** | Exact config, game commit, and result record for the lean PPO run; 225k selected at 2.3167 s canonical. |
| [`experiments/compact-policy-deployment.md`](experiments/compact-policy-deployment.md) | **proposed** | Compact flat/embedding student options and the distillation plan for shipping a JSON policy. |
| [`experiments/neat-1-1.md`](experiments/neat-1-1.md) | historical baseline | Early 32-agent NEAT 1-1 run (2.35 s). The 8×24 archive improved this to 2.15 s. |
| [`experiments/neat-lineage.md`](experiments/neat-lineage.md) | available/historical | One population carried through the campaign in order. |
| [`experiments/ppo-distillation.md`](experiments/ppo-distillation.md) | deployed/reference | Per-level PPO teachers → one level-agnostic 58-value student; the model installed in the game. |

## Reference

| Doc | What it covers |
|---|---|
| [`reference/observation-contracts.md`](reference/observation-contracts.md) | 58 / 754 / 794 / raw observation layouts, action cadence, reset forms. |
| [`reference/reward-contract.md`](reference/reward-contract.md) | Shipped five-term training reward, `gold_ratio`, raw telemetry, force-start, terminal death. |
| [`reference/glossary.md`](reference/glossary.md) | Teacher, student, specialist, lineage, prefix replay, start-jump, canonical, clean route, etc. |
| [`reference/run-artifacts.md`](reference/run-artifacts.md) | What each `runs/` directory contains and how to read its summaries. |
| [`reference/deployed-model-58.md`](reference/deployed-model-58.md) | The frozen 58→256→256→2 JSON student shipped in the game. |

## Repo map

| Path | Purpose |
|---|---|
| `src/plaiground/` | Importable training/data/evaluation package. See [`../src/README.md`](../src/README.md). |
| `scripts/` | Thin baseline, genetic, supervised, reinforcement, and evaluation entry points. |
| `tools/` | Route filtering, start-pool building, observation checks, and reporting utilities. |
| `experiments/` | Committed experiment manifests and result notes. |
| `experiments/0000_neat_clean_450/` | NEAT clean-route collection manifest. |
| `experiments/0001_ppo_clean_start_1_1/` | Lean PPO clean-start manifest. |
| `src/plaiground/envs/` | `GodotEnv`, Gymnasium/SB3 adapters, vector env, legacy wrapper. |
| `src/plaiground/observations/spec.py` | Python mirror of the game's `AIObservationSpec`. |
| `tests/` | Protocol, preprocessing, model, route, pool, and experiment tests. |
| `runs/` | Generated artifacts (gitignored): checkpoints, eval summaries, logs, per-run userdata. |
| `archive/2026-10-27-cleanup/` | Historical scripts/runs/docs. Start at its `README.md`. |
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
- **Cross-repo claims** should name the game branch/commit when they depend on
  game code. The game vault is the source of truth for game behavior.
- **Investigation notes** go in `docs/audit/`; living issues go in
  `docs/known-issues.md`; current state goes in `docs/status.md`.
