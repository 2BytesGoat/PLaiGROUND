# 0002 — PPO clean-start horizon test to 1M

Status: stopped at 972,600 steps; 400k selected
Started: 2026-10-28
Base: [`../0001_ppo_clean_start_1_1/`](../0001_ppo_clean_start_1_1/)

> **Harness note (2026-10-28).** This record was produced with the old
> flag-based CLI. The current reproduction is config-driven:
>
> ```bash
> .venv/bin/python scripts/reinforcement/01_ppo.py \
>   --config experiments/0002_ppo_clean_start_1_1_horizon_1m
> .venv/bin/python tools/evaluate.py \
>   --config experiments/0002_ppo_clean_start_1_1_horizon_1m \
>   --model runs/ppo_clean_start_1_1_horizon_1m/checkpoints/ppo-400008.zip \
>   --tag 400k
> ```
>
> `[run].resume_from` and `[run].preprocessing_from` point at the 225k source
> run, so the training command resumes automatically.

## Question

Does the stable 225k clean-start policy keep improving, plateau, or drift when
trained to the full 1M horizon?

The original 1M run collapsed around 296k. The 200k-resumed repair run used a
lower learning rate, lower target KL, and more entropy, then stopped at 300k on
patience. This run resumes from the selected **225k** checkpoint and disables
the early stop while retaining best-checkpoint selection.

## Setup

```text
observation:  lean 188 -> Python one-hot 954
model:        954 -> 512 -> 512 -> 2, Tanh
starts:       canonical 0.70 / train state 0.30
training:     8 envs x 1 decision, 150 decisions per episode
budget:       1,000,000 total timesteps (775k new from 225k; stopped at 972.6k)
PPO:          lr=1e-4, target_kl=0.02, ent_coef=0.02
checkpoint:   every 25,000
evaluation:   deterministic canonical (one trace), every 25,000
patience:     40 evaluations (effectively disabled for this horizon)
```

The original preprocessing pipeline and start pool are reused exactly. The
225k baseline was re-evaluated at launch and reproduced **2.317 s, 100/100**.

## Game provenance

```text
repo:    ../dragon-jump-remaster
branch:  prepare-for-expo
commit:  adb9278
dirty:   true
```

The dirty working-tree diff and exact PLaiGROUND commit at launch are recorded
in `runs/ppo_clean_start_1_1_horizon_1m/metadata.json` and `game.diff`.

## Run directory

```text
runs/ppo_clean_start_1_1_horizon_1m/
```

The existing completed run at `runs/ppo_clean_start_1_1/` remains untouched.

## Result

Stopped manually at **972,600 timesteps** to investigate repeated deterministic
timeout windows. The selected model is the **400k** checkpoint:

```text
canonical:    1/1, 2.1500 s
train pool:   12/12, median 3.3250 s, best 2.1500 s
holdout pool: 8/8, median 3.8583 s, best 3.4167 s
```

The canonical evaluator now reports one deterministic trial, not 100 identical
clones. The train/holdout pools were rebuilt with standability validation; the
old pre-fix pool is at `runs/neat_clean_450/pool_pre_standability/`.

Root cause and fixes:
[`../../docs/audit/2026-10-28-ppo-timeout-root-cause.md`](../../docs/audit/2026-10-28-ppo-timeout-root-cause.md).

No retraining was performed after the stop.
