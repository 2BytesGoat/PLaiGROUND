# Results — 0002

Status: stopped at 972,600 timesteps; 400k selected  
Completed: 2026-10-28

## Outcome

The 1M horizon run was stopped manually to investigate repeated deterministic
timeout windows. The **400k** checkpoint remains the selected policy:

```text
canonical:    1/1, 2.1500 s
train pool:   12/12, median 3.3250 s, best 2.1500 s
holdout pool: 8/8, median 3.8583 s, best 3.4167 s
```

The 400k best is the same 2.15 s policy found during the horizon run. No model
was installed in the game.

## What happened

The run oscillated between healthy 2.15 s evaluations and 0/100 timeout
evaluations. The 0/100 result was one deterministic timeout repeated across 100
identical canonical agents, not 100 independent failures.

Action tracing found the first divergence between the healthy 400k model and
the failing 575k model at canonical step 5:

```text
400k: idle  → settles, then finishes in 2.15 s
575k: jump  → walled at [12,10], falls, loops until timeout
```

Seed changes and a forced step-5 action did not fix the 575k mode. Training
entropy was already near zero at 575k, and value/clip metrics spiked around
600k. This is deterministic policy drift, not a checkpoint-format or pickle
problem.

## Data issue

The old pool contained state starts at `(7, 7)` and `(24, 10)`, which Godot
reports as non-standable. The game silently fell back and then applied the state
pose. The pool was rebuilt with static standability validation:

```text
old: 14 train / 10 holdout starts
new: 12 train /  8 holdout starts
```

The old pool is preserved at `runs/neat_clean_450/pool_pre_standability/`.
The runtime env now raises if the requested spawn cell is not the actual spawn
cell.

## Fixes

- canonical evaluation reports one deterministic trial;
- patience uses the actual trial count;
- start-pool construction rejects non-standable cells;
- runtime reset rejects silent spawn fallback;
- `tools/trace_ppo.py` records per-step action/state traces.

See
[`../../docs/audit/2026-10-28-ppo-timeout-root-cause.md`](../../docs/audit/2026-10-28-ppo-timeout-root-cause.md)
for the full investigation.

## Decision

Keep the 400k **2.15 s** checkpoint as the candidate. Do not continue the
stopped 1M lineage without fixing the policy stability/selection problem. No
retraining was done as part of this fix.
