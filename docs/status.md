# Status

**Updated:** 2026-10-28
**Branch:** `neat`
**HEAD:** `d7e173d` — `checkpointing before i lose my mind`
**Working tree:** clean at `d7e173d` before the 2026-10-28 docs pass; the docs
restructure is uncommitted at the time of writing.
**Game source:** `../dragon-jump-remaster` (`prepare-for-expo`, HEAD `bb76299`, dirty with the 794 sensor/docs changes)

> This page is the current state. Historical sections in `README.md` and older
> experiment docs are not authoritative. When they disagree, this page and
> [`index.md`](index.md) win.

## Current focus

**DQN route-spawns (new, 2026-10-28).** A fresh 58-value Double DQN
(`58 → 256 → 256 → 2`) is planned for 1-1. NEAT is used only to generate clean
route checkpoints; no NEAT actions are imitation labels. The script and plan are
committed, but the first extraction run did **not** produce a spawn pool, and no
training or evaluation has run. See
[`experiments/dqn-route-spawns.md`](experiments/dqn-route-spawns.md) and the
detailed review in
[`audit/2026-10-28-dqn-pipeline-review.md`](audit/2026-10-28-dqn-pipeline-review.md).

**794 sensor PPO lineage (most recent working pipeline).** PPO with the
794-value `multi_hot_sensor` encoding has 1-1 and 1-2 checkpoints and a 1-3
transfer attempt. 1-1 canonical is **2.1333 s** with full 1-1 strata coverage;
1-2 canonical is **3.0667 s** with full coverage; 1-3 is **not solved**
(canonical 0/1, medium 25/46, hard 5/27). A 1-3 distilled candidate reaches
**3.9 s** canonical with 43/46 medium and 25/27 hard, but is not installed.

**754 multi-hot PPO reference.** The 200k multilevel checkpoint
(`best_multilevel_3maps.zip`) finishes all 242 starts across 1-1/1-2/1-3 at
canonical **1.6667 / 2.9667 / 5.2333 s**. The 300k continuation kept coverage
but regressed canonical to 1.6667/3.0000/5.5667, so 200k remains selected. Not
installed; the shipped JSON runtime is 58-only.

**Frozen game baseline.** The bundled `student_obs58_256.json` remains the
convention AI baseline: 16/16, 100/100 per level, zero deaths, **78.583 s**
campaign sum, 1-1 **2.333 s**. Do not replace it without the all-16 in-game
gate. See [`reference/deployed-model-58.md`](reference/deployed-model-58.md).

## Verified results

These are from the run artifacts and the game-side docs; they are not
re-measured here.

| Track | Level(s) | Result | Artifact |
|---|---|---|---|
| NEAT 8×24 | 1-1 | **2.15 s** canonical; 460 clean routes; 59/60 starts (51/52 meaningful) | `runs/neat_1_1_8x24_archive/` |
| NEAT transfer | 1-2 | **3.0167 s**; 12 clean routes; 71/72 starts | `runs/neat_1_2_transfer_8x24/` |
| NEAT transfer | 1-3 | **3.5667 s**; 26 clean routes; 62/86 starts | `runs/neat_1_3_transfer_8x24/` |
| 754 PPO multilevel, 200k | 1-1/1-2/1-3 | **1.6667 / 2.9667 / 5.2333 s**; 60/60, 84/84, 98/98 starts | `runs/ppo_multilevel_1_1_1_2_1_3/` |
| 754 PPO multilevel, 300k | 1-1/1-2/1-3 | 1.6667 / 3.0000 / 5.5667 s — worse than 200k | same run dir |
| 794 sensor PPO | 1-1 | **2.1333 s** canonical; full 1-1 strata | `runs/ppo_fresh_1_1_sensor_794/` |
| 794 sensor PPO | 1-2 | **3.0667 s** canonical; full 1-2 strata | `runs/ppo_sensor_1_2_from900k/` |
| 794 sensor PPO | 1-3 | not solved: canonical 0/1, medium 25/46, hard 5/27 | `runs/ppo_sensor_1_3_from1_2/` |
| 794 distilled | 1-3 | **3.9 s** canonical; medium 43/46, hard 25/27, easy 24/24 | `runs/distill_1_3_merged_available/` |
| Deployed 58/256 | 1-1…1-16 | 16/16, 100/100, 0 deaths, **78.583 s** sum; 1-1 2.333 s | `godot/src/resources/ai/student_obs58_256.json` in the game repo |

## DQN status (2026-10-28)

- `src/41_dqn_route_spawns.py` has `extract`, `train`, `eval`, and `all` modes.
- `runs/dqn_route_spawns_1_1/` contains `routes/` (460 clean routes, ~6 MB) and
  two Godot logs. `spawns/` is empty; there is no `dqn/` or `eval/`.
- The first extraction run was created at 00:14, the script was edited at 00:16,
  and the game log stops at `TRAINING_SPEEDUP x8.0` at 00:16:21. No process was
  running when this was checked. Treat it as **stopped before writing the pool**,
  not as a completed extract.
- No DQN training result exists. Do not quote one.

## Next

1. **Decide scope.** The game vault currently says DQN/DDQN is deferred until
   after the convention (A32/A33, decisions 2026-10-27). This repo says “active
   experiment.” Record the decision in the game vault or keep DQN explicitly
   side-only.
2. **Fix the extraction before training.** Capture grounded `state` snapshots
   (run time, velocity, `run_active`, `started_walking`), validate spawn
   acceptance, and write incremental output. See the audit.
3. **Run `extract` to completion** and inspect the pool counts/splits.
4. **Smoke then scale:** 1 env × 10k steps, then 4 envs × 100k, then a
   persistent-env evaluation. Do not start with 100k and cell-only spawns.
5. **Keep the game baseline frozen.** No DQN/PPO/754/794 model is installed in
   the game until the full 16-level and in-game gates pass.

## Known issues

See [`known-issues.md`](known-issues.md) for the living list. The highest-impact
current items are the DQN run-clock/reward mismatch, the all-or-nothing
extraction, the checkpoint-vs-route holdout ambiguity, per-spawn evaluation
startup cost, and the coupling of `41_dqn_route_spawns.py` to the 83 KB
`10_ppo_pilot.py`.
