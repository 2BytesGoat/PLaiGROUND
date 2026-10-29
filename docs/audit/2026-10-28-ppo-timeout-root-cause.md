# PPO timeout and start-pool root cause

Status: root cause identified; eval/pool fixes applied; no retraining  
Date: 2026-10-28  
Run: `runs/ppo_clean_start_1_1_horizon_1m/`  
Stopped at: 972,600 timesteps  
Selected model: 400k checkpoint, canonical 2.1500 s

## Summary

The intermittent `100/100` → `0/100` evaluations were not caused by checkpoint
loading or the preprocessor pickle. They came from two separate issues:

1. **Canonical evaluation ran 100 identical clones.** A deterministic policy at
   the fixed canonical start produces one trajectory. The old evaluator created
   100 agents, reset them all to canonical, and reported the same trajectory 100
   times as a 100-agent success rate.
2. **The deterministic policy developed a jump-loop failure mode.** The 400k
   model and the 575k model followed identical actions for five steps, then
   diverged. The 575k model jumped into a wall and looped on the lower platform
   until timeout.

There was also a separate data bug: the training pool contained state starts at
`(7, 7)` and `(24, 10)`, which Godot reports as non-standable. The game silently
fell back and then applied the state pose, so `spawn_ok` checks were not enough.

## Checkpoint/loading evidence

The checkpoints and preprocessing are consistent:

```text
best_model.zip / ppo-400008.zip: Box(954), Discrete(2)
ppo-575008.zip:                  Box(954), Discrete(2)
preprocessor:                     954 inputs
225k baseline:                    2.3167 s
```

The first failed launch today was a real 58-vs-954 mismatch because the lean
flags were omitted. Once corrected, the model loaded and trained. The old pickle
`ModuleNotFoundError` was a module-path issue after the restructure, not a shape
mismatch.

## Deterministic divergence

Tracing the canonical run with one agent and the same 954 pipeline:

```text
400k model:
  steps 0-4: jump
  step 5:    idle
  result:    2.1500 s finish

575k model:
  steps 0-4: jump
  step 5:    jump  ← first divergence
  step 5:    Walled at [12,10]
  steps 6-14: Walled, then Fall
  steps 15-149: jump repeatedly on the lower platform
  result:    150-step timeout
```

Seed 43 produced the same 575k timeout. Forcing action 0 at step 5 did not fix
it; the 575k model jumped again at step 6 and still entered the loop. This is a
policy-mode change, not one bad action or one bad eval seed.

Metrics around the divergence:

```text
575k: entropy ~0.002, explained_variance 0.50, value_loss ~1,034
600k: approx_kl 0.032, clip_fraction 0.105, value_loss ~6,121
```

The policy became extremely deterministic and brittle, then oscillated between
the healthy 2.15 s mode and the jump-loop mode.

## Why `0/100` was misleading

The old evaluator wrote 100 rows with identical `steps`, `run_time`, `reward`,
`efficiency`, and `flow` values. `0/100` meant one deterministic timeout repeated
100 times; it did not mean 100 independent failures.

## Fixes applied

- Canonical evaluation now runs **one deterministic trial**. The legacy
  `--eval-agents` flag was removed. Start-pool evaluation remains the
  independent-scenario test.
- Patience evaluates the actual result count instead of assuming 100 agents.
- `start_pool.py` now mirrors Godot's standability rule: reject wall, spike,
  reset, exit, and unsupported cells before pool construction.
- `SingleAgentGodotEnv` now raises if Godot reports a different spawn cell than
  the pool requested, instead of silently training on a fallback.
- `tools/watch.py` previews the selected PPO/DQN checkpoint in the game; the
  old PPO-only tracer was removed.
- The train/holdout pools were rebuilt after backing up the old pool.

## Corrected pool

```text
old train/holdout starts: 14 / 10
new train/holdout starts: 12 / 8
rejected non-standable train checkpoints: 5,194
rejected non-standable holdout checkpoints: 1,772
```

Neither `(7, 7)` nor `(24, 10)` is present in the new pools. The old pool is
preserved at `runs/neat_clean_450/pool_pre_standability/`.

## Verification

The 400k best was evaluated once on the corrected pool:

```text
canonical:    1/1, 2.1500 s
train pool:   12/12, median 3.3250 s, best 2.1500 s
holdout pool: 8/8, median 3.8583 s, best 3.4167 s
```

No retraining was performed.

## Remaining issue

The 400k model is a real, fast deterministic policy, but it is narrow. The 575k
mode proves that additional PPO updates can flip the policy into a jump loop.
A future training run should use the corrected pool, independent start-pool
selection, and a stability gate; merely loading or formatting a checkpoint
differently will not fix the policy behavior.
