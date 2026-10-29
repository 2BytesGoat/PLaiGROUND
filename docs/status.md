# Status

**Updated:** 2026-10-29
**Branch:** `neat`
**HEAD:** `707b745` — `lean inputs`
**Working tree:** the repository restructure (`src/plaiground/`, `scripts/`,
`tools/`, `experiments/`, `Makefile`, package metadata), the clean-start plan
and audits, the lean encoder, route filtering/start pools, reward/start
changes, and the active PPO run docs are uncommitted.
**Game source:** `../dragon-jump-remaster` (`prepare-for-expo`, HEAD `adb9278`
`more state changes`; the raw/lean observation, force-start, reward,
terminal-death, and clean-start world-state code is in HEAD; game docs are
dirty with the new lean decision/status updates. World-state/powerup restore
and `info_request` are implemented, but the clean-start pool/eval gates are
not yet run.)

> This page is the current state. Historical sections in `README.md` and older
> experiment docs are not authoritative. When they disagree, this page and
> [`index.md`](index.md) win.

## Current focus

**Lean DQN 0013 selected (2026-10-29).** Experiment 0013 continued the 1M DQN
with the saved 200k replay buffer and selected the **1.85M** checkpoint:

```text
runs/dqn_clean_start_1_1_8x1_buffer_finetune/dqn/dqn_final.zip
```

Deterministic evaluation:

```text
canonical:    1/1, 1.6167 s
train pool:   12/12, median 3.2500 s, best 1.6167 s
holdout pool: 8/8, median 3.7917 s, best 3.3167 s
```

The fast 1.6167 s route became the deterministic argmax by 1.8M and remained
stable at 1.825M and 1.85M. The original 1.025M–1.1M checkpoints all stayed at
1.8667 s, so the planned +100k continuation was too short. SB3 treats
`total_timesteps` as additional steps when the timestep counter is not reset;
the run was stopped at 1.853M and the best checkpoint was promoted. The 1.85M
model improves canonical time by **0.5333 s** over the 600k PPO baseline and
**0.2500 s** over the 1M DQN. See
[`../experiments/0013_dqn_clean_start_1_1_8x1_buffer_finetune/README.md`](../experiments/0013_dqn_clean_start_1_1_8x1_buffer_finetune/README.md).

**Stable 600k PPO (historical baseline, 2026-10-28).** Experiment 0007 resumed
the 500k checkpoint for 100k with **8×1 / `n_steps=600` / `batch_size=480`**.
It was the selected 1-1 policy before the lean DQN work:

```text
runs/ppo_clean_start_1_1_best_100k/ppo_pilot_600k.zip
```

Deterministic evaluation:

```text
canonical:    1/1, 2.1500 s
train pool:   12/12, median 3.3250 s
holdout pool: 8/8, median 3.8583 s
```

Training canonical stability over the 100k continuation:

```text
≤2.2 s:   97.7%
timeouts: 0.04%
best:     2.133 s
```

This is a stability win, not a speed win: there were no canonical training
episodes below 1.9 s in the 100k run. The 500k checkpoint remains the
shorter-horizon baseline. See
[`../experiments/0007_ppo_clean_start_1_1_stability_100k/results.md`](../experiments/0007_ppo_clean_start_1_1_stability_100k/results.md).

**Fresh PPO clean-start (2026-10-28, complete; 500k baseline).** Experiment 0004
trained a new lean PPO model from scratch for **500,000 timesteps** using the
latest 1-1 NEAT routes and the corrected 12 train / 8 holdout start pool. No old
PPO checkpoint, preprocessor, or output directory was reused. The 500k baseline
checkpoint is:

```text
runs/ppo_clean_start_1_1_fresh/checkpoints/ppo_500000_steps.zip
```

Deterministic evaluation:

```text
canonical:    1/1, 2.1500 s
train pool:   12/12, median 3.3333 s, best 2.1500 s
holdout pool: 8/8, median 3.8583 s, best 3.4167 s
```

The 475k checkpoint regressed to canonical timeout, 6/12 train, and 5/8
holdout; 500k recovered. The 500k run is complete and preserved in
`runs/ppo_clean_start_1_1_fresh/`.

**1M continuation (2026-10-28, complete; 500k retained).** Experiment 0005
resumed the fresh 500k checkpoint for the remaining **500,000 timesteps** and
reached 1,000,400. The final 1M model was evaluated:

```text
canonical:    1/1, 2.1500 s
train pool:   12/12, median 3.4167 s
holdout pool: 8/8, median 4.2750 s
```

The 500k checkpoint was retained at the time because its train and holdout
medians were better (3.3333 s and 3.8583 s); the later 600k stable continuation
superseded it without a time regression. The 1M run is preserved in
`runs/ppo_clean_start_1_1_resume_1m/`. Training-monitor canonical time peaked
around 850k–950k, collapsed at 970k–980k, and recovered near 990k. See
[`../experiments/0005_ppo_clean_start_1_1_resume_1m/results.md`](../experiments/0005_ppo_clean_start_1_1_resume_1m/results.md).

**PPO stability sweep (2026-10-28, complete; 8×1/600/480 selected).**
Experiment 0006 tested five 50k continuations from the selected 500k
checkpoint. The old 8×1 / 150 / 120 control degraded to canonical **4.9167 s**,
train **8/12**, and holdout **5/8**. Larger rollouts fixed it:

```text
8×1 / 300 / 240:  canonical 2.1500 s, train 12/12 3.3000 s, holdout 8/8 3.8583 s
8×1 / 600 / 480:  canonical 2.1500 s, train 12/12 3.3250 s, holdout 8/8 3.8583 s
16×1 / 150 / 240: canonical timeout,     train 6/12,         holdout 5/8
16×1 / 300 / 480: canonical 2.1500 s, train 12/12 3.4500 s, holdout 8/8 3.9667 s
```

The selected continuation config is **8 environments × 1 agent, `n_steps=600`,
`batch_size=480`** (4,800 decisions per update). It tied the best canonical and
holdout results, had the strongest training-monitor stability, and was faster
than the old config. The 16×1/300 variant was the throughput reference (~820
fps) but had worse deterministic pool medians. See
[`../experiments/0006_ppo_clean_start_1_1_stability_sweep/results.md`](../experiments/0006_ppo_clean_start_1_1_stability_sweep/results.md).

**100k stability confirmation (2026-10-28, complete; 600k PPO selected at the time).**
Experiment 0007 resumed the 500k checkpoint for 100k timesteps with the
winning 8×1/600/480 config. The 600k model is stable and matches the best
deterministic results: canonical 2.1500 s, train 12/12 at 3.3250 s, holdout
8/8 at 3.8583 s. It is selected as the stable 1-1 policy.

**Time-trial reward experiment (2026-10-28, complete; no speed gain).**
Experiment 0008 resumed the 225k checkpoint for 100k canonical-only timesteps
with the new game reward (`5.0/s` time pressure, `+50` flat under-gold finish,
`-100` death, `-100` training timeout). The 325k deterministic result is
canonical **2.1500 s**, train **12/12 median 3.2917 s**, and holdout **8/8
median 3.8583 s**. It did not lower canonical time; the flat bonus applies
equally to 2.13 s and 1.63 s. The 600k PPO model remained the best PPO baseline. See
[`../experiments/0008_ppo_time_trial_1_1_225k_100k/results.md`](../experiments/0008_ppo_time_trial_1_1_225k_100k/results.md).

**Continuous under-gold time-trial (2026-10-28, complete; reward fixed, route
not learned).** Experiment 0009 was the direct A/B against 0008: same 225k
source, same 100k canonical-only budget, same 8×1/600/480 topology, with
`min(50, 25 * max(0, gold_time - run_time))` replacing the flat 50. The
continuous bonus correctly ranks 1.633 s (reward 556.8) above 2.133 s (546.4),
but the fast route appeared only once in 3,114 canonical episodes. The 325k
deterministic result is canonical **2.1500 s**, train **12/12 median 3.5583 s**,
holdout **8/8 median 4.9500 s** — worse than 600k. The 600k stable model remains
selected. More exploration/topology, not another 8×1 canonical-only run, is the
next step. See
[`../experiments/0009_ppo_time_trial_continuous_1_1_225k_100k/results.md`](../experiments/0009_ppo_time_trial_continuous_1_1_225k_100k/results.md).

**8×8 continuous-reward time-trial (2026-10-29, complete; no speed gain).**
Experiment 0010 tested 64 virtual environments with a 9,600-decision rollout
from 225k. It completed at 330.6k: canonical **2.1500 s**, train 12/12 median
**3.4167 s**, holdout 8/8 median **4.2750 s**. The 1.833 s training sample was
not retained; only ~10 PPO updates fit into 100k. The 600k stable model remains
selected. See
[`../experiments/0010_ppo_time_trial_continuous_8x8_225k_100k/results.md`](../experiments/0010_ppo_time_trial_continuous_8x8_225k_100k/results.md).

**Lean 8×8 DQN clean-start baseline (2026-10-29, complete; negative at 1M).**
Experiment 0003 trained from scratch on the same inputs as PPO: lean 188 →
fitted 954 preprocessing, 512×512 ReLU, 8 processes × 8 agents = 64
environments, 1M timesteps, the validated 12-start train / 8-start holdout
pools, and the current reward. The 1M result is canonical timeout, train 8/12
(median 2.9750 s), and holdout 7/8 (median 5.2333 s). It produced only ~3,900
gradient updates because of the 64-env batch; the 8×1 lean run produced ~31k at
the same experience scale. The 600k PPO model was still the best PPO baseline. See
[`../experiments/0003_dqn_clean_start_1_1/results.md`](../experiments/0003_dqn_clean_start_1_1/results.md).

**DQN update-count/n-step run (2026-10-29, complete; source checkpoint).**
Experiment 0012 was the corrected lean 8×1 DQN: 1M timesteps,
`gradient_steps=4`, 3-step returns, 200k replay, 50% exploration, and final
epsilon 0.02. It produced about 123k gradient updates instead of the failed 8×8
run's ~3,900. Deterministic evaluation was canonical **1.8667 s**, train 12/12
median **3.2833 s**, and holdout 8/8 median **3.8583 s**. The saved 200k replay
buffer contained 41 canonical 1.633 s episodes and seeded experiment 0013.

**Buffered DQN fine-tune (2026-10-29, complete; selected).** Experiment 0013
continued the 1M DQN using the saved 200k replay buffer, which contains 41
canonical 1.633 s episodes. It kept the 8×1 lean setup and
`gradient_steps=4`, and continued the timestep counter rather than resetting.
The planned +100k was too short: all 1.025M–1.1M checkpoints stayed at 1.8667 s;
the fast route became deterministic at 1.8M. Training was stopped at 1.853M and
the 1.85M checkpoint selected at **1.6167 s** canonical, train 12/12 median
**3.2500 s**, holdout 8/8 median **3.7917 s**.

**NEAT 1-2 transfer (2026-10-29, complete; first 1-2 pool).** Experiment 0014
evaluated the final 192-genome 1-1 population
(`runs/neat_clean_450/neat-checkpoint-25`) on 1-2, then resumed it there and
collected clean routes. The zero-shot pass finished 46/192 genomes, found 23
unique routes and 5 clean routes, best **3.5000 s**. Six generations on 1-2
raised this to **25 clean routes** (79 unique), best **3.3667 s**, median
**3.6333 s**, with 576 checkpoint snapshots. A permissive filter (Jaccard 0.5,
8 per cluster) produced 7 clusters and 19 routes (16 train / 3 holdout), but one
cluster supplied 22 of 32 train starts. The recommended diversity filter
(Jaccard 0.7, one route per cluster) yields **12 clusters**, 9 train / 3 holdout
routes, and a pool with **31 train entries (30 state + canonical) / 20 holdout
states**; the largest train cluster contributes 11/31 starts. All 20 holdout
states also occur on a train route, so the split is route-held out, not
state-disjoint. Use
`runs/neat_clean_450_1_2_transfer/pool_j07_cap1/`. See
[`../experiments/0014_neat_transfer_1_2/README.md`](../experiments/0014_neat_transfer_1_2/README.md).

**Zero-shot DQN on 1-2 (2026-10-29, complete; baseline).** Experiment 0015
ran the selected 1.85M 1-1 DQN on 1-2 with the diverse start pool. Canonical
finished in **9.1167 s**; train was **17/31** finishes (median 5.5000 s, best
3.1833 s) and holdout **15/20** (median 5.9500 s). It can finish but struggles
with many starts. See
[`../experiments/0015_dqn_zero_shot_1_2/README.md`](../experiments/0015_dqn_zero_shot_1_2/README.md).

**Multi-level DQN fine-tune (2026-10-29, pilot complete; candidate).**
Experiment 0016 resumed the 1.85M 1-1 DQN with four 1-1 and four 1-2
environments for 100k additional steps. The 1-1 retention gate passed exactly:
canonical **1.6167 s**, train 12/12 median **3.2500 s**, holdout 8/8 median
**3.8000 s**. The same model improved 1-2 from zero-shot canonical **9.1167 s**
to **3.7500 s**, and now finishes **31/31** train starts (median 4.3833 s, best
3.0500 s) and **20/20** holdout starts (median 4.0500 s, best 3.0500 s), with no
deterministic timeouts. It is a strong level-agnostic 1-1/1-2 candidate, but the
selected policy remains the 1-1 model until a promotion decision. See
[`../experiments/0016_dqn_multilevel_1_1_1_2_transfer/README.md`](../experiments/0016_dqn_multilevel_1_1_1_2_transfer/README.md).

**1-2 continuation (2026-10-29, complete; candidate).** Experiment 0017
continued the 4×1-1 + 4×1-2 mix from the 1.95M pilot for another 100k steps.
The 1-1 gate still passes: canonical **1.6167 s**, train 12/12 median
**3.2667 s**, holdout 8/8 median **3.8000 s**. 1-2 start-pool medians improved
over the pilot (train **4.0167 s**, holdout **3.3500 s**) while canonical stayed
flat at **3.7833 s**; all 31 train and 20 holdout starts finish with no
timeouts. The model is a strong level-agnostic 1-1/1-2 candidate, but canonical
1-2 is still slower than the old NEAT/PPO references. The next planned step is a
4+2+2 schedule for 1-3 with a ~300k budget.

A deterministic action/Q diagnostic found that the slow 1-2 canonical route is
**confident but jittery**, not epsilon noise: 17 action flips in 55 decisions
(flip rate 0.315), mean Q-margin 19.1, while the fast 1-2 state has 6/27 flips
(0.231) and mean margin 31.3. The clean 1-1 canonical route has 5/23 flips
(0.227), so the canonical 1-2 policy is the outlier. This points to route/action
policy quality rather than Q ties. Diagnostic artifacts are under
`runs/dqn_zero_shot_1_2/eval/diagnostics/`. See
[`../experiments/0017_dqn_multilevel_1_1_1_2_100k/README.md`](../experiments/0017_dqn_multilevel_1_1_1_2_100k/README.md).

**NEAT transfer rule (2026-10-29).** For every new level N, collect routes by
evolving the **previous level's evolved NEAT population** on N, not by
transferring from 1-1. Zero-shot transfer is diagnostic only; canonical-only
start pools are a fallback. For 1-3, the seed is
`runs/neat_clean_450_1_2_transfer/neat-checkpoint-31`. This rule is also in
[`../AGENTS.md`](../AGENTS.md).

**1-3 NEAT route collection (2026-10-29, complete; zero finishers).**
Experiment 0020 evolved the 1-2 population on 1-3 with a 300-step horizon for
20 generations. It finished in **532.8 s** with **0/192 finishes**, 0 clean
routes, and no checkpoint files. The best genome plateaued at fitness 74.67
from generation 32 and ended around `y≈424`; the exit is `y=10` from a
`y≈416` start. This is a wall-jump capability gap, not a timeout near the
finish. Experiment 0019, the canonical-only 1-3 DQN fallback, was stopped at
2.274M and is not a candidate.

**Exploratory 1-3 NEAT (2026-10-29, complete; finishers found).** Experiment
0021 ran a fresh 384-genome, higher-mutation search on 1-3 for 30 generations.
It found **12 unique routes and 113 checkpoint snapshots**, best **4.2000 s**,
with 8/384 finishes in the final generation. No route passed the strict clean
filter: the best route has revisit ratio 0.116 against a 0.10 limit, with zero
backtracks and 0.21 flip rate. This is a real capability jump over the zero
finishers from the 192-genome transfer.

**Longer 1-3 NEAT (2026-10-29, complete; strong portfolio).** Experiment 0022
continued the 384-genome population for 40 more generations in 1,610.1 s. It
finished at **220/384 (~57%)**, with **304 unique routes**, 6,361 finisher
records/checkpoints, and **53 strict clean routes**. The fastest route is
4.2333 s but unclean; the best clean route is **5.4667 s** and the clean median
is **5.9833 s**. The clean set spans 13 route clusters at Jaccard 0.5 (26 at
0.7). This is enough for a real 1-3 start pool and a 1-4 seed portfolio.

**Next NEAT step — two-tier discovery + deployment portfolio (planned).**
After 0022, do not force a Gaussian on the live population. Separate:

```text
discovery:   large/high-mutation NEAT; non-finishers allowed; fitness + novelty
deployment:  finishers only; best genome per route cluster × finish-time bin
```

Suggested deployment portfolio: 10–15% speedrunners (<4.5 s), 60–70% median
finishers (4.5–6.0 s) across different clusters, 10–15% slow/robust only if
behaviorally distinct, and 5–10% exploration kept outside the final seed.
“All must finish” is a final portfolio filter, not an early training constraint.
Vanilla NEAT parameters can widen the distribution but cannot enforce this
portfolio; that needs custom reproduction or a MAP-Elites-style archive. For the
1-4 seed, prefer a portfolio (speedrunner, robust median, diverse route
representative) over one average genome. See
[`../experiments/0022_neat_explore_1_3_long/README.md`](../experiments/0022_neat_explore_1_3_long/README.md).

**Double DQN continuation (2026-10-29, complete; 2.9667 s candidate).**
Experiment 0018 resumed the 2.05M level-agnostic 1-1/1-2 model for 300k steps
with a dedicated Double-DQN class and entry point. The final 2.35M checkpoint
failed the 1-1 holdout gate (median 4.1833 s), so the **2.15M checkpoint** was
selected instead:

```text
1-1 canonical: 1.6167 s
1-1 train:     12/12, median 3.2750 s
1-1 holdout:   8/8, median 3.7917 s   PASS
1-2 canonical: 2.9667 s
1-2 train:     31/31, median 3.9167 s, best 2.9667 s
1-2 holdout:   20/20, median 3.1417 s, best 3.0500 s
```

The selected DDQN checkpoint ties the best PPO 1-2 canonical reference and
beats the old NEAT 1-2 transfer, while preserving the 1-1 behavior. The DDQN
canonical action trace is also smoother: 10 flips / 44 decisions versus 17 / 55
for the 2.05M DQN. The selected model is
`runs/dqn_multilevel_1_1_1_2_ddqn_300k/dqn/checkpoints/ddqn_2150000_steps.zip`.
See
[`../experiments/0018_dqn_multilevel_1_1_1_2_ddqn_300k/README.md`](../experiments/0018_dqn_multilevel_1_1_1_2_ddqn_300k/README.md).

The code is now split into an importable
`src/plaiground/` package, thin `scripts/`, data/evaluation `tools/`, and
committed `experiments/` manifests. The package is installed editable with
`uv pip install --python .venv/bin/python -e . --no-deps`. The 1M PPO process
that was running during the move has since completed; new invocations use
`scripts/reinforcement/01_ppo.py`. The old-to-new
mapping and verification are in
[`audit/2026-10-28-repository-restructure.md`](audit/2026-10-28-repository-restructure.md).

**DQN route-spawns (2026-10-28, design pivoted).** The 794 `multi_hot_sensor`
prefix run is now a historical baseline, not the clean-start target. The
simplified reward, force-start reset, and terminal spike death are shipped.
The prefix baseline was stopped at ~129.5k/200k for the clean-start design
discussion; checkpoints 25k–125k exist, with no final model or eval. The
clean-start core is now implemented: raw state, source-checkpoint time, held
powerups, dynamic `world_state`, one-pass NEAT capture, and a canonical/state
pool. The next step is full clean-start validation and evaluation. See
[`experiments/dqn-route-spawns.md`](experiments/dqn-route-spawns.md), the
pipeline review in
[`audit/2026-10-28-dqn-pipeline-review.md`](audit/2026-10-28-dqn-pipeline-review.md),
and the clock-bias audit in
[`audit/2026-10-28-start-pool-clock-bias.md`](audit/2026-10-28-start-pool-clock-bias.md).

**PPO clean-start baseline (2026-10-28, complete; 225k selected).** The clean
1M attempt collapsed around 296k and is preserved in
`runs/ppo_clean_start_1_1_failed_1m/`. A stable continuation resumed from
`ppo-200000.zip` with `lr=1e-4`, `target_kl=0.02`, `ent_coef=0.02`, and 25k
checkpoint/evaluation cadence. It stopped at 300k on patience and selected the
**225k** checkpoint in `runs/ppo_clean_start_1_1/`.

Deterministic results (old clone evaluation: one canonical trace repeated 100
times, not 100 independent trials):

```text
canonical:    100/100 clones, median 2.3167 s
train pool:   14/14 finishes, median 3.5583 s, best 2.1500 s
holdout pool: 10/10 finishes, median 5.2250 s, best 3.4167 s
```

The policy uses the Godot **lean 188-value** packet, Python one-hot
preprocessing (188 → 954), and canonical 0.70 / train-state 0.30 sampling. It
deliberately does **not** use 794 or prefix replay. The holdout is
route-cluster-level only; it is not state-unseen. The model is not installed.
Compact JSON/distillation options are in
[`experiments/compact-policy-deployment.md`](experiments/compact-policy-deployment.md).

**PPO clean-start horizon test (2026-10-28, stopped at 972.6k; 400k selected).**
Experiment 0002 resumed the 225k policy and ran to 972,600 timesteps. It found a
new best at **400k: 2.1500 s canonical**, then oscillated between the healthy
2.15 s mode and deterministic jump-loop timeouts. The run was stopped manually
for investigation; no retraining was performed.

The timeout investigation found:

- canonical evaluation was running 100 identical deterministic clones and
  reporting them as 100 independent trials;
- the 575k policy diverged from the 400k policy at canonical step 5, jumped into
  a wall, and looped until timeout;
- the old pool contained non-standable state starts at `(7,7)` and `(24,10)`.

Fixes now in place:

- canonical evaluation reports **one deterministic trial**;
- the pool builder rejects non-standable cells, and the runtime env rejects a
  silent spawn fallback;
- `tools/watch.py` previews a PPO/DQN checkpoint in the game window.

The corrected 400k evaluation is canonical **1/1 at 2.15 s**, train pool
**12/12**, holdout pool **8/8**. See
[`../experiments/0002_ppo_clean_start_1_1_horizon_1m/`](../experiments/0002_ppo_clean_start_1_1_horizon_1m/)
and
[`audit/2026-10-28-ppo-timeout-root-cause.md`](audit/2026-10-28-ppo-timeout-root-cause.md).

**Lean training harness (2026-10-28, complete).** Both reinforcement trainers
are now config-driven and share the non-algorithm plumbing:

```text
scripts/reinforcement/01_ppo.py     train only, 7 flags
scripts/reinforcement/02_dqn.py     train only, 7 flags
tools/evaluate.py                   one evaluator for PPO and DQN
tools/extract_route_starts.py       DQN route extraction

ppo.py                    154 lines  model + learn
dqn.py                    116 lines  model + learn
experiments/runner.py     356 lines  config -> run paths/overrides
envs/factory.py           159 lines  env, vec env, eval env, ports, HOME
observations/lean_io.py   131 lines  load/create lean preprocessing
evaluation/rollout.py     492 lines  canonical/pool rollout + summaries
data/route_starts.py      505 lines  NEAT checkpoint extraction
```

The command surface is now `--config`, `--timesteps`, `--resume`,
`--output-dir`, `--show`, `--device`, `--verbose`. The manifest supplies the
level, environment shape, observation encoding, model, hyperparameters,
checkpoint cadence, data paths, and evaluation sets.

Removed from the active path:

- `ppo_cli.py`, `ppo_callbacks.py`, `ppo_env.py`, `ppo_evaluation.py`, and
  `ppo_preprocessing.py`;
- the PPO multi-agent `--agents-per-process` branch;
- DQN `PatienceCallback`, `run_eval_episode`, and the in-trainer extractor;
- the custom episode CSV callback and per-eval checkpoint writer. `VecMonitor`
  writes `monitor.csv`, and SB3's `CheckpointCallback` writes checkpoints;
- in-training eval/patience. Evaluation is a post-run tool, which matches the
  `eval_every = 0` practice used by the active PPO runs.

Commands:

```bash
.venv/bin/python scripts/reinforcement/01_ppo.py \
  --config experiments/0002_ppo_clean_start_1_1_horizon_1m
.venv/bin/python tools/evaluate.py \
  --config experiments/0002_ppo_clean_start_1_1_horizon_1m \
  --model runs/ppo_clean_start_1_1_horizon_1m/checkpoints/ppo-400008.zip \
  --tag 400k
```

DQN has a matching manifest at
[`../experiments/0003_dqn_clean_start_1_1/`](../experiments/0003_dqn_clean_start_1_1/).
The 400k result remains the acceptance test for evaluation parity
(canonical 1/1 at 2.15 s, train 12/12, holdout 8/8). The suite is 55 tests.
See
[`audit/2026-10-28-training-harness-composition.md`](audit/2026-10-28-training-harness-composition.md).
The 0002 training process is stopped.

The collection run stopped at generation 25 with **452 unique clean routes**
and 1,486 checkpoint files. Filtering at `jaccard=0.5`, `max-per-cluster=8`, and
`holdout-fraction=0.25` produced **95 clusters / 216 routes**: 158 train and 58
holdout routes. Checkpoint expansion through the grounded/state/time-to-exit
gates produced only **13 train states and 10 holdout states** because most
snapshots are mid-air. All 10 holdout state keys also occur on train routes, so
this is a route-cluster split, not a state-unseen holdout.

`src/plaiground/data/route_filter.py` + `tools/filter_routes.py` perform the
path filter; `src/plaiground/data/start_pool.py` +
`tools/build_start_pool.py` build the train/holdout pools. The PPO manifest
selects the lean 188-value encoding and Python preprocessing; the fitted
pipeline is saved with the run and applied in canonical and start-set
evaluation.
See [`../experiments/0001_ppo_clean_start_1_1/`](../experiments/0001_ppo_clean_start_1_1/),
[`experiments/ppo-clean-start.md`](experiments/ppo-clean-start.md), and
[`audit/2026-10-28-clean-route-reservoir.md`](audit/2026-10-28-clean-route-reservoir.md).

**Clean-start implementation (2026-10-28).** The one-dict reset, dynamic
`world_state` snapshot, held-powerup restore, and NEAT-side state rejection are
implemented. The game now exposes `Level.capture_world_state()` /
`restore_world_state()`, `Powerup.hold()`/`apply_state()`,
`Player.restore_powerup()`, and an on-demand `info_request` / `world_state`
block. `scripts/genetic/01_neat.py` records checkpoint snapshots + world state during the
population; the historical clean-archive coverage tool preserved the checkpoint file;
`src/plaiground/data/route_starts.py` +
`tools/extract_route_starts.py` build the canonical 0.70 / state 0.30 pool without
replay when checkpoints exist. Validation so far: game AI suite 37/37,
training reset/spawn suites 9/9 (the visuals test requires a truly headless
GdUnit runner), Python suite 28/28, a 192-agent NEAT evaluation produced
87 finishes and 87 checkpoint files, and a raw-state 1-1 DQN smoke trained from
the resulting 10-state pool. Training quality/promotion is still pending. See
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md)
and [`experiments/dqn-clean-start.md`](experiments/dqn-clean-start.md).

**Reward/start/death shipped (2026-10-28).** The training reward is now the
simplified five-term contract: progress, efficiency, gold-ratio time pressure
(`-3.75/s * (1 - gold_ratio)`, `gold_ratio` clamped 0–3), fixed `+500` finish,
and `-50` death. `previous_best` is telemetry only; `reward_profile` is ignored.
Training reset force-starts the run (no idle, no forced first jump). Spike
contact is terminal at the hazard; the next reset uses the normal sampled start,
not canonical. Raw `info` adds `gold_ratio`, `flow_cost`, `flow_in_bounds`, and
`wasted_air_frames` for Python rewards. See
[`reference/reward-contract.md`](reference/reward-contract.md).

**Observation path (2026-10-28).** The raw local-state packet is implemented
and validated: 342 values (7×7 tile IDs + tile states, player state, 8-way
collisions, 8-way flow). The raw player state now carries `gold_ratio` instead
of the old normalized run clock. Godot normalizes units; Python builds the
feature encoding. The 794 sensor path is unchanged but is not the clean-start
target. The one-dict reset, dynamic `world_state` snapshot, held-powerup
restore, and NEAT validation are implemented; the raw 342-value clean-start
state trains end-to-end. [`src/plaiground/observations/`](../src/plaiground/observations/) now
provides the lean-only `StatePipeline` path: `DecodeLeanState` and
`LeanFeatures` for the game's 188-value packet. Godot emits 49 primary tile IDs
+ 49 rotations + 49 signed phases + 9 player + 24 collision + 8 flow; Python
one-hots tile IDs, `state_id`, powerup slots, and collision types, so the model
width is vocabulary-dependent. Game-side primary-symbol priority is implemented;
card respawn and moving-platform phase are still pending. The lean v1 baseline
uses Python one-hot features with the declared `src/plaiground/models/mlp.py` `LeanMLP` 512 × 512
architecture and `lean_policy_kwargs()`; there is no custom extractor. The lean path is wired
into `scripts/reinforcement/01_ppo.py`; the game-side raw 342 encoding remains
available for diagnostics, not as the active preprocessing path.
See
[`audit/2026-10-28-raw-state-observation-decision.md`](audit/2026-10-28-raw-state-observation-decision.md),
[`audit/2026-10-28-lean-state-encoding.md`](audit/2026-10-28-lean-state-encoding.md),
and
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md).

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

**2026-10-28 contract validation:**

- Game reward tests: **16/16 pass** (force-start, fixed finish, clamped
  `gold_ratio`, per-second time term, terminal spike death, raw telemetry).
- Raw observation check: 1-1 and 1-8, **342 values**, cell matches 10/10 and
  20/20, raw `gold_ratio` matches `info`.
- 58 and 794 observation regressions: 1 env × 200 steps, no errors.
- 10k DQN smoke on the clock-safe pool with force-start/new reward: 116
  episodes, `ep_rew_mean` **≈438**, no crash. A 10k deterministic strata eval
  finished **42/44** starts (canonical 1/1 at 3.43 s; medium 27/28; hard
  14/15; no model promotion). Old start-jump checkpoints are out of
  distribution.
- 10k PPO 1-env smoke runs end-to-end; canonical eval is **0/100** at 10k
  (too short to judge; no crash). Use a longer run before comparing.


## DQN status (2026-10-28)

- `scripts/reinforcement/02_dqn.py` trains from a config manifest;
  `tools/extract_route_starts.py` builds the pool and `tools/evaluate.py`
  evaluates checkpoints. The 794 path is a legacy baseline; the clean-start
  target is raw state. The 58-value cell extractor is archived/fallback only.
- `runs/dqn_startpool_1_1_smoke/` — 10k smoke on the clock-safe pool
  (canonical 0.7 / prefix 0.3): 61/93 finishes, no cell resets.
- `runs/dqn_startpool_1_1/` — prefix-replay baseline. Stopped at ~129.5k/200k
  on 2026-10-28 for the design discussion. Checkpoints 25k–125k exist; no
  final model or deterministic eval. Training stats: 79% finish, last 200
  97.5%, median 4.0 s, best 1.62 s stochastic. Do not quote the 1.62 s as a
  result.
- `runs/dqn_route_spawns_1_1/` — original extractor attempt; `spawns/` is
  empty. The first extract stopped before writing the pool.
- No clean-start run exists; the plan is
  [`experiments/dqn-clean-start.md`](experiments/dqn-clean-start.md).

## Next

1. **Decide scope.** The game vault currently says DQN/DDQN is deferred until
   after the convention (A32/A33, decisions 2026-10-27). This repo says “active
   experiment.” Record the decision in the game vault or keep DQN explicitly
   side-only.
2. **Raw-state path.** The Godot packet is implemented and validated (342
   values, 1-1 and 1-8) and now includes `gold_ratio`. Add the Python lean
   encoder and parity/throughput gates, then cut over. See
   [`audit/2026-10-28-raw-state-observation-decision.md`](audit/2026-10-28-raw-state-observation-decision.md).
3. **Validate the clean-start contract.** The game world-state/powerup
   restore, one-dict reset, `info_request`, Python pass-through, and NEAT
   one-pass checkpoint capture are implemented. Re-record 1-8+ powerup routes
   and run the full 1-1/1-8 round-trip, strict NEAT rejection, and raw smoke
   gates. Use the source-checkpoint time and the facing/neighbor holdout. See
   [`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md)
   and [`experiments/dqn-clean-start.md`](experiments/dqn-clean-start.md).
   Do not start with cell-only spawns or 794.
4. **Compact the PPO candidate for deployment.** The original 1M run collapsed
   at ~296k; the horizon run found the **400k / 2.1500 s** checkpoint before it
   was stopped. Preserve that teacher and distill it into a smaller flat MLP for
   JSON deployment. Start with `954 → 256 → 256`, then 192/128 if quality holds.
   Review the compact-policy options before building the supervised pipeline.
   The current holdout is route-cluster-only, not state-unseen. See
   [`experiments/compact-policy-deployment.md`](experiments/compact-policy-deployment.md)
   and [`../experiments/0002_ppo_clean_start_1_1_horizon_1m/`](../experiments/0002_ppo_clean_start_1_1_horizon_1m/).
5. **Smoke the new reward/start/death contract.** The 10k 794/prefix DQN smoke
   and 42/44 legacy strata eval validate the shipped reset/reward/death paths,
   not clean-start quality. PPO 10k canonical is 0/100; run a longer smoke
   before 4 envs × 200k.
6. **Recompute the clean-start pool.** Record checkpoint snapshots and
   `world_state` during the population (one-pass), keep the best final time per
   route signature, filter near-duplicate paths with `42`, split whole clusters
   into train/holdout, and build the raw-state pool from the selected routes.
   If capture overhead is too high, use the selected-route replay fallback.
   The old 794/prefix A/B is historical.
7. **Keep the game baseline frozen.** No DQN/PPO/754/794 model is installed in
   the game until the full 16-level and in-game gates pass.
8. **Use the lean harness.** PPO and DQN are config-driven thin adapters; the
   shared evaluator is `tools/evaluate.py`. Do not add in-training eval or
   algorithm-specific callbacks without a concrete need. The old strata
   evaluator, archive-coverage tool, and per-spawn DQN evaluator are removed.
   The 0002 training process is stopped. See
   [`audit/2026-10-28-training-harness-composition.md`](audit/2026-10-28-training-harness-composition.md).

## Known issues

See [`known-issues.md`](known-issues.md) for the living list. The highest-impact
current items are clean-start validation/holdout coverage, the one-tick reset
clock drift, the all-or-nothing extraction, and the checkpoint-vs-route holdout
ambiguity. The old per-spawn evaluation cost and the separate DQN
evaluator/patience surface are resolved; evaluation is now
`tools/evaluate.py`.
