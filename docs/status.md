# Status

**Updated:** 2026-10-30
**Branch:** `neat`
**HEAD:** `f7853ef` — `getting closer to the best workflow`
**Working tree:** the repository restructure (`src/plaiground/`, `scripts/`,
`tools/`, `experiments/`, `Makefile`, package metadata), the clean-start plan
and audits, the lean encoder, route filtering/start pools, reward/start
changes, and the active PPO run docs are uncommitted.
**Game source:** `../dragon-jump-remaster` (`prepare-for-expo`, HEAD `50dcc66`
`add tool to generate steam auth`, clean tree; the training/observation code is
unchanged since the `fe7821d` commit recorded by the active manifests. The raw/
lean observation, force-start, reward, terminal-death, clean-start world-state,
and the 2026-10-29 hold-aware contract (`legacy_stick` 61-value observation plus
encoding-aware `reassert_held_jump`) are committed. World-state/powerup restore
and `info_request` are implemented; the 1-4 start pool is validated and the
finisher-rich 0046 seed exists. The 188-input 0048 stage and
the expanded 191 fine-tune (0053) were both stopped. The fresh level-by-level
`lean_stick` 191 chain has passed 1-1, 1-2, 1-3, and 1-4: 0058 selected 1M at
**1.5667 s** with full 12/12 and 8/8 coverage; 0060 added 1-2 at 3.0333
canonical; 0063 added 1-3 at 3.1667 canonical; 0067 retained 1-1 at 1.53M;
0073 selected **1.555M** for 1-4 with 1/1 canonical, 11/11 train, and 2/2
holdout. 1-5 now has a selected **3.1667 s** checkpoint (0085); the two-model
chain and the rejected single-model retention attempt are described below.
1-6 now has an **any-start expert** (0099 1.815M): **32/32** NEAT feasible
starts finish with 0 deaths/timeouts at a 6.0417 s median, while canonical is
**4.6000 s** (repeat-stable). The 0091 1.705M **4.5667 s** model remains the
canonical reference but covers only **11/32** starts. The 4.28 s speed gate is
still open; the 4.0667 s replay was not reproducible in isolation and is not
used as teacher data.

**1-7 NEAT search and expert (0101 stopped, 0102 complete, 0103/0104 complete
2026-10-29).** The final 4-frame 1-6 population transferred to 1-7 found
finishers but no strict clean route; 0101 was stopped after 12 generations at
**240 routes / 0 clean**. A high-mutation cleanup seeded from six 1-7 route
clusters (0102) completed 30 generations in **808.3 s** with **902 routes /
37 clusters / 0 strict clean**; best **7.0333 s** (revisit 0.107, backtracks 1).
The 902-route archive expands to **20 grounded feasible starts** (plus
canonical); the 50 near-clean train routes replayed **23/50**, and the
stable-replay filter kept **2,495 transitions / 18 routes** (`source_time -
replay_time <= 0.25`).
The 1.815M 1-6 teacher baseline was canonical timeout but **20/20** feasible
starts (median 11.4917 s, best 7.7500 s).

The 1-7-only DDQN (0103) completed 100k from 1.815M. The selected checkpoint is
**1.877432M**:

```text
canonical:          1/1, 5.2333 s, 0 deaths/timeouts (repeat exact)
near-clean pool:   10/10, 0 deaths/timeouts, median 7.9583, best 5.7667
all-success pool:  18/20, 0 deaths, 2 timeouts (diagnostic)
spawn_ok:          10/10 near-clean, 20/20 all-success
```

The two all-success timeouts are late facing-right states from a high-revisit
route (`8f687...`, revisit 0.43) and are not part of the near-clean gate. The
**provisional** par + 0.30 s gate is **5.2728 s**; 1.877432M passes by 0.0395 s.
The 1.889928M/1.914920M checkpoints have a faster canonical **5.2167 s** but
only 15/16 of 20 all-success starts, so 1.877432M is the any-start selection.
Strict clean routes remain unavailable for 1-7; the accepted 1-7 gate uses
near-clean routes (`revisit_ratio <= 0.20`, `backtracks <= 2`). The final 1-7
target is still open.

**1-8 NEAT and expert (0105-0120 complete 2026-10-30).** The final 1-7
population transferred to 1-8 (DOUBLE_JUMP) and found clean routes immediately:
**26 strict-clean routes / 11 clusters**, best **4.4167 s**, in 16 generations
(0105). After filtering (Jaccard 0.5, 8 per cluster, 25% cluster holdout), the
pool has 16 train / 10 holdout routes, **37 train / 14 holdout** grounded
states, and **40 unique clean states** for evaluation. The 16 train routes
replayed **16/16** exactly; the stable expert buffer kept **1,800 transitions**.

The 1-8-only DDQN (0106) selected **1.939912M** provisionally: canonical
**4.5167 s**, clean **39/40**, all-success **106/107**. A canonical-heavy repair
(0109) reached **40/40** clean states at 1.949912M but canonical stayed 4.5167.
The canonical fast-route fine-tune (0112) reached **4.4667 s** but dropped one
state; two targeted repairs (0114, 0116) narrowed the gap, and a 500-step
low-LR nudge (0118) produced the selected **1.960392M**:

```text
canonical:          1/1, 4.4667 s, 0 deaths/timeouts (repeat exact)
clean pool:        40/40, 0 deaths/timeouts, median 9.8833, best 4.5500
all-success pool: 106/107, 0 deaths, 1 timeout (high-revisit route)
spawn_ok:          40/40 clean, 107/107 all-success
```

The selected model passes the provisional par + 0.30 s gate **4.5144 s** by
**0.0477 s**. The fastest overall NEAT route is **4.4167 s**, but it belongs to
holdout cluster `c000`; the expert reproduces the fastest **train** route at
4.4667 s. That is a coverage/speed split decision, not a checkpoint failure.
The one all-success timeout is `[16,21]` facing `1` from the high-revisit route
`36a6b1e1...` (revisit 0.226), which is not strict clean. The final 1-8 target
is still open. See
[`../experiments/0105_neat_transfer_1_8_from_1_7/README.md`](../experiments/0105_neat_transfer_1_8_from_1_7/README.md)
and
[`../experiments/0118_ddqn_1_8_canonical_nudge/README.md`](../experiments/0118_ddqn_1_8_canonical_nudge/README.md).
The full sequence is in
[`audit/2026-10-30-1-8-expert.md`](audit/2026-10-30-1-8-expert.md).

> This page is the current state. Historical sections in `README.md` and older
> experiment docs are not authoritative. When they disagree, this page and
> [`index.md`](index.md) win.

## Current focus

> **Architecture decision (2026-10-29):** NEAT discovers feasible starts; every
> campaign level gets its own expert trained/evaluated on that level's NEAT
> starts; the final product is one shared student A with no level ID or
> per-level heads. Active implementation is 1-1..1-9; 1-7 has a selected
> near-clean expert (0103 1.877432M), 1-8 has a selected strict-clean expert
> (0118 1.960392M), and 1-9 is a selected **single-route** expert
> (0122 2.060360M) because the user confirmed 1-9 is a precision map with one
> route to optimize. 1-10..1-16 follow the same procedure. See
> [`audit/2026-10-29-expert-portfolio-and-student-decision.md`](audit/2026-10-29-expert-portfolio-and-student-decision.md).

**Level pipeline tooling (2026-10-30).** The post-NEAT artifact step is now
repeatable: `tools/level.py data` classifies strict/near/all-success routes,
filters the accepted tier, builds train/holdout pools, and writes the
deduplicated eval union. It reproduces the existing 1-8 result exactly
(26 strict routes, 37/14 states, 40 eval) and the 1-7 near-clean fallback
(10/10 states). It also warns when the fastest route cluster is holdout-only,
which is the 1-8 4.4167 s / 4.4667 s tradeoff. Follow
[`runbooks/new-level.md`](runbooks/new-level.md) and record selections in
[`reference/level-ledger.md`](reference/level-ledger.md).

**1-9 single-route expert (0121/0122/0123/0124, 2026-10-30).** The 1-8
population transferred to 1-9 immediately. The seeded NEAT run was stopped at
generation 89 after finding 11 strict-clean routes / 6 clusters, best
**2.9500 s**. The fastest route (`926883...`) was replayed in `lean_stick` at
**43/43 decisions, 2.9500 s**, and used for a canonical-only 1-9 DDQN fine-tune
from the 1-8 expert. All eight checkpoints finish canonical 1-9 at **2.9500 s**;
the selected checkpoint is `ddqn_2060360_steps.zip`, under the provisional
**3.0648 s** gate. Training monitor: 85% finish in the last 200 episodes.

Retention is partial: deterministic canonical 1-1..1-9 finishes **1-1
(2.5167 s), 1-3 (6.0167 s), and 1-9 (2.9500 s)**; 1-1 and 1-9 pass the
provisional gates, 1-3 finishes but misses its gate, and 1-2, 1-4, 1-5, 1-6,
1-7, and 1-8 fail. This is a 1-9-only expert, not a multi-level policy. See
[`../experiments/0122_ddqn_1_9_single_route/README.md`](../experiments/0122_ddqn_1_9_single_route/README.md)
and [`reference/level-ledger.md`](reference/level-ledger.md).

**Next level:** 1-10 (DOUBLE_SHORT), par 3.0336 / provisional gate 3.3336. The
1-10 level analysis still needs to be exported. The NEAT population seed is
`runs/neat_transfer_1_9_from_1_8/neat-checkpoint-89`; the expert base is the
selected 1-9 `ddqn_2060360_steps.zip`. See the ledger for the handoff.

**Fresh lean_stick 191 chain (2026-10-29; 1-1..1-5 selected as a two-model chain; 1-6 any-start expert 32/32, 4.60 canonical, 4.28 gate open).** The expanded
191 fine-tune (0053) reproduced the earlier signal-arm failure pattern: 1-3
regressed sharply and 1-4 improved only from 1/11 to 2/11. The chain was
restarted at 1-1 with a fresh 191 backbone, no 188 weights, the 12/8 discovered
1-1 clean starts, and 1M steps.

Stage 1 selected the 1M model:

```text
runs/ddqn_fresh_1_1_lean_stick/dqn/dqn_final.zip
1-1 canonical: 1.5667 s
train pool:    12/12, median 3.2583 s
holdout pool:  8/8, median 3.8000 s
```

The 1.85M continuation (0059) is not needed.

Stage 2 (0060) used a 200k 1-1 buffer from the 1.5667 policy, then trained 4×1-2
+ 4×1-1 for 300k steps. The selected **1.3M** checkpoint is:

```text
runs/ddqn_fresh_1_1_1_2_lean_stick/dqn/checkpoints/ddqn_1300000_steps.zip
1-2 canonical: 3.0333 s
1-2 train:     31/31, median 4.4333 s, best 3.0333 s
1-2 holdout:   20/20, median 3.2333 s, best 3.1000 s
1-1 canonical: 1.5667 s
1-1 train:     12/12, median 3.2833 s
1-1 holdout:   8/8, median 3.8000 s
```

The 1-2 coverage gate passes. The remaining weakness is **1-2 state-start
speed**: the training monitor state median is 4.32 s, so this is a policy
optimization issue, not eval semantics. The 1.1M checkpoint covered only 15/31
train and 12/20 holdout; 1.2M was 30/31 and 19/20; 1.3M is full. The
post-training `dqn_final.zip` is worse on 1-2 holdout (4.0000 s median) than the
1.3M checkpoint, so the checkpoint is selected.

Stage 3 (0063) completed on the fresh 191 chain: 4×1-3 + 2×1-2 + 2×1-1 for
300k additional steps from the 1.3M checkpoint, using the validated
`runs/neat_frequent_rebalance_1_3/` 1-3 pool and holdout, a fresh 200k 1-1/1-2
anchor from the 1.3M model, and protected replay (400k online + 200k anchor,
25% anchor). The **1.5M checkpoint** is selected:

```text
runs/ddqn_fresh_1_1_1_2_1_3_lean_stick/dqn/checkpoints/ddqn_1500000_steps.zip
1-3 canonical: 3.1667 s; train 45/45, median 5.4667; holdout 23/23, 5.5500
1-2 canonical: 3.0167 s; train 31/31, median 4.4167; holdout 20/20, 3.3083
1-1 canonical: 1.5667 s; train 12/12, median 3.4167; holdout 8/8, 4.2667
```

1.5M has the best old-level retention of the candidate set: 1-2 is essentially
the 0060 baseline, 1-3 is fully covered and beats the 0031 canonical reference,
and 1-1 canonical/train coverage are retained. The only remaining regression is
the 1-1 holdout median, driven mainly by one `[25,10]` start (4.6500 s vs
3.5000 s at 1.3M); the other seven starts are within 0.10 s. The 1.6M
checkpoint has the best 1-3 canonical (3.0667 s) but regresses 1-2 canonical to
3.8333 s and holdout to 4.0667 s; 1.575M keeps 1-1 holdout at 3.8000 s but
regresses 1-2 holdout to 3.9500 s. A short 1-1 retention continuation (0067)
is the follow-up for the single remaining holdout gap. See
[`../experiments/0063_ddqn_fresh_1_1_1_2_1_3_lean_stick/README.md`](../experiments/0063_ddqn_fresh_1_1_1_2_1_3_lean_stick/README.md).

**1-1 retention (0067, 2026-10-29; 1.53M selected).** A 50k 1-1-heavy
continuation (4×1-1 + 2×1-2 + 2×1-3) from the 0063 1.5M checkpoint recovered
the 1-1 holdout in 20k steps. The selected 1.53M checkpoint:

```text
1-1 canonical: 1.5667; train 12/12, 3.2583; holdout 8/8, 3.8000
1-2 canonical: 3.0167; train 31/31, 4.3500; holdout 20/20, 3.8833
1-3 canonical: 3.1667; train 45/45, 5.4667; holdout 23/23, 5.5333
```

The failing 1-1 holdout start `[25,10]` recovered from 4.6500 to 3.4333 s and
the 1-1 train median improved from 3.4167 to 3.2583 s. The 1-3 pool is
unchanged and 1-2 canonical/train are retained. One 1-2 holdout start
`[24,17]` regressed from 3.1500 to 4.7333 s, moving the 1-2 holdout median to
3.8833 s. The 1.55M checkpoint over-trained (1-2 canonical 3.8333, holdout
4.1583) and is not selected. The 1-1 retention / old-level checkpoint is
`runs/ddqn_fresh_1_1_retention_lean_stick/dqn/checkpoints/ddqn_1530000_steps.zip`;
the 1-4 chain checkpoint is **1.555M** (0073). See
[`../experiments/0067_ddqn_fresh_1_1_retention_lean_stick/README.md`](../experiments/0067_ddqn_fresh_1_1_retention_lean_stick/README.md).

**Next-level retention schedule (agreed 2026-10-29).** For level N, use
`4×N + 1×(N-1) + 3 retention envs` (for 1-4: `4×1-4 + 1×1-3 + 2×1-2 + 1×1-1`)
with the old-level anchor and per-checkpoint gates on every retained level.

**1-4 probe (0068, 2026-10-29; stopped at 100k, negative).** A 100k probe from
the selected 1.53M checkpoint used `4×1-4 + 1×1-3 + 2×1-2 + 1×1-1`, the
validated 0046 1-4 pool (11 train / 2 holdout), and a fresh 200k 1-1/1-2/1-3
anchor from 1.53M. It is negative at every checkpoint: **1-4 canonical 0/1**
(timeout; 1.63M also death), **1-4 train 1/11** at 6.7833 s, and **1-4 holdout
0/2**. The monitor showed 0/310 canonical finishes and 13/136 state finishes.
Old levels kept full coverage but drifted modestly at 1.63M: 1-1 1.5667
(12/12, 8/8), 1-2 3.1833 (31/31, 20/20), 1-3 3.2333 (45/45, 23/23). This reproduces the
0053 failure: the 1-3 policy does not transfer to 1-4. Do not continue the DDQN
transfer without a different method (NEAT-route distillation, 1-4-specific
curriculum/pool, or a separate 1-4 head). See
[`../experiments/0068_ddqn_fresh_1_4_lean_stick/README.md`](../experiments/0068_ddqn_fresh_1_4_lean_stick/README.md).

**1-4 expert replay (0073, 2026-10-29; complete, selected 1.555M).** The 27
clustered clean NEAT train routes were replayed in the 191 `lean_stick`
environment; 21 reproduce exactly, including the canonical route at 5.7667 s.
The 6 non-reproducing routes are filtered, leaving 2,231 successful transitions
(`stick_active=1` on 18.4%, `jump_held=1` on 69.8%). One DDQN run from the
selected 1.53M checkpoint used `4×1-4 + 1×1-3 + 2×1-2 + 1×1-1`, a protected
25k old-level anchor, and the 1-4 expert buffer sampled for an expert TD loss
plus an action-margin loss; the old anchor also got a margin term. This was a
single training run, not BC-then-DQN.

**Selection: 1.555M** is the 1-4 chain checkpoint. At 1.555M:

```text
1-4 canonical: 1/1 at 4.4667 s
1-4 train:     11/11 at 6.7333 s (best 4.4667)
1-4 holdout:   2/2 at 6.0167 s (best 5.8500)
1-1:           1.5667, 12/12, 8/8
1-2:           3.1500, 31/31, 20/20; holdout 4.4750
1-3:           3.2667, 45/45, 23/23; holdout 5.6500
```

Coverage is full on every old level. The 1-2/1-3 canonical speeds are slower
than the 1.53M baseline (3.0167 / 3.1667). The 1.63M checkpoint keeps a better
1-2 (3.0167, holdout 3.3583) but fails 1-4 (0/1 canonical, 7/11 train, 1/2
holdout). Keep **1.53M** as the old-level-only reference/fallback and **1.555M**
as the 1-4 chain checkpoint. See
[`../experiments/0073_ddqn_expert_replay_1_4_lean_stick/README.md`](../experiments/0073_ddqn_expert_replay_1_4_lean_stick/README.md).

**1-1 → 1-4 windowed chain check (0079, 2026-10-29; reproduces 0073).** Re-ran
the four canonical levels with the 1.555M checkpoint at 1x, one windowed Godot
process per level, on game HEAD `fe7821d`. All four finish with no deaths and
no timeouts: **1-1 1.5667 s, 1-2 3.1500 s, 1-3 3.2667 s, 1-4 4.4667 s**,
matching the 0073 record exactly. The game drift since `adb9278` does not
change the canonical chain. Note that `speedup < 1` is clamped to 1x by the game,
so the 0078 "half speed" show actually ran at normal speed. Watch it with
`make watch` (`WATCH_LEVEL=1-4` for one level). See
[`../experiments/0079_show_chain_1_1_to_1_4_0073/README.md`](../experiments/0079_show_chain_1_1_to_1_4_0073/README.md).

**1-5 route collection + DDQN (0080–0088, 2026-10-29; selected two-model chain).**
0080 extended the 0047 1-5 population from generation 20 to 50: 284 routes,
16 clean, best clean **4.6833 s**, generation 50 382/384. The raw reward ranked
a 7.1167 s finisher above the 4.6833 s route, so 0081 added
`--time-fitness-weight 20` and ran at **action_repeat 2** (30 Hz). It was
stopped at generation 95 with **best clean 4.39999 s** and an archive of
**569 routes / 130 clean**; the speed bin went from 0 to roughly 88–100%, but
the search plateaued above the 3.41 s acceptance gate.

0082 tried the 2-frame 4.40 s expert (32 train routes, 28/32 replayed, 4,191
expert transitions) with a 2-frame anchor (median 3.25 s, 409 timeouts). Its
deterministic 1.58M eval was **1-5 0/1 (death), 1-4 timeout, 1-1 2.9333 s**, so
the control-rate change was rejected. 0085 retried at **4 frames** with all 16
clean 4-frame routes (16/16 replay, **1,510 expert transitions**, 4.68 s cluster
included), the proven 1.53M 4-frame anchor, and the 1.555M resume. Its final
1.655M deterministic eval (repeated twice) is **1-5 3.1667 s** (finish,
efficiency 1.0), **1-3 3.1667 s**, **1-1 1.5667 s**, but **1-4 fail** (death +
timeout) and **1-2 5.3500 s**. The 1.555M old chain model still holds 1-1..1-4
(0079: 1.5667 / 3.1500 / 3.2667 / 4.4667), so the practical chain is two models:
0085 final for 1-5, 1.555M for 1-1..1-4. That passes every 0.2 s gate (1-5
+0.057 s). 0088 then tried a single-model retention fine-tune from 1.655M with
2 x 1-5 + 2 x 1-4 + 2 x 1-3 + 2 x 1-2 and a fresh 1.555M all-level anchor
(5,105 episodes, median 3.00 s, best 1.5667 s, 6 timeouts). Its final checkpoint
gave **1-5 3.5000 s** (0.09 s over the 3.41 s outer gate), **1-4 timeout**,
1-3 3.2000 s, 1-2 3.0167 s, and 1-1 1.5667 s. The fine-tune fixed 1-2 and 1-3
but did not recover 1-4 and pushed 1-5 outside the gate, so the single-model
attempt is rejected. The selected deliverable is the two-model chain. A future
single-model attempt should freeze the 1-5 head or give 1-4 its own expert
buffer rather than repeating the same fine-tune. See
[`../experiments/0081_neat_speed_1_5_from_0080/README.md`](../experiments/0081_neat_speed_1_5_from_0080/README.md),
[`../experiments/0085_ddqn_expert_replay_1_5_4frame/README.md`](../experiments/0085_ddqn_expert_replay_1_5_4frame/README.md),
and
[`../experiments/0088_ddqn_retention_1_5_4frame/README.md`](../experiments/0088_ddqn_retention_1_5_4frame/README.md).

**1-6 any-start expert (0090–0100, 2026-10-29; 32/32 feasible starts, canonical 4.60 s, speed gate open).**
0090 seeded the **4-frame** 1-5 population on 1-6 and found a 4.5500 s best / 25
clean routes. 0091 trained the 1-6 DDQN from the selected 1-5 model. Its best
checkpoint is **1.705M at 4.5667 s** (`runs/ddqn_expert_replay_1_6_lean_stick/dqn/checkpoints/ddqn_1705000_steps.zip`;
1.68M 4.6833; 1.73M 8.60; 1.755M 8.3167; final 7.25). That model is
canonical-only: on the full 32-state pool it finishes only **11/32** (20 deaths,
1 timeout, median 6.1167 s). It lost 1-5/1-4 and is not a chain member.

0093 ran the 1-6 speed search for 30 generations (602 s): best **4.5000 s**,
archive **376 routes / 263 clean**, still +0.22 over the 4.28 gate. Expanding
all 263 clean routes against the exported level analysis gives **32 unique
grounded state starts** (cell/facing/powerup), including the true holdout
`[22,16]` facing `-1`. The old 0090 pool had only 11 train / 6 holdout states
because it used 22 routes.

The 0095 fast replayed routes are **not reproducible** in isolation. A second
replay gave 4.7167 s for the 3.1833 s route, 5.0500 s for the 3.9833 s route,
and a death at 2.7167 s for the 4.0667 s route. The 4.0667 s "clean route" is
therefore not usable teacher data. 0097 rebuilt the expert buffer from the
**85 stable routes / 7,365 transitions** whose replay is no more than 0.25 s
faster than the source, then trained 8 x 1-6 from the 1.705M checkpoint on
canonical + 31 state starts (50/50 weight), with no old-level anchor.

The 0097 sweep:

```text
1.7425M: canonical 4.5500; 27/32 states
1.7550M: canonical 4.6500; 31/32 states (1 timeout)
1.7675M: canonical 4.6500; 32/32 states (0 deaths/timeouts)
1.7925M: canonical 4.5500; 31/32 states (one death)
1.8050M: canonical 4.5500; 31/32 states (one death)
```

0099 repaired the single remaining failure (`[21,16]` facing `-1`) with a
targeted suffix expert buffer and selected **1.815M**:

```text
runs/ddqn_1_6_start_repair/dqn/checkpoints/ddqn_1814952_steps.zip
canonical:   1/1, 4.6000 s, 0 deaths, 0 timeouts
32 states:   32/32, 0 deaths, 0 timeouts, median 6.0417 s, best 4.6667 s
spawn_ok:    32/32
repeat:      same 4.6000 / 32/32
```

This is the first 1-6 expert that covers every NEAT-discovered feasible start.
It is still **0.32 s over the 4.28 s gate** and 0.62 s over the user's 3.98 s;
the stable NEAT route is 4.50 s, so the speed gap is now a route/search problem,
not only a start-coverage problem. Keep the 1.705M 4.5667 model as the
canonical reference and the 1.815M model as the any-start teacher. The next
speed options are a proper 2-frame attempt or a faster clean NEAT route; do not
re-use the non-reproducible 3.3/4.07 replays as deterministic targets. See
[`../audit/2026-10-29-1-6-ddqn-investigation.md`](../audit/2026-10-29-1-6-ddqn-investigation.md),
[`../audit/2026-10-29-run-reliability.md`](../audit/2026-10-29-run-reliability.md),
and
[`../experiments/0097_ddqn_1_6_feasible_starts/README.md`](../experiments/0097_ddqn_1_6_feasible_starts/README.md).

**Evaluation/action-semantics fix (2026-10-29).** The eval factory did not pass
`reassert_held_jump` to Godot, while training did. That made the first 0058 1M
eval report `2.7000 s` instead of `1.5667 s`. The Python side now defaults the
flag from the observation encoding (`legacy_stick`/`lean_stick` → true), uses
one `Run.reassert_held_jump` property across training/eval/preview, and fails
fast if Godot reports a different value. A Godot export TODO is recorded in
`known-issues.md`: exported held-stick models must set the same action
semantics. Previous DDQN evals that used `reassert_held_jump` — notably the 0053
1-4 probe — still need re-running before their results are treated as final.

**1-4 DDQN lean_stick 191 probe (2026-10-29; 0053 stopped at 100k).** Experiment 0035
transferred the protected 1-3 population (checkpoint 76) to 1-4. The zero-shot
pass produced 1/384 finishes at 10.5333 s; the 20-generation evolution reached
only 6/384 finishes with a 10.5333 s best and **0 clean routes**. A fresh
high-mutation run (0036) then produced **0/384** finishes and no routes. The
seeded run (0037) found **40/384 finishes and 5.9167 s** at generation 1, but its
live population collapsed to 2/384 because it had no rebalancing, and it was
stopped at generation 25. The 1-4 **STICK** ceiling-stick mechanic exposed a
training-contract mismatch: the legacy observation had no `on_ceiling` state and
the training adapter could not reassert a held jump after `JumpState.exit()`.
The contract is now versioned: `legacy_stick` appends `on_ceiling`,
`jump_held`, and `stick_active`; new runs opt into `reassert_held_jump` while
existing 58-input agents keep the frozen legacy contract. Experiment 0038
reached the clean target in **generation 1**: **173/384 finishes, 5.7667 s,
10 clean routes**. Filtering produced **5 train / 5 holdout routes** and a
**8-state train / 3-state holdout** pool; all non-canonical starts were reset-
smoked.

Experiment 0039 (protected DDQN, 4×1-4 + 2×1-3 + 1×1-2 + 1×1-1) completed at
2.525M. It did **not** learn 1-4: final deterministic 1-4 was **0/1 canonical,
1/9 train, 1/3 holdout** at 6.7833 s, identical to the 2.35M intermediate. No
1-4 checkpoint was selected. Old-level retention was mostly intact: 1-1
1/1/12/12/8/8 at 1.6167 s, 1-2 1/1/30/31/19/20 at 3.0333 s, 1-3
1/1/45/45/23/23 at 3.2333 s.

The 0038 **post-rebalance** checkpoint retained only **23 archived finishers**,
not the 173 live finishers; the archive stores one genome per route signature.
The first fix, 0044, classified live finishers from the just-finished generation
but still lost most of them because `Population.run` reproduces before
rebalancing: generation 1 reported 173 live finishers, but only 26 survived into
the rebalanced population. 0044 was stopped as a diagnostic.

0045 then merged evaluated finisher genomes back after reproduction. It reached
**380/384** at generations 6 and 8, but its first rebalance still selected only
**123** finishers: the **50 finishers slower than 10 s** were classified as
exploration and dropped. It was stopped as a second diagnostic.

Experiment **0046** used `is_finisher` as a separate property, preserved every
evaluated finisher including >10 s routes, and rebalanced every 2 generations.
It completed 30 generations with **380/384** finishes at generation 30, peak
**383/384**, best **4.9833 s**, and **75 clean routes**. An independent
checkpoint-30 evaluation finished **382/384**, best **4.9833 s**. The final
rebalance selected **384 finishers**.

Filtering the 75 clean routes produced **27 train / 6 holdout routes** and an
**11-state train / 2-state holdout** 1-4 pool. The combined multilevel train pool
has **99 starts** (11 1-4 + 88 old-level starts).

Experiment **0048** used the 0046 pool but `lean` 188, which has none of the
hold signals. At 2.525M it was **0/1 canonical, 1/11 train, 0/2 holdout**. It
was stopped at ~2.626M as the 188/new-pool baseline.

Experiment **0053** is the versioned **`lean_stick` 191** branch: `lean 188 +
on_ceiling + jump_held + stick_active`. The Python pipeline maps 191 values to
**957 model inputs**; the selected 2.225M 188 model was expanded by copying the
first 954 weights and zero-initializing the three new columns. The anchor buffer
was zero-padded to 957.

The 100k probe was stopped for the decision point:

```text
1-4 canonical: 0/1 (death)
1-4 train:     2/11 at 6.7833 s
1-4 holdout:   0/2
1-1:           1/1 1.6167; 12/12 3.3667; 8/8 3.9083
1-2:           1/1 4.3333; 31/31 5.0833; 20/20 4.8500
1-3:           1/1 6.5000; 41/45 6.6333; 19/23 6.3167
```

1-4 improved only slightly (1/11 -> 2/11) and 1-3 regressed, so the checkpoint
is not promotable. Do not retrain the whole backbone yet; the next decision is
whether to continue 191 with more old-level protection or freeze part of the
188 backbone. The 1-5 transfer (0047) was a diagnostic (384/384, 9 clean at
4.6833 s); 0080/0081 now extend it into a speed-targeted route search. See the
[level-transfer playbook](audit/2026-10-29-level-transfer-playbook.md) and
the [reward/fitness audit](audit/2026-10-29-reward-fitness-audit.md).

**Protected DDQN selected 2.225M (2026-10-29).** Experiment 0031 ran the
controlled 300k continuation from 2.15M with the state/Q-gap anchor. The selected
checkpoint is
`runs/ddqn_protected_1_1_1_2_1_3/dqn/checkpoints/ddqn_2225000_steps.zip`:

```text
1-1: 1.6167 canonical; 12/12 train 3.2667; 8/8 holdout 3.7917
1-2: 2.9667 canonical; 31/31 train 4.3500; 20/20 holdout 3.1333
1-3: 3.2500 canonical; 45/45 train 5.2667; 23/23 holdout 5.3333
```

The anchor fixed the 1-2 canonical regression (`3.8667 -> 2.9667 s`) without
losing 1-1 or 1-3 coverage. 1-3 is 0.15 s slower canonically than the old 2.30M
control, so this is a retention win rather than a 1-3 speed win. The next speed
attempt should reduce the anchor fraction to 10–15% or use a 1-3 high-gap anchor.
See
[`../experiments/0031_ddqn_protected_1_1_1_2_1_3/README.md`](../experiments/0031_ddqn_protected_1_1_1_2_1_3/README.md).

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
0.7). Filtering the clean set yielded 46 routes (35 train / 11 holdout) and a
pool with **44 train entries (43 state + canonical) / 26 holdout states**.

**Rebalanced 1-3 continuation (2026-10-29, complete; mixed).** Experiment 0023
resumed the rebalanced 384-genome checkpoint 66 for 10 generations on 1-3 using
8 processes × speedup 12 on AC. It finished in **257.8 s** with **238/384
(62.0%)** final finishes, up from 220/384 (57.3%) in the source final generation.
The archive improved to **364 unique routes / 60 clean routes / 54 clusters at
Jaccard 0.5**, with 154 routes and 25 clean routes new versus 0022. However, the
**live final population lost its speed tail**:

```text
source final (70):  5 speed / 4 median / 207 slow / 168 exploration
0023 final (76):    0 speed / 7 median / 224 slow / 153 exploration
```

So one rebalance preserves/expands the archive but does not preserve the live
deployment distribution. Do not use the one-shot final population as the 1-4
seed. Rebalance every 2–3 generations or protect the speed/median genomes during
reproduction. The interval checkpointer missed generation 76; the trainer now
always saves the final checkpoint. The 4-generation replay from checkpoint 72
saved a valid checkpoint 76 but is a separate trajectory (219 finishes).

**Checkpoint rebalancer (2026-10-29, implemented; first test).** Added
`src/plaiground/algorithms/genetic/neat/portfolio.py` and
`tools/rebalance_neat.py`. It loads a checkpoint, adds archived route genomes to
the candidate pool, bins genomes by finish time, selects route-cluster diversity,
and writes a rebalanced checkpoint. On checkpoint 66 + the 0022 archive it found
678 candidates, 213 finishers, and 46 clusters. The selected 384-genome
population keeps all 213 finishers (3 speed, 56 median, 154 slow) and fills 171
exploration slots; there are not enough fast finishers yet to fill the target
portfolio quotas. The first output at
`runs/neat_explore_1_3_long/rebalanced/rebalanced-66` had a low innovation
tracker and produced one collision warning during 0023. The fixed rebalancer
advances the tracker past the highest archived innovation and writes
`runs/neat_explore_1_3_long/rebalanced_fixed/rebalanced-66` (tracker 5691). Use
the fixed checkpoint for future seeds.

**NEAT parallelism benchmark (2026-10-29, complete; 8×12 recommended).**
Added `tools/benchmark_neat.py`, a reusable hardware-aware sweep. On the
Apple M5 laptop, the battery sweep favored **8 processes × speedup 12** at
**42.11 s/generation**. Plugged into AC with Low Power Mode off, the same combo
improved to **28.81 s/generation** over 3 generations. A 12×12 rerun on AC was
**40.53 s/generation**, confirming that 12 processes oversubscribe the 10-core
CPU. Recommendation for this laptop: **8 processes, speedup 12** while on AC.
The battery results were 46.59 s for 8×8 and 42.11 s for 8×12; 12×8 was 59.44 s.
Re-run the benchmark on the new PC before adopting a default there.

**Frequent-rebalance validation (2026-10-29, complete; hypothesis supported).**
Experiment 0024 started from `rebalanced_fixed/rebalanced-66`, merged the 0022
and 0023 archives (458 routes/genomes), and rebalanced the live population every
2 generations. It completed the resumed run in **230.0 s** and produced a
strong 1-4 seed:

```text
final generation:   384 / 384 finishes (100%)
best time:          4.2333 s
archive best clean: 4.7667 s
archive:            655 routes / 105 clean / 60 clusters @ 0.5
final bins:         3 speed / 78 median / 227 slow / 76 exploration
protected seed:     311 deployable genomes
```

This is a decisive improvement over the one-shot 0023 final (238/384 finishes,
0 speed / 7 median / 224 slow / 153 exploration, 364 routes, 60 clean, 46
clusters). Frequent rebalancing preserved the live speed/median tail and roughly
doubled the clean-route archive. The final checkpoint
`runs/neat_frequent_rebalance_1_3/neat-checkpoint-76` is the current 1-4 seed;
`rebalance-76.json` contains the 311-genome deployment portfolio. A separate
evaluation of that checkpoint finished **384/384**, with 3 speed / 78 median /
228 slow / 75 >10 s, so the seed is validated rather than only archived. The
runner now repairs NEAT's occasional `pop_size - 1` shortfall after
reproduction.

**1-4 NEAT attempts (2026-10-29; 0038 clean pool complete; 0039 DDQN negative; 0044/0045 diagnostics; 0046 seed validated; 0053 100k DDQN probe stopped).**
Experiment 0035 used `runs/neat_frequent_rebalance_1_3/neat-checkpoint-76` as
the 1-4 seed. Zero-shot: **1/384** finishes, 10.5333 s. After 20 generations:
**6/384** finishes, best 10.5333 s, 4 routes, **0 clean routes**. Experiment 0036
then ran the fresh high-mutation 384-genome search (0021 settings) for 30
generations: **0/384** finishes, 0 routes, 0 clean routes, best genome timed out.
Experiment 0037 seeded from the four rough 0035 finishers; generation 1 reached
**40/384** finishes and **5.9167 s**, but without rebalancing the live population
collapsed to 2/384 by generation 24, and the run was stopped at generation 25.

The contract fix is now implemented and scoped:
`legacy_stick` is a versioned 61-value extension that keeps the first 58 values
byte-for-byte and appends `on_ceiling`, `jump_held`, and `stick_active`.
`reassert_held_jump` is an opt-in training action mode; existing 58-input
policies keep edge-triggered semantics and are not re-encoded. Experiment 0038
seeded the top four 0037 routes into the 61-input population, enabled held-input
reassertion, and reached the clean target in **generation 1**:

```text
0038: 173/384 finishes, 5.7667 s, 10 clean routes, 4 clusters
      filter: 5 train / 5 holdout routes
      pool:   8 train / 3 holdout states; all 11 starts reset-smoked
```

The fastest 1-4 route (5.7667 s) is holdout, not train. Filtering used Jaccard
0.5, max 8 per cluster, 25% holdout. Experiment 0039 resumed the selected
2.225M DDQN for 300k steps on 4×1-4 + 2×1-3 + 1×1-2 + 1×1-1 with protected
replay and held-input action semantics. It completed at 2.525M with **no 1-4
checkpoint selected**; 1-4 stayed at 0/1 canonical, 1/9 train, and 1/3 holdout.
Experiment 0046 completed the 1-4 seed rebuild with finisher-preserving
rebalancing: **382/384** evaluated, best **4.9833 s**, 75 clean routes.
Experiments 0048 and 0053 are the DDQN follow-ups; 0048 (188) was negative and
0053 (lean_stick 191) was stopped at the 100k decision point. See the
[level-transfer playbook](audit/2026-10-29-level-transfer-playbook.md) and records:
[0035](../experiments/0035_neat_transfer_1_4_from_1_3/README.md),
[0036](../experiments/0036_neat_explore_1_4_high_mutation/README.md),
[0037](../experiments/0037_neat_seeded_1_4_from_0035/README.md),
[0038](../experiments/0038_neat_seeded_1_4_stick/README.md),
[0039](../experiments/0039_ddqn_protected_1_1_1_2_1_3_1_4/README.md),
[0044](../experiments/0044_neat_full_finisher_1_4/config.toml),
[0045](../experiments/0045_neat_full_finisher_1_4_retry/config.toml),
[0046](../experiments/0046_neat_full_population_1_4/config.toml),
[0048](../experiments/0048_ddqn_protected_1_4_full_pool/config.toml),
[0053](../experiments/0053_ddqn_protected_1_4_lean_stick/config.toml).

**Two-tier discovery + deployment portfolio (design).** Do not force a Gaussian
on the live population. Separate:

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

**Protected replay for DDQN (2026-10-29, complete; selected 2.225M).** Added
`src/plaiground/algorithms/reinforcement/replay.py` with a
`RareReplayBuffer` and a two-part `ProtectedReplayBuffer`: a 400k-transition
online FIFO plus a fixed, non-evicting 200k anchor from the 2.15M model, sampled
25% anchor / 75% online. The anchor records **current-state buckets**
(`level × current cell × facing × run-time bin × powerups`), return-to-go, and
`Q_old(s,a)`. Positive relative gaps `max(0, return - Q_old)` above the 0.90
quantile receive profit weight; all states retain rarity weight. The 200k
anchor is complete: 3,675 state buckets, 141,249 positive-gap transitions,
threshold 0.12649. The 300k continuation selected **2.225M** over the old 0025
2.30M control: 1-2 canonical recovered from 3.8667 to 2.9667 s, 1-1 stayed
1.6167/12/12/8/8, and 1-3 stayed fully covered at 3.2500/45/45/23/23. The final
2.45M checkpoint was not selected; 2.30/2.325/2.35 failed the 1-1 holdout gate,
and 2.375 lost one 1-2 train start. Evaluation configs 0032-0034 cover 1-3,
1-1, and 1-2. The fixed anchor distribution is cached after the first minibatch;
this restored throughput from about 160 fps to about 720 fps. Old start-bucketed
and state-only buffers are superseded.

**DDQN 1-3 training (2026-10-29, complete; selected 2.30M).** Experiment 0025
resumed the selected 2.15M 1-1/1-2 Double-DQN checkpoint for 300k additional
steps with the 4+2+2 mix. The combined train pool had 88 level-tagged starts:
12 × 1-1, 31 × 1-2, and 45 × 1-3 (44 state + canonical), built from the 0024
archive. The final 2.45M checkpoint failed the 1-1 holdout gate (4.0083 s vs the
3.89 s limit), so **2.30M is selected**:

```text
1-1: 1.6167 canonical; 12/12 train 3.2750; 8/8 holdout 3.8000
1-2: 3.8667 canonical; 31/31 train 4.0000; 20/20 holdout 3.1667
1-3: 3.1000 canonical; 45/45 train 5.2000; 23/23 holdout 5.4500
```

The 2.15M baseline solved only 40/45 train and 15/23 holdout on 1-3, with a
5.1333 s canonical. 2.30M makes 1-3 fully solvable. The remaining tradeoff is
the 1-2 canonical regression from 2.9667 to 3.8667 s; 1-2 train/holdout medians
are close to the prior model. The selected checkpoint is
`runs/dqn_multilevel_1_1_1_2_1_3_ddqn_300k/dqn/checkpoints/ddqn_2300000_steps.zip`.

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
| DDQN single-route | 1-9 | **2.9500 s** canonical; 8/8 checkpoints; retention 3/9 finish, 2/9 pass gate | `runs/ddqn_1_9_single_route/` |
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
