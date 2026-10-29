# 1-6 DDQN investigation — 2026-10-29

Status: **any-start 1-6 expert selected (0099 1.815M: 32/32 starts, canonical
4.6000 s); 4.28 s speed gate still open.** The 3.3333 s stochastic mode and one
2-frame attempt remain the high-value speed levers. The 4.0667 s replay was
later found not reproducible in isolation and is not teacher data.

## Question

Why does the 1-6 DDQN stop at **4.5667 s** when the 1-5 DDQN reaches
**3.1667 s**, and is the 512×512 MLP too small for the level?

## Reference numbers

| Policy / target | 1-5 | 1-6 |
|---|---:|---:|
| Level par | 2.9088 s | **3.7056 s** |
| User time | 3.11 s | **3.98 s** |
| User + 0.30 gate | 3.41 s | **4.28 s** |
| Shipped 58-value student | 2.967 s | 5.050 s |
| Our selected DDQN | **3.1667 s** | **4.6000 s** any-start (1.815M) |
| Fastest deterministic 1-6 canonical | — | **4.5667 s** (1.705M, 11/32) |
| Fastest recorded stochastic finish | — | **3.0667 s** (0094 anchor) |
| Fastest stable clean expert route | — | **4.5000 s** |
| 1-6 NEAT best | — | 4.5000 s |

The 1-6 par is 0.797 s longer than 1-5 par. The 1-6 user is +0.274 over par;
1-5 user is +0.201. The 1-5 AI is +0.258 over par; the 1-6 AI is +0.861 over
par. The 1-6 policy is the outlier, not the level par.

The 1-6 **3.0667 s** row is a real finished episode (positive finish reward),
not a death. It is stochastic, not a deterministic policy, and is only one of
155 finished anchor episodes under 4.28 s. The 0099 any-start model is 4.6000 s;
the 0091 1.705M model is the fastest canonical reference at 4.5667 s but covers
only 11/32 starts. The 4.50 s NEAT route is the fastest stable clean route; the
4.0667 s replay is not reproducible in isolation.

## Experiments

| Exp | What | Result |
|---|---|---|
| 0090 | 4-frame 1-6 NEAT from the 1-5 checkpoint-50 | gen 51: 31 routes / 25 clean, best **4.5500 s**; stopped at clean target |
| 0091 | 1-6 DDQN from the 1-5 model, 4×1-6 + old, old anchor | best **4.5667 s** at 1.705M; 1.68M 4.6833; 1.73M 8.60; 1.755M 8.3167; final 7.25; lost 1-5/1-4 |
| 0093 | 1-6 NEAT speed search from 0090 | best **4.5000 s** (gen 68); 376 routes / 263 clean in 602 s; still +0.22 over gate |
| 0094 | 1-6-only anchor from 1.68M | 200k rare; 3,224 episodes; median **4.6833**; 1,412 deaths / 115 timeouts; best **stochastic finish 3.0667**, 155 finishes under 4.28 |
| 0095 | 1-6-only retrain, 8×1-6 + 1-6 anchor + 4.07 replay | final **6.3833 s**; checkpoints timeout / 8.50 / death / 6.38; not selected; the 4.07 replay is not stable in isolation |
| 0096 | 1-6-only eval config | used for 0091/0095 checkpoints |
| 0097 | 1-6 any-start, 8×1-6, 32 state starts, 85 stable expert routes | first 32/32 at **1.7675M** (4.65); 1.805M 4.55 but 31/32 |
| 0098 | 32-state eval | baseline 1.705M **11/32**; 0097 sweep |
| 0099 | targeted suffix repair from 1.805M | selected **1.815M: 4.6000 canonical, 32/32, 0 deaths/timeouts** |
| 0100 | selected 1-6 eval | repeat 4.6000 / 32/32 |
| — | Expert replay of 0093 filtered train routes | 89/104 finished; 8,645 transitions; buffer 7,586; best **4.0667 s** (not standalone-reproducible) |
| — | 0093 filter (seed 0) | 133 clean / 33 clusters; 104 train / 29 holdout; 4.50 in train |

## The 4.07 s replay — not reproducible

The 0093 archive's 4.50 s cluster replayed at 4 frames produced a reported
**4.0667 s finished route** (source route time 6.2667 s) in the 0095 batch. A
second replay for 0097 did not reproduce it: the 4.0667 s route died at 2.7167 s
after 39 steps. The 3.1833/3.8667/3.9833 s results also changed or failed when
replayed in isolation. The route replay order carries RNG/world state between
routes.

```text
0095 replay    0097 batch replay   0097 standalone
3.1833 s       4.7167 s            —
3.8667 s       3.9833 s            5.0500 s
3.9833 s       5.0000 s            —
4.0667 s       death at 2.7167 s   —
```

The 4.07 s replay is a **diagnostic**, not a clean teacher route. The 0097
expert buffer uses only routes whose replay is at most 0.25 s faster than the
source: **85 routes / 7,365 transitions**. The fastest stable clean NEAT route
is **4.50 s**. The fastest recorded 1-6 finish is the stochastic **3.0667 s**
anchor episode; the 3.2/3.33 s stochastic modes are also real finishes, but
their action sequences were not saved as routes. The selected any-start
**deterministic policy** is the 0099 1.815M checkpoint at **4.6000 s**; the
1.705M canonical reference is 4.5667 s but covers only 11/32 starts.

## False fast times and real stochastic finishes

The 1-6 anchor's `best=2.4333` and the 2.6–2.8 s rows in the retrain monitor are
**deaths**, not finishes:

```text
row 27: finished=False  length=35  reward=-48.49  run_time=2.4333
2.6000: finished=False  reward=-45
2.6833: finished=False  reward=-45
2.7333: finished=False  reward=-45
2.8000: finished=False  reward=-44
```

The collector's old `_fmt_stats` took `min(run_time)` over all canonical
episodes without filtering on `finished`, so a death at 2.43 s was reported as
“best.” The collector now reports `best_finished_time`, `best_death_time`,
`canonical_finished`, and `canonical_deaths`, and computes medians/speed bins
from finished episodes only. The existing 1-6 anchor metadata still contains
the old `best=2.4333`; treat that field as the death value until the anchor is
re-collected.

The **3.0667 s**, 3.2 s, and 3.3333 s rows are not deaths. They are real
finished episodes from the 0094 anchor's 2% exploration policy:

```text
0094 1-6: 1,697 finishes / 3,224 episodes
  3.0–3.5 s:  153 finishes, 9 deaths
  4.5–5.0 s: 1,390 finishes, 3 deaths
  median finish: 4.6833 s
```

The fast modes are real but stochastic; the deterministic final of 0095 was
6.3833 s. The 3.3333 s mode appears 96 times, the 3.2 s mode 19 times, and the
3.0667 s mode once. This is the central reliability gap: the behavior
distribution contains a <4.0 s mode, but the greedy checkpoint does not select
it.

## Why 1-6 is slow

### Not network capacity

```text
shipped 58-value student:  58 -> 256 -> 256 -> 2, 81,410 params, 1-6 5.050 s
our DDQN:                  957 -> 512 -> 512 -> 2, ~754k params, 1-6 4.5667 s
```

The 9× smaller student clears 1-6. The 754k DDQN already beats it by 0.48 s.
A wider MLP is not the first fix; if capacity is tested, a frame stack or
recurrent model for the dissolve timing is a better test than more width.

### The expert/route is the bottleneck

The 1-6 NEAT best is 4.50 s; the 0099 any-start DDQN reached 4.60 s from the
stable expert routes. The 4.07 s replay does **not** prove a reproducible faster
clean route; it only appeared after another route in the batch. The 0094 anchor
proves a 3.33 s mode exists under 2% exploration, but it was not turned into a
deterministic policy. The 1-5 precedent is the opposite: a 4.68 s expert became
a 3.1667 s policy. The difference is route discoverability and timing, not input
capacity.

### The training mix was not the fix

0095 removed the old levels and old anchor, used 8×1-6 and a 1-6-only anchor,
and started from the best 1.6M checkpoint. It still ended at 6.38 s. The
1-6-only mix is not sufficient on its own; the 4-frame path is the limit.

### Timing / control rate

1-6 is `DISSOLVE` with par 3.7056; 1-5 is `BUNNY_HOP` with par 2.9088. The
1-5 2-frame search found 4.40 s versus 4.68 s at 4 frames. A 2-frame 1-6
search/DDQN is untested and is the natural next lever. The 1-6 model can be a
per-level model, so 2-frame old-level retention does not matter.

### Reward

The NEAT search uses `--time-fitness-weight 20`; the DDQN uses the game reward
plus expert TD/margin and has no explicit time penalty. That can lock the policy
to the expert route. A time-shaped DDQN reward is a possible later experiment,
not the first fix.

## 0097 replay audit — fast routes are not reproducible

The 0095 replay reported 3.1833 / 3.8667 / 3.9833 / 4.0667 s routes. A second
replay for 0097 showed that those fast results depend on route order or carried
RNG/world state:

```text
0095 replay    0097 batch replay   0097 standalone
3.1833 s       4.7167 s            —
3.8667 s       3.9833 s            5.0500 s
3.9833 s       5.0000 s            —
4.0667 s       death at 2.7167 s   —
```

The 4.0667 s route is **not** an independently reproducible clean route. The
0097 32-state expert therefore uses only routes whose replay is no more than
0.25 s faster than the source route: 85 routes / 7,365 transitions. Treat the
3.07/3.2/3.33 stochastic modes and the 4.07 replay as diagnostics, not as
teacher actions, until a route replay is stable in isolation.

## Current selection

```text
1-1..1-5:  the two-model chain (1.555M old levels + 0085 final 1-5)
1-6:       1.815M any-start expert (0099); 32/32 starts, canonical 4.6000 s
1-6 canon:  1.705M reference (0091); 4.5667 s, only 11/32 starts
4.07:      NOT a reproducible clean route; excluded from teacher data
3.07/3.2:  stochastic modes in the 0094 anchor; not deterministic checkpoints
```

The 1.815M checkpoint is
`runs/ddqn_1_6_start_repair/dqn/checkpoints/ddqn_1814952_steps.zip`. Its
deterministic evaluation was repeated with the same result: canonical
4.6000 s, 32/32 state starts, 0 deaths, 0 timeouts, spawn_ok 32/32, median
6.0417 s. It is the 1-6 teacher for the shared student.

The selected expert still misses the **4.28 s** gate by 0.32 s and the user's
3.98 s by 0.62 s. The stable NEAT route is 4.50 s; reaching the gate needs a
faster clean route or a 2-frame attempt, not another replay of the unstable
3.3/4.07 modes.

The 4.5667 s 1-6 model is clean and stable on the canonical trace:

```text
1/1 finish, 0 deaths, 0 timeouts, efficiency 1.0
repeated at speedup 12 and 16
```

It is **canonical-only** in practice. On the full 32-state pool it finishes
**11/32**, dies 20, times out once, and has median **6.1167 s** (best 4.5667 s).
The earlier 13/18 diagnostic used a smaller 18-entry pool. This is expected:
0091 trained on a canonical-only 1-6 pool. The 4.5667 result is a canonical
reference, not an any-start result.

The 0099 1.815M checkpoint is the selected any-start expert: **32/32 finishes,
0 deaths/timeouts, median 6.0417 s**, canonical **4.6000 s**. It is 0.32 s over
the 4.28 gate and 0.62 s over the user's 3.98 s. The 1.705M checkpoint was
missed by the original checkpoint sweep; the first selected 1.68M checkpoint is
4.6833 s.

## Checkpoint and pool gaps

The first selection looked at 1.68M, 1.73M, and the final only. A full sweep of
the four 0091 checkpoints gives:

```text
1.680M  4.6833
1.705M  4.5667   <- selected
1.730M  8.6000
1.755M  8.3167
final   7.2500
```

A post-training sweep over every saved checkpoint is mandatory; the final
`dqn_final.zip` is not the selected model.

The 1-6 start pool was canonical-only because `runs/level_analysis/1-6.json`
did not exist. It is now exported. Expanding all **263 clean routes** gives
**32 unique state starts**, not the old 11: the first pool used only the 0090
22-route set. `pool_state_50/` is the 0.5/0.5 training pool;
`pool_state_70/` is the 0.7/0.3 pool used by 0099;
`pool_state_50/eval_all_start_pool.json` is the 32-state coverage gate. The
selected 1.815M model finishes all 32; the 1.705M reference finishes 11.

## Next options

1. **Use the 1.815M any-start expert as the 1-6 teacher** (32/32 coverage,
   4.6000 canonical, 0 deaths/timeouts); keep 1.705M as the canonical reference.
2. **One 2-frame 1-6 attempt**: speed search from the current 1-6 population at
   `action_repeat=2`, then a 1-6-only 2-frame DDQN. This is the main untested
   lever with a plausible path below 4.28.
3. **Find a faster stable clean route.** The stable 4.50 s route is 0.22 s
   above the gate; a NEAT speed search or a clean 4.0-4.2 route is needed. Do
   not re-use the order-dependent 3.3/4.07 replays.
4. **Capacity/temporal test** only after 2-frame: frame stack or recurrent
   model; not a simple 1024×1024 MLP.

## Artifacts

```text
runs/neat_transfer_1_6_from_1_5/          # 0090 NEAT + archive
runs/neat_speed_1_6_from_0090/            # 0093 speed search + filtered/
runs/ddqn_expert_replay_1_6_lean_stick/   # 0091 DDQN; best 1.705M = 4.5667
runs/ddqn_expert_replay_1_6_lean_stick/eval/state_pool_1705/ # 13/18 state-start diagnostics
runs/ddqn_anchor_1_6_1680/                # 0094 1-6-only 200k anchor; 3.0667/3.2/3.33 stochastic
runs/ddqn_1_6_retrain/                    # 0095 8x1-6 retrain; final 6.38
runs/ddqn_1_6_retrain/expert_1_6/         # 4.07 s batch replay; not standalone-reproducible
runs/neat_transfer_1_6_from_1_5/pool_state # old 11 train / 6 holdout state starts
runs/neat_speed_1_6_from_0090/pool_state_50/  # 31 train states + canonical; 0.5/0.5
runs/neat_speed_1_6_from_0090/pool_state_70/  # 31 train states + canonical; 0.7/0.3
runs/neat_speed_1_6_from_0090/pool_state_50/eval_all_start_pool.json # 32-state coverage gate
runs/ddqn_1_6_feasible_starts/             # 0097 100k any-start training
runs/ddqn_1_6_start_repair/                # 0099 selected 1.815M any-start expert
experiments/0090..0100                     # manifests
```

See also: [`../status.md`](../status.md), [`../known-issues.md`](../known-issues.md).
