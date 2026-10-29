# Decision — per-level NEAT experts and one shared student

**Status:** accepted  
**Date:** 2026-10-29  
**Scope:** the Dragon Jump campaign AI and the level-testing use case. The
architecture covers all campaign levels; the current active implementation set
is 1-1 .. 1-8, with 1-9 .. 1-16 added by the same procedure.

## Decision

1. **NEAT is the feasibility search.** For each level, NEAT discovers clean
   routes and route checkpoints. Those checkpoints define the feasible route
   starts for that level. We do not assume every standable cell is feasible.

2. **Build one expert per level.** Each expert is trained on one level only:
   canonical plus all resolvable NEAT-discovered feasible starts, with a
   held-out set of NEAT starts for evaluation. The goal is that the expert can
   solve the level from every feasible start that NEAT has demonstrated, not
   only from the canonical start. The target covers every campaign level; the
   active implementation order is 1-1 .. 1-8, then 1-9 .. 1-16.

3. **Do not mix a new level with old-level training.** A new-level expert run
   contains no old-level environments, no old-level anchor, and no old-level
   gates. Old-level retention is a separate decision; it is not part of
   learning the new level.

4. **The per-level experts are teachers, not the final product.** They may be
   level-specific and may lose other levels. Their job is to produce reliable
   behavior, data, and target policies for the hard levels.

5. **The final product is one shared student, target A:**
   - one checkpoint;
   - one shared policy;
   - no level ID in the observation;
   - no per-level heads at inference;
   - the level is inferred from the normal observation.

   Level labels may be used to balance training data and to report results,
   but they are not model inputs.

6. **The student is trained across all levels deliberately.** This is not a
   violation of the “one level per expert run” rule: the expert runs are
   single-level; the student phase is explicitly multi-level.

7. **The student must be evaluated for generalization.** At minimum, evaluate
   seen levels and held-out/unseen levels. The exact unseen-level gate is still
   open, but “one agent that generalises” is not established by canonical
   1-6 alone.

## What “feasible position” means here

A feasible position is a NEAT-discovered route checkpoint/start for the level,
including the supported state snapshot fields:

```text
cell, facing, position, velocity, run state, run_time, powerup/world state
```

The game reset supports grounded state snapshots; arbitrary mid-air states and
unsupported powerup/dissolve states are not promised. NEAT is the oracle for
which starts are feasible.

The current route-based train/holdout split is the default. Where possible,
the holdout should be split by route cluster or cell/facing so the same
feasible start does not appear in both sides.

## Gates

Each per-level expert:

```text
canonical finish and time gate
finish coverage over held-out NEAT feasible starts
death/timeout rate
```

The existing 1-6 canonical gate remains 4.28 s (user 3.98 + 0.30).

The shared student A:

```text
one checkpoint, no level ID
old-level gates retained
1-6 canonical <= 4.28 s
held-out NEAT starts evaluated
unseen-level diagnostic/gate recorded
```

A canonical-only result does not satisfy the any-feasible-start requirement.
For 1-6, the 1.705M model is 4.5667 s canonical but only **11/32** on the full
state pool (the older 13/18 diagnostic used a smaller pool). It is a canonical
reference, not a finished any-start expert.

**Result update (2026-10-29):** the rebuilt 32-start pool and the 0097/0099
single-level runs produced a **1.815M any-start expert** at 32/32 finishes, 0
deaths/timeouts, median 6.0417 s, canonical 4.6000 s (repeat-stable). The 4.28 s
gate remains open; the stable NEAT route is 4.50 s. The old 4.0667 s replay was
not reproducible in isolation and is not teacher data. See
[`2026-10-29-1-6-ddqn-investigation.md`](2026-10-29-1-6-ddqn-investigation.md).

**1-7 result update (2026-10-29):** strict clean routes were not available:
the 1-6 transfer (0101) produced 240 routes / 0 clean, and a 30-generation
high-mutation cleanup (0102) produced 902 routes / 0 clean, best 7.0333 s
(revisit 0.107). The accepted 1-7 gate therefore uses **near-clean** routes
(`revisit_ratio <= 0.20`, `backtracks <= 2`), which give 10 grounded states. The
1-7-only DDQN (0103) selected **1.877432M**: canonical **5.2333 s**, near-clean
**10/10** with 0 deaths/timeouts (median 7.9583), and the 20-state all-success
diagnostic **18/20** (two timeouts on a high-revisit route). It passes the
provisional par + 0.30 s gate (**5.2728 s**) by **0.0395 s**. The final 1-7
target is still open, and strict-clean 1-7 routes remain an open data-quality
gap.

**1-8 result update (2026-10-30):** the 1-7 population transferred to 1-8
(DOUBLE_JUMP) and produced **26 strict-clean routes / 11 clusters**, best
**4.4167 s** (0105). The clean pool expands to 37 train / 14 holdout grounded
states, with 40 unique clean evaluation states; the 16 train routes replay
exactly and the stable expert buffer has 1,800 transitions. After the 0106
sweep and a short canonical/start repair sequence (0109-0118), the selected
**1.960392M** expert reaches **canonical 4.4667 s**, **40/40** clean states with
0 deaths/timeouts (median 9.8833), and **106/107** all-success states with one
timeout on the high-revisit route `36a6b1e1...` (revisit 0.226). It passes the
provisional par + 0.30 s gate (**4.5144 s**) by **0.0477 s**. The final 1-8
target is still open. This is the first level with a strict-clean any-start
expert; 1-9 follows the same NEAT transfer → start pool → single-level expert
procedure.

## Why this decision

The evidence from the recent 1-6 work:

- **0091** mixed 1-6 with 1-5/1-4/1-3. Its best checkpoint was 4.5667 s
  (1.705M), but attribution and old-level retention were muddy.
- **0095** was 1-6-only, but used a 104-route expert mixture and TD/margin
  losses; it ended at 6.3833 s. Being single-level was necessary, not
  sufficient.
- The 0094 anchor contains real stochastic 3.3333/3.2/3.0667 s finishes under
  2% exploration. They were not stable deterministic routes; the 0097 replay
  audit also showed that the 0095 3.18/3.98/4.07 fast replays depend on route
  order and are not standalone clean routes.
- The 1-6 NEAT archive has **263 clean routes**. Expanding them gives **32
  unique state starts**, not the old 11. The 0097/0099 runs trained on those
  starts and reached 32/32 coverage.
- The shipped 58-value student is already a single shared agent, but its 1-6
  time is 5.05 s. One agent is possible; it needs better teachers and data.

The old 1-5 two-model chain is a reliable fallback and a source of teachers.
It is not the final architecture under this decision.

## Consequences

1. 0097/0099 did this: the 1-6 expert now trains on all 32 NEAT feasible
   starts. The selected 1.815M checkpoint is 32/32 with 0 deaths/timeouts.
2. The fast 1-6 routes were replayed; the 4.0667 s / 3.98 s / 3.18 s results
   were not reproducible in isolation. Only stable routes are teacher data.
3. The 1-6 expert is evaluated on canonical plus all 32 NEAT starts; the
   canonical-only 1.705M result (4.5667 s, 11/32) is not the acceptance
   criterion. The selected 1.815M is 4.6000 s canonical and 32/32 any-start;
   the 4.28 s speed gate remains open.
4. The student phase is a separate multi-level distillation/DAgger effort.
   It must not be described as another single-level training run.
5. A per-level head or level-ID model is a fallback, not target A, and would
   require a new decision.
6. Action-repeat changes are still global: if a 2-frame attempt is made, the
   expert, anchor, and replay data must all be 2-frame.

## Open questions

- Exact coverage gate for future levels (1-6 is resolved at **32/32**, 1-7 at
  **10/10 near-clean**, and 1-8 at **40/40 strict clean**; a practical threshold
  still needs to be chosen per new level).
- Whether the student observation is the 191-value `lean_stick` encoding or the
  deployed 58-value encoding. Target A is about policy shape; deployment
  encoding may require a final distillation step.
- Which unseen level(s) form the generalization gate (1-9 is the next
  out-of-distribution candidate; 1-7/1-8 now have teachers).
- The implementation order after 1-8; the architecture is the same for every
  campaign level.

## Related records

- [`2026-10-29-1-6-ddqn-investigation.md`](2026-10-29-1-6-ddqn-investigation.md)
- [`2026-10-29-run-reliability.md`](2026-10-29-run-reliability.md)
- [`2026-10-30-1-8-expert.md`](2026-10-30-1-8-expert.md)
- [`../reference/ddqn-1-5-chain.md`](../reference/ddqn-1-5-chain.md)
- [`../status.md`](../status.md)
