# Now

**Updated:** 2026-10-30
**Repo:** branch `neat` @ `2e68348`; run `make brief` for live status.
**Game:** `../dragon-jump-remaster` branch `prepare-for-expo` @ `50dcc66`;
run `make brief` for live status.

> This is the current-state page. [`status.md`](status.md) is the chronological
> log and contains historical sections that may contradict this page.
> [`reference/level-ledger.md`](reference/level-ledger.md) is authoritative for
> the selected expert, time, coverage, and gate; if the two disagree, the
> ledger wins. [`index.md`](index.md) is the full catalogue.

## Current focus

The campaign distillation candidate is selected and **finishes 16/16 canonical
levels**. The selected model is a shared one-hot 957 student with no level ID or
per-level heads:

```text
runs/distill_campaign_student/students/cand_verified_onehot957_512_false_ddqn2_w11x2_s2/student.pt
957 -> 512 -> 512 -> 2, seed 2, level weight 1-11=2
finish 16/16; strict 12/16; sum 82.4833 s
```

The four strict misses are 1-3 (+1.9666), 1-8 (+0.0333), 1-12 (+0.0500), and
1-14 (+1.2500). 1-14 is evaluated with `reassert_held_jump=false`; the other
levels use true semantics. The 1-8 and 1-12 misses are one action-repeat frame.

The game runtime still only accepts legacy 58 or multi-hot dense JSON models,
so this checkpoint is **not yet a drop-in game artifact**. The bundled
`student_obs58_256.json` remains the production fallback (16/16, 78.583 s sum)
until a `lean_stick` one-hot runtime or multi-hot distillation pass lands. See
[`../experiments/0130_distill_campaign_student/SHIPPING_CANDIDATE.md`](../experiments/0130_distill_campaign_student/SHIPPING_CANDIDATE.md).

- The 1-1..1-4 chain is the selected `0073` 1.555M model, not four independent
  experts.
- 1-5 is the selected `0085` 1.655M model.
- 1-6 has an any-start expert (`0099` 1.815M, canonical **4.6000 s**) but the
  **4.28 s speed gate is still open**.
- 1-7 uses the near-clean fallback (`0103` 1.877432M, **5.2333 s**); no strict
  clean archive exists.
- 1-8 is a coverage expert (`0118` 1.960392M, **4.4667 s**, 40/40 clean).
- 1-9 is a single-route expert (`0122` 2.060360M, **2.9500 s**); retention is
  partial by user decision.
- 1-10 is a coverage expert (`0126` 2.122840M, **5.3667 s**, 22/25 eval
  states); the speed gate **3.3336 s is open**. The 4.7000 s targeted-repair
  checkpoint covers only 20/25, so it is a speed alternative, not selected.

## Level ledger

| Level | Selected | Canonical | Coverage | Status |
|---|---|---:|---|---|
| 1-1 | `0073` 1.555M chain | 1.5667 | 12/12 train, 8/8 holdout | selected chain member |
| 1-2 | `0073` 1.555M chain | 3.1500 | 31/31 train, 20/20 holdout | selected chain member |
| 1-3 | `0073` 1.555M chain | 3.2667 | 45/45 train, 23/23 holdout | selected chain member |
| 1-4 | `0073` 1.555M chain | 4.4667 | 11/11 train, 2/2 holdout | selected chain member |
| 1-5 | `0085` final 1.655M | 3.1667 | canonical finish; retained levels separate | selected 1-5 expert |
| 1-6 | `0099` 1.815M | 4.6000 | 32/32 feasible starts | speed gate `4.28 s` open |
| 1-7 | `0103` 1.877432M | 5.2333 | near-clean 10/10; all-success 18/20 | final target unset |
| 1-8 | `0118` 1.960392M | 4.4667 | strict-clean 40/40; all-success 106/107 | final target unset |
| 1-9 | `0122` 2.060360M | 2.9500 | canonical single route | selected; retention partial |
| 1-10 | `0126` 2.122840M | 5.3667 | 22/25 eval states | selected coverage expert; speed gate `3.3336 s` open |

## Next 3

1. **Decide the shipping path for the selected student.** Either add a
   `lean_stick` + fitted one-hot spec to the game `MLPPolicy`, or distill this
   candidate into a `multi_hot`/legacy student that the existing JSON runtime
   accepts. Do not overwrite the bundled 58-value model before that gate passes.
2. **Do not retrain for canonical coverage right now.** The current candidate is
   16/16 finish. The remaining work is timing: 1-3 and 1-14 are material; 1-8
   and 1-12 are one frame. Treat strict timing as the next quality pass.
3. **Keep the 7x7 encoder deferred.** It did not improve the canonical set
   (1-9/1-10/1-11 failures) and would require a new game-side convolution
   runtime. Revisit only if the flat model plateaus or strict 1-3 becomes the
   primary objective.

## Active traps

Read the matching section in [`known-issues.md`](known-issues.md) before
changing these areas. Resolved issues stay there for history.

- **Mid-route cell resets zero the run clock** (critical). Start time and
  reward must come from the source checkpoint, not a fresh reset.
- **One-shot NEAT rebalancing does not preserve the live distribution** (high).
- **DQN extraction is all-or-nothing** (high): a failed extraction writes no
  pool.
- **Spawn validation is incomplete** (high): check `spawn_ok` before training.
- **Clean-start promotion gates are incomplete** (high).
- **The holdout is checkpoint-level, not route-level** (high): cluster routes
  before splitting.
- **Changing `action_repeat` breaks the old-level anchor** (high).
- **Single-model 1-5 training damages 1-4 and late 1-2** (high).
- **1-10 fast-route and coverage states conflict** (high): the 4.70 s route
  covers 20/25; the 5.37 s coverage expert covers 22/25.
- **Checkpoint selection is manual and finals regress** (high).
- **Canonical eval is one deterministic trace** (medium): it is not a
  robustness measure.
- **NEAT fitness does not rank fast finishers first** (medium).

## Commands

| Command | What it answers |
|---|---|
| `make brief` | branch, game commit, running jobs, selected levels, recent runs |
| `make run RUN=<name>` | one run's checkpoints, eval summaries, and log tail |
| `make search q="..."` | capped text search; skips `runs/`, `data/`, binaries |
| `make context q="..."` | ranked code, docs, experiments, and runs |
| `make level-status LEVEL=1-9` | one level's experiments and data summaries |
| `make index` / `make index-check` | regenerate / verify the code map |

## Where to look

| Need | Read |
|---|---|
| Current state | this page |
| Selected expert and gate | [`reference/level-ledger.md`](reference/level-ledger.md) |
| Full new-level procedure | [`runbooks/new-level.md`](runbooks/new-level.md) |
| Architecture | [`architecture.md`](architecture.md) |
| Traps and drift | [`known-issues.md`](known-issues.md) |
| Code map (where) | [`code-map.md`](code-map.md) — query it, do not read it |
| History | [`status.md`](status.md) |
