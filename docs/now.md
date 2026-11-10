# Now

**Updated:** 2026-10-31
**Repo:** branch `neat` @ `fc747ea` (dirty: untracked
`tools/export_student_json.py`); run `make brief` for live status.
**Game:** `../dragon-jump-remaster` branch `prepare-for-expo` @ `229422d`
(working tree: exported `student_lean_stick_512.json`, new golden fixture, and
an untracked `student_lean_stick_512_baseline.json`; run `make brief` for live
status).

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

A 1-3 DAgger repair (`0134`) now improves the shared student to **13/16 strict
and 80.2833 s**, still 16/16 finish:

```text
runs/distill_campaign_student/students/cand_dagger_1_3_13of16/student.pt
1-3 5.2333 -> 3.2667; 1-14 8.1833 -> 7.4500
remaining strict misses: 1-8 (+0.5166), 1-12 (+0.0500), 1-14 (+0.5167)
```

The repair is exported to the game working tree as
`student_lean_stick_512.json`; the previous export is retained as
`student_lean_stick_512_baseline.json`. It is not committed and has not passed
the in-game all-16/powerup gate. The four baseline strict misses were 1-3
(+1.9666), 1-8 (+0.0333), 1-12 (+0.0500), and 1-14 (+1.2500). 1-14 is evaluated
with `reassert_held_jump=false`; the other levels use true semantics. The 1-8
and 1-12 baseline misses were one action-repeat frame.

The selected student is now exported to the **game working tree** as
`student_lean_stick_512.json`, using the uncommitted spec v7 `lean_stick` +
one-hot runtime. It is **not committed or gated**; the committed fallback is
still `student_obs58_256.json` (16/16, 78.583 s sum). In-game review found two
follow-ups: jump holds stay on too long (mostly teacher style; residual late
releases on 1-14/1-16) and mid-air powerups are under-taught (a held jump
auto-casts at Fall entry; there are only 1-2 deliberate cast examples per late
level). See
[`audit/2026-10-31-student-hold-and-powerups.md`](audit/2026-10-31-student-hold-and-powerups.md)
and
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

1. **Trace the exported student in game vs Python** on 1-8, 1-12, 1-14, plus
   the 1-9/1-10/1-16 canaries. Compare per-decision actions, release/cast
   timestamps, and the first-action offset. Settle the
   `_first_action_pending` one-decision hypothesis before touching the model.
2. **Add temporal features and powerup demonstrations in one retrain.** Add
   `hold_age`, `fall_age`/`air_age`, and `last_action` (optionally
   `time_since_powerup`), plus targeted mid-air cast examples from the
   heuristic/scripted states. Keep `action_repeat=4` for this pass; do not
   change it without a separate experiment.
3. **Keep canonical 16/16 as the regression gate.** Do not overwrite the
   committed 58-value fallback or claim the new model is shipped until the
   working-tree runtime is committed and the all-16 in-game + powerup gates
   pass. Keep the 7x7 encoder deferred.

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
- **Student holds jump too long and powerups are under-taught** (high): long
  holds are teacher style; held jump auto-casts at Fall entry. Read
  [`audit/2026-10-31-student-hold-and-powerups.md`](audit/2026-10-31-student-hold-and-powerups.md)
  before changing `action_repeat`, the action space, or the export.
- **Canonical eval is one deterministic trace** (medium): it is not a
  robustness measure.
- **NEAT fitness does not rank fast finishers first** (medium).

## Commands

| Command | What it answers |
|---|---|
| `make brief` | branch, game commit, running jobs, selected levels, recent runs |
| `make run RUN=<name>` | one run's checkpoints, eval summaries, and log tail |
| `make search q="..."` | capped text search; skips `runs/`, `data/`, binaries |
| `make context q="..."` | ranked code, docs, experiments and open work |
| `make level-status LEVEL=1-9` | one level's experiments and data summaries |
| `make index` / `make index-check` | regenerate / verify `.repo-brief/map.md` |
| `make preflight` | map, profile, front doors, dates, and references |

## Where to look

| Need | Read |
|---|---|
| Current state | this page |
| Selected expert and gate | [`reference/level-ledger.md`](reference/level-ledger.md) |
| Student hold/powerup follow-up | [`audit/2026-10-31-student-hold-and-powerups.md`](audit/2026-10-31-student-hold-and-powerups.md) |
| Full new-level procedure | [`runbooks/new-level.md`](runbooks/new-level.md) |
| Architecture | [`architecture.md`](architecture.md) |
| Traps and drift | [`known-issues.md`](known-issues.md) |
| Code map (where) | [`.repo-brief/map.md`](../.repo-brief/map.md) — query it, do not read it |
| Decisions | [`project/decisions.md`](project/decisions.md) |
| History | [`status.md`](status.md) |
