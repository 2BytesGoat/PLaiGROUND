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

The level-by-level expert chain is the active path. Levels **1-1 through 1-9**
have selected experts; **1-10 is next**. Each expert is trained and evaluated
on that level's NEAT starts. The end product remains one shared student A with
no level ID or per-level heads.

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

## Next 3

1. Export the 1-10 level analysis to
   `../dragon-jump-remaster/runs/level_analysis/1-10.json`.
2. Seed 1-10 NEAT from the 1-9 population
   (`runs/neat_transfer_1_9_from_1_8/neat-checkpoint-89`), collect routes, then
   run `tools/level.py data` for 1-10.
3. Train and evaluate the 1-10 expert; record the selection in
   [`reference/level-ledger.md`](reference/level-ledger.md).

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
