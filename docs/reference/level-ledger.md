# Level ledger

Status: active  
Last updated: 2026-10-30

One row per campaign level: the route source, selected expert, canonical time,
accepted coverage, and the next open decision. This is the quick answer to
"where are we on level N?"; the experiment READMEs remain the detailed record.

The gate column is the provisional `par + 0.30 s` unless a level has an
explicit user target. A selected time above that number is not silently a
failure; the status column records the exception.

| Level | Par / +0.30 | Route source | Selected model | Canonical | Accepted coverage | Status |
|---|---:|---|---|---:|---|---|
| 1-1 | 2.2944 / 2.5944 | 1-1 NEAT lineage | `0073` 1.555M chain | **1.5667** | 12/12 train, 8/8 holdout | selected chain member |
| 1-2 | 2.9376 / 3.2376 | 1-2 transfer | `0073` 1.555M chain | **3.1500** | 31/31 train, 20/20 holdout | selected chain member |
| 1-3 | 3.0816 / 3.3816 | 1-3 NEAT lineage | `0073` 1.555M chain | **3.2667** | 45/45 train, 23/23 holdout | selected chain member |
| 1-4 | 4.1856 / 4.4856 | `0046` 1-4 population | `0073` 1.555M chain | **4.4667** | 11/11 train, 2/2 holdout | selected chain member |
| 1-5 | 2.9088 / 3.2088 | `0080`/`0081` 1-5 | `0085` final 1.655M | **3.1667** | canonical finish; retained levels reported separately | selected 1-5 expert |
| 1-6 | 3.7056 / 4.0056 | `0090`/`0093` 1-6 | `0099` 1.815M | **4.6000** | 32/32 feasible starts | speed gate `4.28 s` still open |
| 1-7 | 4.9728 / 5.2728 | `0102` near-clean archive | `0103` 1.877432M | **5.2333** | near-clean 10/10; all-success 18/20 | final target unset |
| 1-8 | 4.2144 / 4.5144 | `0105` strict-clean archive | `0118` 1.960392M | **4.4667** | strict-clean 40/40; all-success 106/107 | final target unset |
| 1-9 | 2.7648 / 3.0648 | `0121` single-route archive | `0122` 2.060360M | **2.9500** | canonical single route | selected; retention: 1-1/1-3/1-9 finish |
| 1-10 | 3.0336 / 3.3336 | `0125` NEAT transfer | `0126` 2.122840M | **5.3667** | 22/25 eval states | coverage expert; 3 deaths; speed gate open |

## Selected checkpoint paths

```text
1-1..1-4: runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip
1-5:      runs/ddqn_expert_replay_1_5_4frame/dqn/dqn_final.zip
1-6:      runs/ddqn_1_6_start_repair/dqn/checkpoints/ddqn_1814952_steps.zip
1-7:      runs/ddqn_1_7_feasible_starts/dqn/checkpoints/ddqn_1877432_steps.zip
1-8:      runs/ddqn_1_8_canonical_nudge/dqn/checkpoints/ddqn_1960392_steps.zip
1-9:      runs/ddqn_1_9_single_route/dqn/checkpoints/ddqn_2060360_steps.zip
1-10:     runs/ddqn_1_10_feasible_starts/dqn/checkpoints/ddqn_2122840_steps.zip
```

## Shared distillation candidate

The campaign-level student is not a per-level expert replacement yet, but it is
the first shared policy that finishes every canonical level:

```text
checkpoint: runs/distill_campaign_student/students/cand_verified_onehot957_512_false_ddqn2_w11x2_s2/student.pt
input:      lean_stick 191 -> fitted 957 one-hot
model:      957 -> 512 -> 512 -> 2, seed 2
canonical:  16/16 finish, 12/16 strict, sum 82.4833 s
strict miss: 1-3 +1.9666, 1-8 +0.0333, 1-12 +0.0500, 1-14 +1.2500
```

1-14 is evaluated with `reassert_held_jump=false`; all other levels use true
held-jump semantics. The game JSON runtime still needs a `lean_stick` one-hot
spec or a multi-hot/legacy distillation pass before this model can replace the
bundled 58-value policy. See
[`../../experiments/0130_distill_campaign_student/SHIPPING_CANDIDATE.md`](../../experiments/0130_distill_campaign_student/SHIPPING_CANDIDATE.md).

## 1-10 handoff to 1-11

```text
NEAT population: runs/neat_transfer_1_10_from_1_9/neat-checkpoint-131
selected expert: runs/ddqn_1_10_feasible_starts/dqn/checkpoints/ddqn_2122840_steps.zip
level analysis:  ../dragon-jump-remaster/runs/level_analysis/1-11.json (export first)
```

## Ledger notes

- **1-1 through 1-4** are the selected 0073 chain, not four independently
  trained single-level experts. The single-level expert campaign starts at
  1-5; do not relabel the chain retroactively.
- **1-5** is the selected 0085 4-frame model. It is not a full any-start expert
  in the same sense as 1-6/1-7/1-8; use its experiment README for the retained
  level matrix.
- **1-6** passes the start-coverage gate but not the `4.28 s` speed gate. The
  stable clean route is 4.50 s; faster replays are not reproducible.
- **1-7** has no strict-clean route archive. The accepted gate uses the
  near-clean 10-state pool; the 18/20 all-success result is diagnostic.
- **1-8** is a coverage expert. The fastest overall NEAT route (4.4167 s) was
  in holdout cluster `c000`, so the expert reproduced the fastest **train**
  route at 4.4667 s. This is a route-split/speed tradeoff, not a checkpoint
  failure.
- **1-9 is a single-route expert by user decision.** It was trained on the
  fastest clean route only (`926883...`, 2.9500 s), with canonical-only resets.
  It is not a full any-start expert. The 11-route/6-cluster pool is retained as
  a diagnostic artifact.
- **1-9 retention is partial.** Deterministic canonical 1-1..1-9: 1-1 passes
  (2.5167 s), 1-3 finishes slowly (6.0167 s, gate fail), 1-9 passes
  (2.9500 s); 1-2, 1-4, 1-5, 1-6, 1-7, and 1-8 fail. Do not treat the 1-9
  expert as a multi-level policy.
- **1-10 is a coverage expert with the speed gate open.** The selected
  2.122840M checkpoint is canonical **5.3667 s** and covers **22/25** eval
  states (3 deaths, 0 timeouts, spawn 25/25). The 4.7000 s NEAT train route was
  reproduced by a targeted repair (`ddqn_2127832_steps.zip`) but that checkpoint
  covers only **20/25** and is a speed alternative, not selected. The three
  missing states are `[8,25]` f1, `[9,25]` f1, and `[26,20]` f1; they die at
  11/11/31 steps in both the coverage and repair runs. The bundled game AI also
  finishes 1-10 at **4.48 s**, so the 3.0336 s par is beyond current controllers.
- Gate numbers are provisional until the user sets a final target. Record the
  final target here when it is decided.
