# Audit — student jump-hold timing and mid-air powerups

**Date:** 2026-10-31  
**Status:** open follow-up; diagnosis complete, no fix applied  
**Model:** `runs/distill_campaign_student/students/cand_verified_onehot957_512_false_ddqn2_w11x2_s2/student.pt`  
**Export:** `../dragon-jump-remaster/godot/src/resources/ai/student_lean_stick_512.json`  
**Trigger:** in-game WATCH AI review of the 16/16 distilled student.

The student finishes all 16 canonical levels in Python (12/16 strict, 82.4833 s
sum), but two behaviours look wrong in play:

1. it holds jump longer than expected and releases late;
2. it does not appear to use powerups deliberately in mid-air.

This note records what the data and the game code actually say. The short
version: **the long hold is mostly teacher style, not an inference bug; the
powerup gap is real and is a data problem.** Neither is caused by the model
being broken. `action_repeat=4` is a constraint, not the root cause.

## What is deployed in the game working tree

As of 2026-10-31 the game repo (`prepare-for-expo` @ `50dcc66`, dirty) has an
uncommitted spec v7 `lean_stick` runtime and an exported student:

```text
model_name:        student_lean_stick_512
spec_version:      7
obs_encoding:      lean_stick
observation_size:  191
input_size:        957
hidden_sizes:      [512, 512]
action_repeat:     4
auto_start_run:    true
reassert_held_jump: true
reassert false:    1-14
trained_levels:    1-1 .. 1-16
```

The committed/bundled fallback is still `student_obs58_256.json`. The new
runtime is not committed and has not passed the all-16 in-game gate plus a
powerup gate. Do not treat it as shipped.

## Finding 1 — long holds are in the teacher data

The teachers hold jump for long runs at 15 decisions/s (4 physics frames per
decision). Longest canonical `1` run in the selected expert/teacher paths:

| Level | Longest `1` run | Held time |
|---|---:|---:|
| 1-10 | 43 decisions | ~2.87 s |
| 1-7 | 26 | ~1.73 s |
| 1-4 | 18 | ~1.20 s |
| 1-3 | 11 | ~0.73 s |
| 1-1 | 9 | ~0.60 s |

`reassert_held_jump=true` re-sends the held input every decision, so a held
jump chains through wall contacts and ceiling sticks. That is why the route
looks like it never lets go: it is a bunny-hop / wall-jump chain, not one
over-long jump.

The student is a very close imitator on the canonical traces:

- 100% action match on 9/16 canonical paths (1-1, 1-2, 1-4, 1-5, 1-6, 1-7,
  1-9, 1-13, 1-15).
- 93.8% (1-3), 98.5% (1-8), 98.7% (1-10), 99.2% (1-11), 97.7% (1-12),
  93.2% (1-14), 98.9% (1-16). The legacy 1-14 edge-triggered path is 97.1%.
- Release-boundary errors: **3 late releases / 115** canonical releases; 2 of
  the 3 are on 1-14, 1 is on 1-16.
- Press-boundary errors: **6 missed presses / 116** canonical presses; 1-14
  accounts for 2, 1-12 for 2, 1-8 for 1, 1-11 for 1.

So the student is not inventing long holds. The residual timing errors are
concentrated on 1-14 — the only level evaluated with
`reassert_held_jump=false`, and the strict outlier (8.1833 s vs the 6.9333 s
path). 1-16 has one late release. Those are the places to attack first.

### Why `action_repeat` is not the main cause

The teacher data was also collected at `action_repeat=4`, so the model cannot
learn sub-66 ms release timing from it. Action repeat only sets the minimum
hold quantum; it does not make a policy hold longer than its teacher. Reducing
it would require new data and would break the old-level anchor (see
[`../known-issues.md`](../known-issues.md) §27). Treat it as a separate
experiment, not a default fix.

### Runtime caveat to measure

`PlayerAIModelController` mirrors training with a pending first action and a
one-action-repeat delay (`_first_action_pending`). The old 58-value model's
in-game 1-1 was 2.35 s vs 2.333 s in Python, so the cadence is broadly correct.
But the new model's per-decision trace has not been compared. A one-decision
offset would show up exactly as a late release. Measure before retraining.

## Finding 2 — powerup casts are sparse and mostly auto-casts

`FallState.physics_update` consumes a powerup whenever `wants_to_jump` is true
while falling. It does not require a fresh press. With held-jump semantics, a
`1` that is held through the jump will **auto-cast at the first Fall frame**.

Cast examples in the scanned per-level source trajectories:

```text
1-1 .. 1-7   zero casts
1-8          36 casts (all from held/auto-cast states)
1-9           6
1-10         23
1-11          2
1-12          1
1-13          1
1-14          1
1-15          2
1-16          1
```

The student reproduces the canonical casts (including 1-14 step 89, the
double-jump cast), but the dataset contains almost no example of the useful
pattern: release jump after take-off, wait until the correct fall height, then
press again to cast. Most labels are "hold jump, auto-cast on entering Fall."
That is why it can finish the canonical routes and still look like it cannot
use powerups deliberately.

## Hypotheses, in priority order

1. **H1 — 1-14 release timing.** The 1-14 path is the old 58-value student
   under edge-triggered semantics. The distilled student's two late releases
   and two missed presses there are the material strict miss. Fix this first.
2. **H2 — single-frame aliasing.** The 191 observation has `jump_held` (one
   bit) but no hold age, fall age, or previous action. At a release boundary
   the same frame can support both "keep holding" and "release"; the model
   needs temporal context. A few scalars (`hold_age`, `fall_age`, `last_action`,
   optional `time_since_powerup`) should be enough.
3. **H3 — powerup data coverage.** Add deliberate mid-air cast demonstrations,
   especially for 1-9 and 1-11..1-16. The heuristic controller already casts
   only when falling and no safe landing is below (`_should_air_jump`), so it
   is a usable labeler for targeted states.
4. **H4 — runtime offset.** The pending-first-action path may add one decision
   of latency. Confirm with a trace before changing it.

## Proposed plan

### Phase 0 — trace, do not retrain yet

Add an opt-in per-decision trace to the game model controller and the Python
evaluator: action, state, `jump_held`, powerup stack, `run_time`, position.
Run 1-3, 1-9, 1-10, 1-14, 1-16 in both. Compare:

- release and press timestamps vs the teacher path;
- whether the first action lands one decision late;
- whether casts happen at Fall entry or at a chosen fall height.

**Gate:** game and Python traces agree within one physics frame, or the
difference is explained and fixed in the controller.

### Phase 1 — release timing

Preferred: add temporal scalars to the observation (spec v8):
`hold_age`, `fall_age`/`air_age`, `last_action`, optional
`time_since_powerup`. Retrain the same 512×512 student. This directly encodes
what the single frame lacks without doubling the input.

Cheap alternative: retrain with release-boundary weighting (upsample `1→0`
and `0→1` transitions, asymmetric penalty for false holds). Faster to try, but
it cannot fully solve aliasing if the frame is genuinely ambiguous.

### Phase 2 — powerup demonstrations

Collect targeted casts instead of relying on the canonical paths:

- use the heuristic controller's `_should_air_jump` policy to label falling
  states with no safe landing below;
- or script cast states at varied fall heights on 1-8..1-16 and record
  successful outcomes;
- oversample cast transitions in the distillation set.

Add a powerup gate to evaluation: pickups, casts, casts per pickup, premature
casts, and at least one deliberate mid-air cast where the route requires it.

### Phase 3 — action repeat (separate experiment)

Only if Phase 0 implicates it: test `action_repeat=2` vs `4` on a few levels
with the teacher and student. If it materially improves release timing, it
needs new data and a retrain. Do not change the default first.

## Acceptance gates

- Canonical finish stays 16/16.
- Strict improves, with 1-14 at or below 6.9333 s and 1-16 at or below
  6.1333 s; 1-3 remains the route-level target.
- Powerup gate passes on 1-8..1-16: casts happen at useful heights, not at
  Fall entry, and the model does not burn a powerup when a safe landing is
  available.
- Game and Python traces match within one physics frame.

## Reproduce the diagnosis

Offline: load each `runs/distill_campaign_student/raw*/.../transitions.npz`,
run `StudentMLP` on `obs_onehot`, and compare `argmax` to `actions`. Count
`1`-run lengths and powerup-stack changes using `LEAN_PLAYER_OFFSET` from
`src/plaiground/observations/spec.py`. The per-level source dirs are listed in
`runs/distill_campaign_student/dataset_verified_false_ddqn/index.json`.

In-game: `python3 ../dragon-jump-remaster/tools/ai_bench.py --no-log 1-3 1-14`
for canonical times. Python: use the 1-14 command in the shipping card with
`--no-reassert-held-jump`, and the same command without that flag for the other
levels. The per-decision trace is the Phase 0 work, not yet implemented.

## Relevant code

- Game cast rule:
  `../dragon-jump-remaster/godot/src/player/states/fall_state.gd`
- Game jump release / hover:
  `../dragon-jump-remaster/godot/src/player/states/jump_state.gd`
- Game model cadence:
  `../dragon-jump-remaster/godot/src/player/controllers/player_ai_model_controller.gd`
- Held-jump action semantics:
  `../dragon-jump-remaster/godot/src/player/controllers/player_ai_training_controller.gd`
- Export:
  `tools/export_student_json.py`
  → `../dragon-jump-remaster/godot/src/resources/ai/student_lean_stick_512.json`
- Student:
  `runs/distill_campaign_student/students/cand_verified_onehot957_512_false_ddqn2_w11x2_s2/student.pt`
- Shipping card:
  [`../../experiments/0130_distill_campaign_student/SHIPPING_CANDIDATE.md`](../../experiments/0130_distill_campaign_student/SHIPPING_CANDIDATE.md)
