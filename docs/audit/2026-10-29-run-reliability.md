# Run reliability audit — 2026-10-29

Scope: the active 1-1 .. 1-6 DQN/DDQN and NEAT runs, plus the shared
evaluation, replay-collection, and Godot-process harness. This is an
investigation and a set of low-risk harness fixes; no training run was
restarted.

## Executive summary

The largest 1-6 problem was not the network or the level: it was **checkpoint
selection**. The first 1-6 sweep evaluated 1.68M, 1.73M, and the final, but
skipped 1.705M. A complete sweep finds:

```text
1.680M   4.6833 s
1.705M   4.5667 s   <- selected
1.730M   8.6000 s
1.755M   8.3167 s
final    7.2500 s
```

The 1.705M checkpoint is reproducible on the canonical trace at speedup 12 and
16, with one finish, zero deaths, zero timeouts, and efficiency 1.0. It is still
above the 4.28 s gate, but it is the correct 1-6 baseline and 0.1167 s better
than the previous selection.

The second-largest issue is **evaluation semantics**. The old summary counted
every non-finish as a timeout, so a death appeared as `deaths=1, timeouts=1`.
That makes death/timeout reliability impossible to read and has already
mislabeled 1-5 and 1-6 rows. This is fixed and tested.

The third issue is **behavioral mode selection**. The 1-6 anchor contains real
finished episodes at 3.0667, 3.2, and 3.3333 s under 2% exploration, but the
greedy checkpoints select a 4.68 s route. The fast modes are not deaths and not
a single random outlier: 3.3333 s appears 96 times and 3.2 s 19 times in 3,224
anchor episodes. Extracting those actions and testing BC/DAgger is a credible
path to a sub-4.28 deterministic policy.

The rest of the reliability risk is harness-level: blocking socket reads with
no timeout, a close path that did not reap the Godot process, silent spawn
fallback in evaluation, canonical-only 1-6 training data, and manual checkpoint
selection.

## Evidence: 1-6 outcomes

From `runs/ddqn_anchor_1_6_1680/dqn/monitor.csv` (1-6-only, 2% exploration,
3,224 episodes):

```text
finished:        1,697
deaths:          1,412
timeouts:          115
median finish:    4.6833 s
finishes <4.28:     155
```

Fast finish counts:

```text
3.0667 s           1
3.2000 s          19
3.2167 s           9
3.3333 s          96
4.6833 s       1,027
```

The 3.0667/3.2/3.3333 rows have `finished=True`, `timed_out=False`, and
positive finish rewards (~603-607). They are real finishes, not the 2.43 s
death. The fast route is a **mode** of the behavior policy, not a deterministic
checkpoint.

The 0091 training monitor shows the same pattern:

```text
1-6 episodes:    553
finishes:        274
deaths:          223
timeouts:         56
fastest finish:  3.0833 s
median finish:   5.4167 s
```

The 0095 retrain monitor:

```text
1-6 episodes:    539
finishes:        324
deaths:          159
timeouts:         56
fastest finish:  3.0833 s
median finish:   5.4500 s
final deterministic: 6.3833 s
```

The existence of 3.08 s stochastic episodes in both training runs confirms the
fast mode is reachable; the deterministic policy is not selecting it.

## What is wrong with the old 1-6 numbers

- `4.6833 s` was the best **examined** checkpoint, not the best saved checkpoint.
  `1.705M` is 4.5667 s.
- The 0094 collector reported `best=2.4333`; that is a death. The best
  deterministic canonical model is 4.5667 s; the best recorded stochastic
  finish is 3.0667 s; the reported 4.0667 s replay is **not reproducible in
  isolation** and is not a clean teacher route (0097 follow-up).
- `3.20 s` was previously described as a stochastic training finish. It is a
  real finish, but it is not a policy checkpoint; the 0095 final is 6.3833 s.
- Old eval `timeouts` counts include deaths. Do not compare an old eval's
  `timeouts` field to a new one; use the CSV rows' `finished`, `timed_out`, and
  `deaths` columns, or re-run the eval.

## Harness defects found

### 1. Evaluation summary conflated deaths and timeouts — fixed

`summarize()` previously did:

```python
"timeouts": sum(1 for row in rows if not row["finished"])
```

A terminal death is also `not finished`, so it was counted as a timeout. For
example, the 0091 1-5 row reported `deaths=1, timeouts=1`, and the 0095 1.7175M
1-6 row did the same. A real timeout is `timed_out=True`.

Now:

```python
"deaths":  non-finished, non-timed-out rows
"timeouts": timed_out rows
"spawn_failures": rows whose requested start was not applied
```

The 1-6 canonical eval now reports `deaths=0, timeouts=0`; a death would report
`deaths=1, timeouts=0`. Regression tests cover both.

### 2. Evaluation accepted silent spawn fallback — fixed

`run_starts()` recorded `spawn_ok=False` but `evaluate_set()` still published a
summary. A start-pool entry that Godot silently replaced with a canonical start
could therefore produce a plausible-looking but invalid result.

`evaluate_set()` now raises a `RuntimeError` listing the failed starts after
writing the per-start CSV. The CSV remains for diagnosis; no summary is
published. A unit test uses a fake wrong spawn cell.

### 3. Godot socket could hang or leak a process — fixed

The accepted connection had no timeout (`connection.settimeout` was commented
out). `_get_data()` caught `socket.timeout` but none was set, and a `recv()`
returning `b""` would loop forever. A stuck game could therefore hang a
training or evaluation run indefinitely; a port collision could look the same.

The close path sent a close message, slept one second, and closed the socket but
did not wait for or terminate the Godot process. A process that ignored close
could keep the port and make the next run fail.

Now:

- accepted connections inherit `GodotEnv.DEFAULT_TIMEOUT`;
- `_recv_exact()` raises `TimeoutError` on timeout and `ConnectionError` on a
  closed connection instead of looping;
- `close()` closes the socket, waits for the process, terminates, and finally
  kills it;
- `atexit` registration happens immediately after launch, so a failed handshake
  still cleans up;
- `_launch_env()` passes an argument list rather than a space-split string, so
  paths with spaces are safe;
- bind failures report the port and likely collision;
- a real 1-6 eval after the change passed and left no Godot process running.

Tests cover length-prefixed framing, closed sockets, timeouts, partial frames,
and invalid lengths.

### 4. The 1-6 start pool was canonical-only — data gap closed

The 1-6 pool summary said:

```text
canonical-only because the 1-6 level analysis JSON is not generated
state_starts: 0
```

That means 0090/0091/0094/0095 saw only the canonical start for 1-6, unlike
1-1..1-5 which have multi-start pools. The game-side analysis has now been
exported for 1-6, and a new state pool was built without overwriting the old
pool:

```text
runs/neat_transfer_1_6_from_1_5/pool_state/
  train_states:   11
  holdout_states:  6
  shared_keys:     5
  canonical_weight: 0.70
```

It was not used by the existing runs. A one-step reset smoke on all 18 entries
(1 canonical + 11 train states + 6 holdout states) applied every requested
spawn cell (`spawn_ok=18/18`). It is the right pool for the next 1-6-only
training attempt, especially if the goal is robustness across starts rather
than only the canonical trace.

### 5. Final checkpoints are not selected checkpoints — process fix

The following runs show the same failure mode:

- 0091: final 7.25 s, best 1.705M 4.5667 s.
- 0095: final 6.3833 s; it was the best of that run, but still worse than the
  0091 1.705M 4.5667 s baseline it was retraining from.
- 0060: final 1-2 holdout 4.00 s median; selected 1.3M was 3.35 s.
- 0085: final 3.1667 s; selected final was best there, but old levels failed.

The harness has no automatic best-checkpoint selection. `dqn_final.zip` is
simply the last save. Every run must evaluate all saved checkpoints and record
the selected path. A checkpoint sweep is not optional.

Recommended standard:

```bash
for ckpt in runs/<run>/dqn/checkpoints/*.zip; do
  .venv/bin/python tools/evaluate.py \
    --config experiments/<eval-config> \
    --model "$ckpt" \
    --level <level> \
    --tag "sweep_$(basename "$ckpt" .zip)"
done
```

Then select from `eval/sweep_*/summary.json`, not from `dqn_final.zip`.

### 6. One canonical trace is not a robustness result — open

The evaluator uses one deterministic trace. It is reproducible for 1-6 (1.705M
is 4.5667 s at both speedups), but it does not measure deaths or timeouts,
because a deterministic policy follows one path. The 1-6 anchor needed 2%
exploration to expose the 3.0-3.5 s modes.

Do not use the canonical trace alone to claim a policy is reliable. At minimum:

1. repeat canonical evaluation in a fresh process;
2. evaluate the train/holdout start pools;
3. run a fixed-epsilon stochastic sweep (for 1-6, epsilon 0.02, N >= 100) and
   report finish/death/timeout rates and best/median finish;
4. only then choose the deterministic checkpoint.

The first two are available today. The stochastic evaluator is not implemented;
the 0094 collector is the current substitute.

### 7. Provenance is manual and stale — open

The active run directories have no `metadata.json`; `tools/record_provenance.py`
was not used. The game repo is now at `50dcc66` (two non-training commits after
the `fe7821d` recorded in the manifests), while the training code is unchanged.
The manifests are still valid for behavior, but the actual run provenance is
not recorded in the run directories.

Run `tools/record_provenance.py --experiment <config> --run-dir <run>` for every
new run, or integrate the same call into the trainer. Record the exact game
commit and whether the tree was dirty; do not rely on the manifest alone.

### 8. Action-repeat mixing is still a footgun — known

0082/0083 showed that a 2-frame online env with a 4-frame anchor/replay is
invalid: 1-5 died and 1-4 timed out. 1-6 has not been tried at 2 frames. If a
2-frame 1-6 attempt is made, the online env, expert replay, and anchor must all
be 2-frame, and the selected 4-frame models must not be evaluated as if they
were 2-frame.

The 4.5667 s model is a **canonical** result, not a robust start-set result. On
the later full **32-state** 1-6 pool it finishes **11/32**, with 20 deaths and
one timeout (median 6.1167 s). The older 13/18 figure used a smaller 18-entry
pool. 0091 trained on a canonical-only pool, so this is the expected
generalization gap. 0097/0099 trained on the full pool and selected the 1.815M
any-start expert: **32/32 finishes, 0 deaths/timeouts, median 6.0417 s,
canonical 4.6000 s**.

## Standard run checklist

Use this for every new 1-6 or chain run:

- [ ] Record game HEAD, branch, dirty state; confirm the manifest commit.
- [ ] Confirm one `action_repeat` across online env, expert buffer, and anchor.
- [ ] Confirm preprocessing output size equals the model input size.
- [ ] Confirm unique `base_port` and `eval_port`; no stale Godot process.
- [ ] Use a level-tagged start pool with canonical and state starts; keep
      holdout routes out of training.
- [ ] Let the run finish, then evaluate **every** checkpoint, not only the final.
- [ ] Run canonical twice in fresh processes and the train/holdout pools once.
- [ ] For 1-6, run a fixed-epsilon stochastic sweep; report finish/death/timeout.
- [ ] Select the checkpoint by the gate metric; record its exact path.
- [ ] Write provenance and update the experiment README and `docs/status.md`.

## What is reliable already

The per-level chain is the reliable selection strategy:

```text
1-1  1.5667 s   1.555M chain model
1-2  3.1500 s   1.555M chain model
1-3  3.2667 s   1.555M chain model
1-4  4.4667 s   1.555M chain model
1-5  3.1667 s   0085 final (confirmed twice)
1-6  4.5667 s   0091 1.705M (confirmed at speedup 12 and 16)
```

The 1-5 single-model retention attempt (0088) failed, and the 0091 1-6 model
loses 1-5/1-4; a per-level model is explicitly allowed. Do not force one
backbone to retain all levels without a frozen per-level head or dedicated
expert buffer.

## Prioritized actions

### P0 — do before the next 1-6 training run

1. Use `runs/ddqn_expert_replay_1_6_lean_stick/dqn/checkpoints/ddqn_1705000_steps.zip`
   as the 1-6 reference; stop calling 1.68M/4.6833 the baseline.
2. Sweep every checkpoint of every new run. Record the selected path in the
   experiment README and `docs/status.md`.
3. Use the full 1-6 state pool (32 unique starts; `pool_state_50` / `pool_state_70`)
   for the next 1-6-only
   attempt; keep canonical at 0.70 and state at 0.30.
4. Add a repeated canonical eval (fresh process) plus a fixed-epsilon stochastic
   eval. Report finish/death/timeout, best/median, and the per-mode times.
5. Run `record_provenance.py` for every new run; update the game HEAD note in
   docs.

### P1 — next model attempt

6. Do not use the 3.3333 s / 3.2 s anchor mode or the 4.0667 s replay as clean
   teacher routes until route-order-independent replay is demonstrated. The
   3.3333 s mode appears 96 times; it is evidence that a <4.28 policy may be
   learnable, but it still needs a reproducible action trace.
7. The 1-6-only DDQN with the new state pool is complete: 0099 1.815M is 32/32
   any-start at 4.6000 s canonical. Keep the 4-frame contract.
8. Only if that misses the speed gate, run the one 2-frame 1-6 attempt; re-collect the anchor
   and expert at 2 frames.
9. Do not enlarge the network first. The 58-value 81k student clears 1-6 at
   5.05 s; the 754k DDQN is at 4.5667 s. Test temporal context/frame stack
   before width.

## Implemented in this pass

- `src/plaiground/evaluation/rollout.py`: death/timeout/`spawn_failures`
  separation; strict start-pool spawn failure.
- `src/plaiground/envs/godot.py`: socket timeouts and close detection, process
  reap/terminate/kill, early atexit registration, list-based launch, clear port
  bind errors.
- `tests/test_evaluation.py`: death vs timeout, spawn failure.
- `tests/test_godot_socket.py`: framing, timeout, closed socket, partial frame,
  invalid length.
- 1-6 analysis export and `runs/neat_transfer_1_6_from_1_5/pool_state/`.
- 1.705M checkpoint eval tags and documentation correction.

Validation: full test suite passes; a real 1-6 eval completed with no leftover
Godot process.

## Artifacts

```text
runs/ddqn_expert_replay_1_6_lean_stick/eval/check_1705000/
runs/ddqn_expert_replay_1_6_lean_stick/eval/check_1705000_repeat/
runs/ddqn_expert_replay_1_6_lean_stick/eval/check_1705000_speed12/
runs/ddqn_expert_replay_1_6_lean_stick/eval/selected_default_check/
runs/ddqn_expert_replay_1_6_lean_stick/eval/state_pool_1705/
runs/ddqn_anchor_1_6_1680/dqn/monitor.csv
runs/neat_transfer_1_6_from_1_5/pool_state/
../dragon-jump-remaster/runs/level_analysis/1-6.json
```

See also: [`2026-10-29-1-6-ddqn-investigation.md`](2026-10-29-1-6-ddqn-investigation.md),
[`../status.md`](../status.md), [`../known-issues.md`](../known-issues.md).
