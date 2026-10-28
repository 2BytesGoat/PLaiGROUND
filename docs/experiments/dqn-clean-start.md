# Clean-start DQN — level-agnostic player-init contract

Status: **core implementation complete 2026-10-28; 1-1 smoke passed; validation/promotion pending**
Started: 2026-10-28
Last updated: 2026-10-28
Levels: all; 1-1 first
Driver: [`../../scripts/reinforcement/02_dqn.py`](../../scripts/reinforcement/02_dqn.py)
Decision record:
[`../audit/2026-10-28-clean-start-state-capture.md`](../audit/2026-10-28-clean-start-state-capture.md)
Related: [`dqn-route-spawns.md`](dqn-route-spawns.md),
[`../audit/2026-10-28-start-pool-clock-bias.md`](../audit/2026-10-28-start-pool-clock-bias.md),
[`../audit/2026-10-28-dqn-pipeline-review.md`](../audit/2026-10-28-dqn-pipeline-review.md)

> **Decision.** A clean start is an explicit **player-init payload**, not a
> cell teleport and not a prefix replay. The game restores position, facing,
> zero velocity, a source-checkpoint run time, the run-active flags, held
> powerups, and the dynamic world state. The same schema is used on every level;
> 1-1 simply has an empty powerup/world-state payload. Cell-only and
> prefix-replay starts are not the clean-start path.

> **Observation target.** Clean-start training uses the raw state / new lean
> encoder, not 794. The 58/794 checkpoints remain archived baselines only. The
> raw packet is implemented and validated (342 values, 1-1 and 1-8); the Python
> lean encoding is implemented in Godot (188 values) and Python one-hots its
> categoricals; driver wiring is still pending.

> **One reset dict.** The per-step observation is one dict; the reset request is
> another single dict. The observation contains only policy inputs. Restore
> metadata (`run_time`, `best_run_time`, `world_state`) stays in the reset dict;
> `world_state` is nested there and never enters the observation.

## 1. Why this design exists

- A cell-only reset zeroes `run_time`. A start near the exit then reads
  `gold_ratio = 3` (faster than gold) and gets a misleading time term. See the
  clock-bias audit.
- Prefix replay preserves the clock, but it also carries NEAT velocity, jump
  phase, and powerups. It is not a neutral clean start. Training reset now
  force-starts the run, so old prefix pools are invalid.
- The flow field is a BFS distance, not a time. `time_to_exit` is `-1` on the
  floor rows, including the canonical start, so it cannot provide a clock.
- Levels 1-8+ contain powerups. A held powerup must be restored so the agent
  can use it later; the pickup node must stay despawned so it is not collected
  twice.
- Levels with dissolve blocks, moving platforms, or other dynamic objects need
  their dynamic state restored, not just the player. The clean-start state
  carries a generic world snapshot for this.
- The source checkpoint time and state are state-dependent. Do not average or
  take a minimum across different routes/powerup stacks.

## 2. The contract

### 2.1 Reset payload

```json
{
  "agent": 0,
  "cell": [4, 7],
  "facing": 1,
  "position": [64.0, 112.0],
  "velocity": [0.0, 0.0],
  "state_name": "Move",
  "powerup_stack": [0],
  "run_active": true,
  "started_walking": true,
  "run_time": 1.08,
  "best_run_time": 1e9,
  "world_state": {
    "version": 1,
    "objects": {},
    "cells": {}
  }
}
```

- `position` is optional. If absent, the game uses the cell centre.
- `velocity` is `[0, 0]` for a clean start.
- `state_name` is a grounded state (`Idle`/`Move`); jump/fall states are
  rejected by NEAT before they reach the pool.
- `powerup_stack` is the NN-visible held stack, bottom → top, length 0–3:
  `0` double jump, `1` stomp, `2` dash, `3` grapple.
- `run_time` is the source checkpoint time (see §3), not a mean/min. The
  observation encoder ignores it.
- `best_run_time` is `1e9`/INF; telemetry only.
- `run_active` and `started_walking` are `true`; training reset force-starts the
  run and injects no first jump.
- `world_state` is the dynamic level snapshot; see §2.2. It is not an
  observation field.

### 2.2 World state, not held-card coordinates

`world_state` is a general snapshot of every resettable dynamic object and
dynamic cell in the level:

- powerup pickups: stable ID, type, `visible`, `held`, `stack_index`;
- dissolve blocks: state/timer;
- bounce/reset blocks: orientation/state;
- moving platforms and other resettable objects: position/phase/state.

The snapshot is versioned and keyed by stable IDs. It is not the NN state. It
uses the level under the agent's viewport, not a global scene-tree group.

Powerup restore/despawn rules:

- only held pickups are hidden/disabled;
- consumed pickups already respawn in the reference run, so they remain visible;
- `Powerup.hold()` / `apply_state()` stops the Spawn animation, hides the node,
  disables collision, and sets a `held` flag;
- `_on_animation_ended()` and `Powerup.reset()` must respect `held`;
- `Player.restore_powerup(area)` appends silently in LIFO order;
- do **not** `queue_free` the pickup; the game's pickup mechanic is reusable;
- a missing pickup ID or a stack mismatch is rejected by NEAT extraction; Godot
  returns an error rather than silently restoring a partial stack.

### 2.3 Game-side reset order

1. `Player.reset()` — clears held powerups and starts their respawn animation.
2. Restore `world_state` — dynamic objects/cells and held pickup state.
3. Restore player pose/velocity/state and `run_time`.
4. Rebuild the player powerup stack from the held entries.
5. Force-start the run.

`_apply_state_after_reset()` already restores position, velocity, state name,
run time, best time, and run flags for grounded, powerup-free states. The
clean-start work extends it with the one-dict reset and the world-state/powerup
path.

### 2.4 NEAT validation gate

The acceptance gate belongs to NEAT extraction/pool building, not Godot. Godot
applies a well-formed reset and may return an error for a malformed or
unsupported restore; it does not decide whether a candidate is a good training
start.

Reject the candidate; do not silently fall back to canonical, when:

- the exact position is not grounded/on-floor;
- the cell is exit/spike/invalid;
- the world state contains an unsupported dynamic object;
- an unheld pickup overlaps the start position;
- the first action window has a hazard;
- a held pickup ID is missing;
- the restored powerup stack does not match the requested kind stack;
- `run_time` is invalid or the requested cell is not accepted.

The Python pool builder validates the same conditions and checks `spawn_cell` /
`last_reset_spawns` after reset.

## 3. Route time source

The offset is the elapsed time the **same reference run** had already spent at
the checkpoint.

- Record `run_time` at each `(cell, facing, powerup stack)` checkpoint.
- Store the source `run_id` / route / step with the scenario.
- Do not use a minimum or mean across different runs; state is state-dependent.
- If a better individual follows the same route, use its checkpoint times and
  snapshots; replace the old route scenario set.
- Freeze the time for the training run. Never update it from the current policy.
- If a checkpoint has no source sample, skip it or mark it low-confidence. Do
  not silently use 0. The flow field is only a distance, not a clock.

Powerups can only be activated in the air, so a grounded checkpoint with a held
stack is stable. The source run time remains valid for that held state. The
clean start zeroes velocity, which is slightly harder than a moving source
checkpoint; accept and measure, or record velocity if an exact source state is
needed later.

## 4. Route capture pipeline

### 4.1 One-pass capture during population (preferred)

For each genome, record:

- actions;
- route signature: sequence of `(cell, facing)`;
- final time / finish status;
- at each checkpoint transition: physical reset state, `run_time`, held
  powerup IDs/stack, and `world_state`.

A checkpoint transition is the first visit to a new
`(cell, facing, powerup stack)`, not every physics step.

World state is requested on demand after the step, while the game is paused
between the step reply and the next action:

```python
env.request_info(["world_state"], agents=[agent_index])
```

At the end:

1. Group runs by route signature.
2. Keep the best final time per signature.
3. Keep that run's checkpoint snapshots and source times.
4. Discard losing runs.
5. Select diverse routes for the pool.
6. Build the pool.

If a better individual follows the same route, its recorded checkpoint
snapshots replace the old route scenarios during selection. No replay, no mean.

### 4.2 Two-pass fallback

If one-pass storage or checkpoint capture is too expensive:

1. Record paths and times during the population only.
2. Select the best route per signature.
3. Replay the selected actions.
4. Request `world_state` at checkpoints during the replay.
5. Verify the replay signature/time; otherwise discard or re-cluster.

The source time is still from the captured/replayed run, never a mean.

### 4.3 Route registry / scenario data

```text
RouteRecord:
  signature: [(cell, facing), ...]
  best: {genome, final_time, actions}
  scenarios: {
    (cell, facing, powerup_stack): {
      state, run_time, world_state, source_step
    }
  }
```

The pool uses the best route's scenarios. A route replacement is atomic: never
mix snapshots from different runs.

## 5. Pool construction

1. Use grounded cells only; no exit; no unsupported dynamic object.
2. Keep the source checkpoint time; no mean/min across routes.
3. Emit `state` entries with:
   - zero velocity;
   - `run_time` from the source checkpoint;
   - `powerup_stack` recorded;
   - `world_state` from the same checkpoint;
   - no `prefix_actions`.
4. Weights: **canonical 0.70 / state cells 0.30**.
5. Split before training and freeze in `pool/split.json`:
   - **train** checkpoints;
   - **facing holdout**: mirror facing of selected training checkpoints;
   - **neighbor holdout**: an adjacent standable cell one block left/right,
     not in training;
   - every split entry carries its source time/world state; if no source exists,
     skip or mark low-confidence; do not silently use 0.

The split is a start-condition test, not an unseen-level test. The agent still
sees those cells during canonical runs.

## 6. Protocol

The world state is a separate optional block, not a column in per-step `info`.

Handshake advertises:

```json
{
  "supports_info_request": true,
  "world_state_version": 1
}
```

Request / response:

```json
{"type": "info_request", "request": ["world_state"], "agents": [0]}
```

```json
{
  "type": "info",
  "request": ["world_state"],
  "world_state": {
    "0": {"version": 1, "objects": {}, "cells": {}}
  }
}
```

The game is paused after the step reply. The `info_request` handler must send
the reply and keep waiting for the next action; it must not unpause or step.

Training does not request world state. The start pool stores it; the reset sends
it as the top-level `world_state` field. Future requests can ask for other
blocks with the same message.

## 7. Python changes

- `GodotEnv`: add `request_info(keys, agents=None)` and decode the separate
  `world_state` reply. Keep the step/reset info columnar.
- `SingleAgentGodotEnv.reset()`: add a single reset-dict branch that forwards the
  full payload; no start-jump injection; set `start_source="state"` and
  `start_kind`.
- `load_start_pool()`: preserve and validate the physical fields, `run_time`,
  held powerup IDs/stack, and `world_state`.
- `scripts/reinforcement/02_dqn.py`:
  - replace the 794 clean-start path with raw state;
  - extend extraction with route signatures, checkpoint snapshots, source
    times, and route-registry best-per-signature selection;
  - add the one-pass checkpoint capture and the two-pass fallback;
  - remove legacy 794/58 defaults, branches, comments, and the old-prefix pool
    usage from the active clean-start path;
  - make `evaluate()` attach the state/offset/world state to state starts and
    run canonical separately.
- Tests:
  - pure tests for route grouping, best-per-signature replacement, checkpoint
    keys, source times, split construction, and state-field preservation;
  - wrapper test for reset-dict pass-through, no start jump, and
    `info_request`;
  - NEAT/pool tests for route grouping, source-time/world-state preservation,
    and strict candidate rejection;
  - game-side GdUnit tests for powerup restore/despawn, world-state
    capture/restore, and run-time/gold ratio.

## 8. Training and evaluation

- **Completed smoke (2026-10-28):** 128 DQN steps on 1-1 raw state (342-value
  input) with a 10-state + canonical pool. The 192-agent NEAT evaluation
  produced 87 finishes and 87 checkpoint files; `41 extract` built the 10-state
  pool without replay. This validates plumbing, not training quality.
- **Smoke:** 1 env × 10k, raw state, new encoder, clean-start pool.
  Verify:
  - obs size is the expected raw/lean size;
  - every state entry resets with the requested `run_time`;
  - velocity is zero;
  - powerup stack and world state match;
  - only canonical and `state` starts are sampled;
  - no prefix entries;
  - canonical and 1-8 powerup smoke pass.
- **Full:** 4 envs × 200k after smoke, checkpoints every 25k,
  `runs/dqn_clean_start_1_1`; evaluate checkpoints offline.
- **Eval:** canonical + facing holdout + neighbor holdout + full strata
  coverage. Report each group separately. Do not compare clean-start cell
  coverage to cold-cell or prefix-replay results.
- **Gates:**
  - all state entries validate;
  - canonical is competitive with the NEAT teacher (2.15 s) and the established
    canonical baseline before any claim;
  - held-out facing/neighbor finish rates do not collapse;
  - no model is installed in the game.

## 9. Open risks

- `Level.capture/restore_world_state` must cover each dynamic object type we
  allow; NEAT rejects candidates containing unsupported objects.
- Powerup animation/held-state race must be tested across frames and after
  consumption.
- The one-pass capture adds world-state work to population evaluation; fall
  back to the two-pass replay if throughput is unacceptable.
- Absolute `gold_ratio` makes late checkpoints time-favourable; keep canonical
  evaluation separate and switch to a local remaining-gold reward target only if
  state starts dominate.
- Zero-velocity starts are harder than moving source checkpoints; accept and
  measure.
- The reset reply advances one physics tick past the restored clock; measure
  the drift. The raw/lean encoders exist; driver wiring is still pending.

## 10. References

- Decision record:
  [`../audit/2026-10-28-clean-start-state-capture.md`](../audit/2026-10-28-clean-start-state-capture.md)
- Observation contract:
  [`../reference/observation-contracts.md`](../reference/observation-contracts.md)
- Reward contract:
  [`../reference/reward-contract.md`](../reference/reward-contract.md)
- Raw-state decision:
  [`../audit/2026-10-28-raw-state-observation-decision.md`](../audit/2026-10-28-raw-state-observation-decision.md)
- Clock-bias audit:
  [`../audit/2026-10-28-start-pool-clock-bias.md`](../audit/2026-10-28-start-pool-clock-bias.md)
