# Audit — clean-start state capture, world state, and route selection

**Date:** 2026-10-28
**Status:** decision record; core implementation complete 2026-10-28; route/eval
gates pending.
**Scope:** Godot training reset/observation/info contract, PLaiGROUND NEAT
route capture, and the clean-start DQN start pool.

> This note supersedes the powerup-specific held-card, minimum-offset, and
> 794-based parts of
> [`../experiments/dqn-clean-start.md`](../experiments/dqn-clean-start.md).
> The clean-start plan remains the implementation task; this note records the
> decisions that implementation must follow.

## 1. Context

The simplified reward, force-start reset, terminal spike death, and raw-state
observation are shipped. The clean-start design still needed a precise answer
for four things:

1. what belongs in the per-step observation versus reset/restore metadata;
2. how to restore a held powerup without the pickup reappearing;
3. how to capture resettable dynamic level state;
4. how NEAT routes become start scenarios without replaying every route.

The decisions below resolve those. The key boundary is between:

- the **per-step NN state/observation** — only features the policy consumes;
- the **per-step `info` sidecar** — reward and telemetry, not a second
  observation;
- the **reset payload** — one control dict that initialises the simulation,
  with the dynamic world snapshot nested inside it.

## 2. Observation, per-step info, and reset payload

### 2.1 NN state / observation

The raw observation remains the policy input (342 values):

- 7×7 tile IDs and tile states;
- player cell/position, velocity, facing, state;
- grounded/jump phase;
- held powerup stack as **kind IDs**;
- `gold_ratio`;
- run-active / started-walking / finished flags.

It contains no pickup node names, no dynamic-object IDs, no reward counters, and
no route-time offset.

### 2.2 Per-step info; separate world state

The per-step `info` sidecar remains auxiliary and is not fed to the policy. It
carries:

- `run_time`, `best_run_time`;
- `gold_ratio` (also useful for reward);
- `flow_cost`, `flow_in_bounds`;
- `wasted_air_frames`;
- `finished`, `deaths`.

The `gold_ratio` overlap between observation and `info` is intentional: the
network consumes it, while the reward/telemetry path can read the same value.

The dynamic world snapshot is **not** a per-step `info` column. It is captured
on demand and stored in the reset payload when a scenario is built. Held-pickup
IDs/order live inside the `world_state` block, not in the observation.

### 2.3 Reset request (one dict)

The reset request is one dict. There is no `state` / `info` split:

```json
{
  "agent": 0,
  "cell": [12, 18],
  "facing": 1,
  "position": [px, py],
  "velocity": [0.0, 0.0],
  "state_name": "Move",
  "powerup_stack": [0],
  "run_active": true,
  "started_walking": true,
  "run_time": 1.08,
  "best_run_time": 1e9,
  "world_state": { "...": "..." }
}
```

- `position`, `velocity`, `state_name`, `powerup_stack`, and the run flags are
  the physical restore fields.
- `run_time` is the source checkpoint time; the game computes `gold_ratio` from
  it for the next observation.
- `world_state` restores dynamic level cells/objects and held pickups.
- `best_run_time` is telemetry only.
- The observation encoder ignores `run_time`, `world_state`, and any other
  non-NN reset fields.

## 3. Clean-start contract

A clean start is:

- grounded and zero-velocity;
- exact position, cell, and facing restored;
- run clock restored from the source checkpoint;
- held powerup stack restored;
- `run_active` / `started_walking` true;
- force-started by the game; no idle state and no forced first jump;
- no prefix replay.

Powerups can only be activated in the air, so a grounded state with remaining
powerups is stable. The source checkpoint time and the held stack come from the
same run; there is no averaging and no minimum across routes.

Death remains terminal at the hazard. The next reset uses the normal sampled
start, not a forced canonical reset.

Pool policy:

- canonical **0.70**;
- state cells **0.30**;
- facing-mirror and neighboring-cell holdouts;
- state entries are keyed by `(cell, facing, powerup stack)`.

## 4. Dynamic world state

Instead of sending held-pickup coordinates, capture a general snapshot of every
resettable dynamic object and dynamic cell in the level:

- powerup pickups: stable ID, type, visible, held, stack index;
- dissolve blocks: state/timer;
- bounce/reset blocks: orientation/state;
- moving platforms and other resettable objects: position/phase/state.

The snapshot is versioned and keyed by stable IDs. It is not powerup-specific
and not part of the NN state.

### 4.1 Capture / restore

Add a level-scoped API equivalent to:

```text
Level.capture_world_state() -> Dictionary
Level.restore_world_state(snapshot) -> bool
```

The controller/agent owns the level reference, so each viewport restores its
own level. Do not resolve pickup nodes through a global scene-tree group.

Reset order:

1. `Player.reset()` (clears held powerups and starts their respawn animation);
2. restore `world_state` from the reset dict;
3. restore the player pose/run state and `run_time`;
4. rebuild the player powerup stack from the held entries;
5. force-start the run.

### 4.2 Held powerups / despawn

The tricky part is the `Powerup` lifecycle:

- `Player.pick_powerup(area)` hides the node and disables collision;
- `Player.reset()` drops held powerups by calling `Powerup.consume()`, which
  shows the node and plays a `Spawn` animation;
- `_on_animation_ended()` normally calls `Powerup.reset()` and re-enables the
  pickup.

A direct re-pick immediately after reset can therefore be undone when the
animation ends. The restore path needs:

- `Powerup.hold()` / `apply_state()` that stops the animation, hides the node,
  disables collision, and sets a `held` flag;
- `_on_animation_ended()` and `Powerup.reset()` respecting `held`;
- `Player.restore_powerup(area)` that appends the node silently and in LIFO
  order;
- no `queue_free`: the pickup remains reusable after consumption;
- only held pickups are hidden. Consumed pickups already respawn during the
  reference run and must remain visible after reset.

If a held pickup ID cannot be resolved, or its stack does not match the
requested kind stack, the NEAT extraction/pool builder must reject the
candidate. Godot may return an error for a malformed/unsupported restore, but
it must not silently restore fewer powerups or fall back to canonical.

## 5. On-demand world-state protocol

The world state is a separate optional block, not a column in the per-step
`info`. The training loop must not pay for it on every step.

### 5.1 Handshake

```json
{
  "supports_info_request": true,
  "world_state_version": 1
}
```

### 5.2 Request / response

Python sends between steps:

```json
{
  "type": "info_request",
  "request": ["world_state"],
  "agents": [0]
}
```

Godot replies:

```json
{
  "type": "info",
  "request": ["world_state"],
  "world_state": {
    "0": {
      "version": 1,
      "objects": { "...": "..." },
      "cells": { "...": "..." }
    }
  }
}
```

`request` is a list so later blocks (`level_stats`, traces, etc.) can use the
same mechanism.

The game is paused after the step reply and is waiting for the next action. The
`info_request` branch must send the reply and **continue waiting for the action**
(`return handle_message()`); it must not unpause or step. The next action then
resumes the lockstep.

### 5.3 When to request

- **NEAT route capture:** after a step that reaches a new checkpoint
  `(cell, facing, powerup stack)`. Not every step.
- **Training:** never. The start pool already contains the snapshot; the reset
  carries it in the top-level `world_state` field.
- **Evaluation:** optional diagnostics only.
- **Future:** the same request can expose any info block on demand.

## 6. NEAT one-pass route capture

The preferred pipeline records route scenarios during the population run:

1. Run the population normally.
2. For each genome record:
   - actions;
   - route signature: sequence of `(cell, facing)`;
   - final time / finish status;
   - at each checkpoint transition: physical reset state, `run_time`, held
     powerup IDs/stack, and `world_state`.
3. At the end, group runs by route signature.
4. Keep the best final time per signature.
5. Keep that run's checkpoint snapshots and source times; discard the losers.
6. Select diverse routes for the start pool.
7. Build the pool with canonical 0.70 / state 0.30 and the facing/neighbor
   holdout.

A checkpoint transition is the first visit to a new
`(cell, facing, powerup stack)`; do not request world state for every physics
step.

If a better individual follows the same route, its recorded snapshots replace
the old route's scenarios during selection. There is no replay and no mean.

**Fallback:** if storing/capturing world state for every genome is too
expensive, record only paths/times during the population, select the best route
per signature, and replay those selected actions to capture world state. The
replay must reproduce the signature; if it does not, discard or re-cluster it.

The source time at a checkpoint is the `run_time` from the same run. Do not use
a minimum or mean across routes; the state is state-dependent.

## 7. NEAT validation / rejection gate

This gate belongs to NEAT extraction/pool building, not Godot. Godot applies a
well-formed reset and may return an error for a malformed/unsupported restore;
it does not decide whether a candidate is a good training start.

A state is rejected, not silently canonical-reset, when:

- the exact position is not grounded/on-floor;
- the cell is exit/spike/invalid;
- the world state contains an unsupported dynamic object;
- an unheld pickup overlaps the start position;
- the first action window has a hazard;
- a held pickup ID is missing;
- the restored powerup stack does not match the requested kind stack;
- `run_time` is invalid or the requested cell is not accepted.

The Python pool builder validates the same conditions and verifies
`spawn_cell` / `last_reset_spawns` after a reset.

## 8. Time signal caveat

`gold_ratio` is absolute run time. A late-route checkpoint starts with a
favorable ratio: example gold 2.0 s, offset 1.5 s gives `gold_ratio = 1.33`, so
the time term is already positive. The skipped prefix is not directly rewarded
because the reward clock is seeded at the offset, but the state task starts at a
favorable point and has a shorter remaining route. The fixed +500 finish makes
state starts attractive.

Decision for now:

- keep `gold_ratio` absolute in observation and reward;
- keep canonical 0.70 and evaluate canonical separately;
- if state starts dominate or canonical speed regresses, switch the reward to a
  local remaining-gold target while leaving the observation absolute;
- zero-velocity starts are slightly harder than the moving source checkpoint;
  accept and measure, or record velocity if an exact source state is needed.

## 9. Legacy cleanup

- The clean-start target is the raw state/lean encoder; **794 is not used**.
- 58/794 checkpoints remain archived baselines, not active clean-start paths.
- Remove legacy 794/58 branches, defaults, comments, and old-prefix pool usage
  from active clean-start code.
- Re-record route data with checkpoint snapshots and source times; old NEAT
  path logs without world state are insufficient for the one-pass pool.
- No model install until the full gates pass.

## 10. Implementation status

Implemented:

- simplified reward, force-start reset, terminal spike death;
- gold-ratio time term and raw observation `gold_ratio`;
- raw info fields (`flow_cost`, `flow_in_bounds`, `wasted_air_frames`);
- raw 1-1/1-8 validation and game reward tests;
- `Level.capture_world_state` / `restore_world_state` with stable cell IDs;
- held powerup `Powerup.hold` / `apply_state` and `Player.restore_powerup`;
- one-dict game reset with player/run state, held stack, and dynamic
  `world_state` restore;
- `info_request` / `world_state` protocol and Python `GodotEnv.request_info`;
- Python state/info pass-through, single-agent state start, and multi-process
  info request mapping;
- NEAT one-pass checkpoint capture and route archive checkpoint files;
- canonical 0.70 / state 0.30 checkpoint extraction without replay.

Validated:

- game AI suite 36/36; training suite 12/12; Python suite 21/21;
- raw 1-1 state-start smoke (128 DQN steps, 342-value input);
- 192-agent NEAT evaluation: 87 finishes, 87 checkpoint files;
- checkpoint archive -> clean routes -> 10-state raw pool.

Pending:

- re-record and validate 1-8+ powerup routes end to end;
- strict pickup/hazard/first-action rejection tests and full holdout;
- clean-start deterministic eval and promotion gates;
- optional Python lean encoder;
- remove legacy 794/58 defaults and docs from active clean-start paths.

## 11. References

- Clean-start plan: [`../experiments/dqn-clean-start.md`](../experiments/dqn-clean-start.md)
- Observation contract: [`../reference/observation-contracts.md`](../reference/observation-contracts.md)
- Reward contract: [`../reference/reward-contract.md`](../reference/reward-contract.md)
- Raw-state decision: [`2026-10-28-raw-state-observation-decision.md`](2026-10-28-raw-state-observation-decision.md)
- Clock-bias audit: [`2026-10-28-start-pool-clock-bias.md`](2026-10-28-start-pool-clock-bias.md)
