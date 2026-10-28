# Architecture

How PLaiGROUND connects to Dragon Jump, and how the training/evaluation stack is
layered. For game-side behavior, the source of truth is the game vault:
`../dragon-jump-remaster/docs/technical/training.md` and
`../dragon-jump-remaster/docs/technical/ai-model-runtime.md`.

## Two repositories

```text
dragon-jump-remaster/          PLaiGROUND/
  godot/                         src/            Python drivers
    src/training/                src/wrappers/   socket + Gymnasium + SB3
    src/player/controllers/      docs/           this documentation
    ...                          runs/           generated artifacts
  docs/                          environments/   game binary / macOS shim
```

The game owns the simulation, observation construction, reward, and reset
semantics. PLaiGROUND owns the learner, the data collection loop, and the
export/evaluation tooling. The boundary is a local TCP JSON protocol.

## Runtime data flow

```text
Godot training scene
  TrainingWorld × N (isolated SubViewports)
        │
  PlayerAITrainingController.get_obs() / get_reward() / get_done()
        │
  synchronizer.gd  ── TCP JSON ──▶  wrappers/godot_env.py (GodotEnv)
                                         │
                    ┌────────────────────┴─────────────────────┐
                    │                                          │
            SingleAgentGodotEnv                       MultiAgentGodotVecEnv
            (Gymnasium, one process/world)            (N agents/process, SB3 VecEnv)
                    │                                          │
              SB3 PPO / DQN                              SB3 PPO
```

- **`GodotEnv`** is the low-level client. It launches the game process, opens a
  socket, sends `reset`/`step`/`call` messages, and decodes observations and
  info. It can host multiple agents (NEAT, 100-agent evaluation).
- **`SingleAgentGodotEnv`** exposes one training world as a Gymnasium
  environment for SB3. It owns the start pool, the optional reset-time start
  jump, observation dropout, optional trajectory saving, and the 5-tuple
  `terminated/truncated` contract. Start-pool entries can be canonical, a
  cell+facing teleport, a NEAT-prefix replay, or a full state snapshot. Only
  prefix replay and state snapshots preserve the run clock; a cell reset zeroes
  it. See [`audit/2026-10-28-start-pool-clock-bias.md`](audit/2026-10-28-start-pool-clock-bias.md).
- **`MultiAgentGodotVecEnv`** exposes N agents in one Godot process as N virtual
  SB3 environments, using the game's per-agent `reset(agents=...)` support.
- **`StableBaselinesGodotEnv`** is the older Gym wrapper kept for tutorials /
  legacy notebooks. Active drivers use the classes above.

## Training families

| Family | Driver(s) | What it learns | Action source |
|---|---|---|---|
| NEAT | `07_neat_agent.py`, `08_neat_all_levels.py`, `09_neat_lineage.py` | Population of small networks; fitness = game reward | Genome per agent |
| PPO | `10_ppo_pilot.py` | One shared policy; on-policy | Policy |
| Distillation / DAgger | archived `12_ppo_distill.py`; deployed student | Supervised student from teacher probabilities/actions | Teacher labels |
| DQN | `41_dqn_route_spawns.py` | One shared Q-network; off-policy replay | Epsilon-greedy policy |

NEAT evaluates a whole population in one process and uses the game reward
directly. PPO uses one process per world (or N agents/process) and SB3's
vector-env contract. Distillation is supervised and does not use the game
reward. DQN is the newest branch and has no result yet.

## Observation contracts

The game emits a flat float vector. PLaiGROUND mirrors the schema in
`src/obs_spec.py`; the game owns the canonical definition in
`godot/src/ai/ai_observation_spec.gd`.

| Encoding | Size | Used by | Notes |
|---|---:|---|---|
| `legacy` | 58 | deployed 58/256 student, NEAT, DQN, most PPO baselines | 7×7 tile IDs + 9 player/route values. |
| `multi_hot` | 754 | 754 PPO teachers/students | 7×7×15 semantic grid + 19 player/route values. |
| `multi_hot_sensor` | 794 | 794 sensor PPO lineage | 754 + 20 head-ray distances + 20 hit flags. |
| `raw` | 342 | validation only; Python encoder pending | 7×7 tile IDs + tile states, player state, 8-way collisions, 8-way flow. |

> **Observation change (2026-10-28, in progress):** the `raw` encoding is
> implemented and validated (1-1 and 1-8); Python feature encoders and the
> reset/powerup restore are still pending. The 58 runtime encoder stays in
> Godot; the 754/794 layouts above remain the current reference until the
> cutover. See
> [`audit/2026-10-28-raw-state-observation-decision.md`](audit/2026-10-28-raw-state-observation-decision.md).

Action space is binary `{idle, jump}`. The default control cadence is
`action_repeat=4`: one decision every four physics frames, 15 decisions/s at
60 Hz. `--speedup` scales wall-clock physics ticks/time scale without changing
per-tick simulation or run times.

See [`reference/observation-contracts.md`](reference/observation-contracts.md)
for the field tables and reset/reward contracts.

## Reset semantics

The game's training reset supports three useful forms:

1. **Canonical** — the level's normal start.
2. **Cell + facing** — `{"cell": [x, y], "facing": 1|-1}`. Only position and
   facing are restored. The run clock starts at zero, so a mid-route start
   reads `gold_ratio = 3` and is not comparable to a full run. Training reset
   force-starts the run; there is no idle state or forced first jump.
3. **Grounded state snapshot** — an optional `state` dictionary with position,
   velocity, state name, `run_time`, `best_run_time`, `run_active`, and
   `started_walking`. The game supports this for grounded, powerup-free states;
   jump states, powerup stacks, and dissolve timers are not supported. This is
   the preferred form for mid-route DQN/PPO checkpoints because the run clock is
   preserved.

The clean-start target extends this contract to one flat reset dict: the
physical fields and source-checkpoint `run_time` are top-level, and the general
dynamic `world_state` snapshot (powerups, dissolve/moving/reset objects) is
nested. There is no separate `state`/`info` reset split; the observation encoder
ignores the restore metadata. Powerups are restored into the player stack while
their pickup nodes remain held/despawned. NEAT extraction/pooling rejects a
candidate that cannot be reproduced; Godot applies the reset or returns an
error. This extension is implemented and smoke-tested; the 1-8+ route and
promotion gates remain pending. See
[`audit/2026-10-28-clean-start-state-capture.md`](audit/2026-10-28-clean-start-state-capture.md).

The low-level wrapper also supports **prefix replay**: reset to canonical, then
replay a recorded action prefix before handing control to the policy. This
preserves the full run clock without a state snapshot, but it uses recorded
actions during reset (not as imitation labels). Training reset now force-starts
the run, so old prefix pools should be re-recorded; clean state starts replace
prefix replay for clean-start training.

Training death is **terminal at the hazard**: the game freezes the player in the
Die state, the controller applies `-50`/`done`, and the next reset uses the
normal sampled start (never canonical). Normal play keeps the death cinematic.

## Reward

The game computes one simplified five-term reward (2026-10-28):

```text
progress + efficiency + time + finish + survival
+1.0*(C(s)-0.99*C(s')) -0.02*dwasted
-3.75*delta_run_time*(1-gold_ratio) +500 finish -50 death
```

`gold_ratio` is clamped to `[0, 3]`; `best_run_time` is telemetry only. The old
`default`/`time`/`time_speed` profiles and `--reward_profile` are gone/ignored.
`info` also exposes `gold_ratio`, `flow_cost`, `flow_in_bounds`, and
`wasted_air_frames` for Python custom rewards. See
[`reference/reward-contract.md`](reference/reward-contract.md) and
[`known-issues.md`](known-issues.md) § DQN for the mid-route clock caveat.

## Run artifacts

A typical run directory:

```text
runs/<run>/
  monitor.csv                  SB3 VecMonitor episode summary
  episodes.csv                 one row per episode (PPO, start kind/source)
  checkpoints/                 model checkpoints
  eval*/summary.json           deterministic evaluation aggregates
  eval*/results.json           per-start results
  godot/                       raw Godot stdout/stderr per process
  userdata/                    isolated HOME/XDG trees per process
  tensorboard/                 SB3 logs (PPO/DQN)
```

`runs/` is gitignored. See
[`reference/run-artifacts.md`](reference/run-artifacts.md) for the current
inventory and what to open first.

## Environment setup

- Python 3.13 + `.venv`; dependencies in `pyproject.toml` / `requirements.txt`.
- The game reads its training arguments from the command line; PLaiGROUND
  launches it with `--port`, `--level`, `--nb_agents`, `--env_seed`,
  `--action_repeat`, `--speedup`, and optional observation/reward flags.
- **macOS:** `environments/DragonJump.app/Contents/MacOS/DragonJump` is a source
  shim that execs `/Applications/Godot.app/...` with
  `--path ../dragon-jump-remaster/godot`. It is not an exported app. The README's
  “Linux/Windows only” note is about the upstream project; the local shim works.
- **Linux/Windows:** place an exported `DragonJump` binary in
  `environments/` (see the README). Docker is the portable path.
- Each Godot process needs a unique port. Port collisions look like hangs; the
  game's own log may stop at `TRAINING_SPEEDUP` with no error.

## Where to change what

| Change | Location |
|---|---|
| New training algorithm / driver | add `src/NN_name.py`, document in `docs/experiments/`, update `src/README.md` + `docs/status.md` |
| Socket protocol / info decoding | `src/wrappers/godot_env.py` + `tests/test_info_protocol.py` |
| Observation schema mirror | `src/obs_spec.py` (and the game's `AIObservationSpec` on the game side) |
| Reward/observation behavior | game repo, not here; update the game vault and this repo's reference docs |
| Run evaluation | reuse `src/35_eval_ppo_strata.py` patterns; do not launch one process per spawn |
| Deployed model | export JSON from the student checkpoint; install in the game only after the gate |
