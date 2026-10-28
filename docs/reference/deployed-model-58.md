# Deployed AI model: 58-value student

Status: deployed and frozen for the demo  
Built: 2026-10-26  
Runtime artifact: `godot/src/resources/ai/student_obs58_256.json`

> Current state: [`../status.md`](../status.md) · Docs index:
> [`../index.md`](../index.md) · Run artifacts:
> [`run-artifacts.md`](run-artifacts.md).

This is the model currently used by `WATCH AI` and campaign AI in Dragon Jump.
It is one level-agnostic policy: there is no level ID and no per-level model
selection. Some game docs quote 1-1 as **2.35 s**; the deployed evaluation in
this note is **2.333 s**. Re-run one clean bench before freezing a ship number.

## At a glance

```text
7x7 terrain + 9 player/route values
                |
                v
       58 -> 256 -> 256 -> 2
                |
                v
          idle / jump
```

| Property | Value |
|---|---|
| Input | 58 values |
| Hidden layers | 2 × 256 ReLU |
| Output | 2 actions: idle, jump |
| Parameters | 81,410 |
| Decision rate | every 4 physics frames (15 Hz) |
| Level conditioning | none |
| Runtime | pure-GDScript MLP |
| JSON size | about 1.7 MB |

## What the model sees

The input is a local view of the level plus the player's current state.

### 49 terrain values

A 7×7 grid of tile IDs centred on the dragon:

```text
7 x 7 = 49 values
```

Each cell is the tile ID at that position. This is a local window, not the
whole level.

### 9 player and route values

| Input | Values | Meaning |
|---|---:|---|
| Exit direction | 2 | normalized x/y direction to the exit |
| Flow distance | 1 | normalized distance-to-exit scalar |
| Velocity direction | 2 | normalized x/y velocity direction |
| On floor | 1 | 0/1 |
| On wall | 1 | 0/1 |
| Jump peak | 1 | percentage through the jump arc |
| Has powerup | 1 | 0/1 |

Total:

```text
49 terrain + 9 player/route = 58 inputs
```

The policy does not receive an explicit level name, a one-hot level ID, or a
campaign index.

## Network

```text
input (58)
   |
   v
Linear 58 -> 256
ReLU
   |
   v
Linear 256 -> 256
ReLU
   |
   v
Linear 256 -> 2
   |
   v
argmax
   |
   v
idle / jump
```

The exported JSON stores the three weight matrices and biases, the layer
shapes, the activation, the input contract, and `action_repeat=4`.

## How it was trained

### 1. Build per-level PPO teachers

A PPO teacher was trained for each campaign level, 1-1 through 1-16. Each
teacher was evaluated deterministically before being used as a label source.

The teacher policies are not shipped. They exist only to generate training
data for the single student.

### 2. Collect teacher demonstrations

For every level:

- run the teacher with sampled actions;
- use 100 parallel agents;
- record two episodes per level;
- store the observation, the teacher's action probabilities, and the sampled
  action.

The collection produced **263,519 samples** across all 16 levels.

The agent owned the first jump during this historical collection; there was no
forced start action. Training reset now force-starts the run (2026-10-28); this
does not change the shipped inference model.

### 3. Train the base student

The collected observations train a single supervised student:

```text
58 -> 256 -> 256 -> 2
25 epochs
batch size 512
loss: cross-entropy against teacher action probabilities
```

The base student reached **98.2% validation action agreement** and cleared all
16 levels at 100/100 agents.

### 4. DAgger correction

The student was run on its weakest levels, and the teachers labelled the states
the student actually visited. This corrects the mismatch between teacher states
and student states.

The selected export was trained after DAgger corrections on **1-3 and 1-7**.
A later 1-11 DAgger pass produced an alternative model; it was not selected
because it regressed 1-3.

### 5. Export and install

The selected checkpoint was converted to JSON by:

- flattening each weight matrix and bias in `[out, in]` order;
- writing the 58-input contract;
- verifying the JSON forward pass against PyTorch on random observations;
- checking argmax agreement;
- installing the JSON in Godot.

The exported JSON is the artifact the game loads. The runtime does not use
Python, PyTorch, or ONNX.

## How long it took

These are artifact timestamps from the distillation run on the local machine.
Teacher training happened before this window and is not included.

| Time | Step |
|---|---|
| 14:09:43 | Distillation run starts |
| 14:09:56–14:14:22 | Collect all 16 teacher datasets |
| 14:14:50 | Base student trained |
| 14:14:50–14:17:10 | Base evaluation |
| 14:17:10–14:17:25 | DAgger runs on 1-3 and 1-7 |
| 14:18:18 | Selected student trained |
| 14:18:18–14:20:16 | Selected student evaluation |
| 14:20:24 | JSON exported and installed |

**Collection to installed JSON: about 10 minutes 41 seconds.**

The later 1-11 DAgger pass ran at 14:27 and produced the rejected alternative,
so it is outside the selected model’s build window.

## Deployed evaluation

Deterministic evaluation, 100 agents per level, no forced first jump:

```text
16 / 16 levels cleared
100 / 100 agents per level
0 deaths
campaign best-time sum: 78.583 s
```

| Level | Best time | Level | Best time |
|---|---:|---|---:|
| 1-1 | 2.333 s | 1-9 | 2.900 s |
| 1-2 | 3.017 s | 1-10 | 4.467 s |
| 1-3 | 3.233 s | 1-11 | 8.250 s |
| 1-4 | 4.250 s | 1-12 | 5.967 s |
| 1-5 | 2.967 s | 1-13 | 6.267 s |
| 1-6 | 5.050 s | 1-14 | 6.800 s |
| 1-7 | 7.617 s | 1-15 | 6.117 s |
| 1-8 | 3.217 s | 1-16 | 6.133 s |

In-game headless bench: all 16 levels finish, zero deaths, about **78.8 s**
campaign sum.

## Runtime behaviour

At each decision:

1. the game builds the 58-value observation;
2. `MLPPolicy` runs the JSON MLP;
3. the larger output selects idle or jump;
4. the chosen action is held for 4 physics frames;
5. the controller repeats.

AI runs are non-scoring: they do not write practice times, medals, unlocks,
saves, ghosts, or leaderboard entries.
