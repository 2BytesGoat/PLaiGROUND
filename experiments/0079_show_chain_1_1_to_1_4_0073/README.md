# 0079 — windowed canonical chain check, 1-1 → 1-4 (1.555M)

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

## Question

Does the selected 0073 **1.555M** checkpoint still finish the canonical route on
every level from 1-1 through 1-4, with the game window visible, on the current
game tree?

## Setup

```text
model:        runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip
observation:  lean_stick 191 -> 957
game:         ../dragon-jump-remaster, prepare-for-expo, HEAD fe7821d (clean)
levels:       1-1, 1-2, 1-3, 1-4 (canonical only; one windowed process per level)
action repeat: 4
max decisions: 150
speedup:      1.0
```

Run:

```bash
make watch   # 1-1 .. 1-4, windowed, 1x
```

or directly:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0079_show_chain_1_1_to_1_4_0073 \
  --model runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip \
  --tag watch_1555k_1x --show
```

Artifacts: `runs/chain_check_1_1_to_1_4_1555k/eval/watch_1555k_1x/`
(`summary.json`, `canonical-<level>.csv`, one Godot log per level).

## Result

```text
1-1 canonical: 1/1 at 1.5667 s   deaths 0   efficiency 1.000   flow 1.0
1-2 canonical: 1/1 at 3.1500 s   deaths 0   efficiency 0.557   flow 1.0
1-3 canonical: 1/1 at 3.2667 s   deaths 0   efficiency 1.000   flow 1.0
1-4 canonical: 1/1 at 4.4667 s   deaths 0   efficiency 1.000   flow 1.0
```

All four levels finish on the first attempt with no deaths and no timeouts.
The times match the 0073 record exactly (1.5667 / 3.1500 / 3.2667 / 4.4667), so
the nine game commits between the recorded `adb9278` and the current `fe7821d`
(observation/reward and AI-control fixes) do not change the canonical chain.

## Caveat

`speedup` values below 1.0 are silently clamped to 1x: the game's
`_apply_speedup()` uses `clampf(..., 1.0, 16.0)` and returns early at <= 1.0.
The 0078 manifest's `speedup = 0.5` therefore ran at normal speed. This run
uses `1.0` explicitly. See `docs/known-issues.md` item 25.
