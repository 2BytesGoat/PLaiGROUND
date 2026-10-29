# 0112 — canonical fast-route suffix fine-tune for 1-8

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

The 0109 repair reached 40/40 clean states but canonical stayed at 4.5167 s.
The deterministic policy matched the fastest **4.4667 s** NEAT route for the
first 45 decisions, then diverged in six late decisions. Can a short
canonical-only suffix fine-tune close that gap?

## Data

```text
base:            runs/ddqn_1_8_canonical_repair/.../ddqn_1949912_steps.zip
route:           63278cb6... (4.4667 s, 66 actions)
expert subset:   runs/ddqn_1_8_canonical_repair/expert_1_8/canonical_fast_route.npz
expert buffer:   runs/ddqn_1_8_canonical_repair/expert_1_8/canonical_fast_buffer.pkl (66 transitions)
training pool:   runs/ddqn_1_8_canonical_repair/canonical_only_start_pool.json
budget:          5,000 steps; checkpoints every 2.5k
```

## Setup

```text
levels:         8 x 1-8
action repeat:  4
lr:             5e-5
exploration:    0.02 flat
expert TD:      1.0
behavior margin: 1.0 x 0.5
old anchor:     none
```

## Result

```text
checkpoint      canonical   clean states   deaths   timeouts   best
1.952408M       4.4667      39/40          0        1          4.5167
1.954904M       4.4667      39/40          0        1          4.5667
```

The suffix fine-tune fixed canonical speed (**4.4667 s**, under the provisional
4.5144 s gate) but lost the `[4,21]` facing `-1` start from route `1501831d...`.
The follow-up 0114 targeted repair fixes that one state while preserving the
canonical route.

## Artifacts

```text
runs/ddqn_1_8_canonical_route/dqn/checkpoints/ddqn_1952408_steps.zip
runs/ddqn_1_8_canonical_route/expert_1_8/canonical_fast_buffer.pkl
runs/ddqn_1_8_canonical_route/eval/ddqn_1952408_steps/summary.json
```
