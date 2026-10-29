# 0113 — evaluate the 1-8 canonical fast-route fine-tune

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Purpose

Evaluate the 0112 canonical-route checkpoints on 1-8 canonical plus the
40-state strict-clean pool.

## Result

```text
checkpoint      canonical   clean states   deaths   timeouts   best
1.952408M       4.4667      39/40          0        1          4.5167
1.954904M       4.4667      39/40          0        1          4.5667
```

Both checkpoints pass the canonical provisional gate. The remaining clean
timeout is `[4,21]` facing `-1` from route `1501831d...`. The 1.952408M
checkpoint is the base for the 0114 targeted repair.

## Artifacts

```text
runs/ddqn_1_8_canonical_route/eval/ddqn_1952408_steps/summary.json
runs/ddqn_1_8_canonical_route/eval/ddqn_1954904_steps/summary.json
```
