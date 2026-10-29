# 0111 — 1-8 canonical force-start diagnostic

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Purpose

Check whether the remaining canonical gap is caused by the normal-start delay.
The config enables `force_start_jump` for the canonical episode.

## Result

The selected 1.949912M checkpoint finishes the force-started canonical episode
at **7.8167 s**, slower than the normal-start **4.5167 s**. The force-start
policy takes a different route, so the canonical gap is policy behavior rather
than a fixed start-delay offset. This diagnostic is not used as a gate.

## Artifacts

```text
runs/ddqn_1_8_canonical_repair/eval/force_start/summary.json
runs/ddqn_1_8_canonical_repair/eval/diag_canonical.log
```
