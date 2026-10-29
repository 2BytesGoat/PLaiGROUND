# 0100 — evaluate the 1-6 start-repair fine-tune

Status: complete  
Started: 2026-10-29  
Completed: 2026-10-29

Deterministic canonical + 32-state evaluation for 0099. The selected checkpoint
is evaluated twice with the same result:

```text
model:      runs/ddqn_1_6_start_repair/dqn/checkpoints/ddqn_1814952_steps.zip
canonical:  4.6000 s, 0 deaths, 0 timeouts
32 states:  32/32, 0 deaths, 0 timeouts, median 6.0417 s
spawn_ok:   32/32
```

Summary:
`runs/ddqn_1_6_start_repair/eval/selected_repeat/summary.json`.

See [`../0099_ddqn_1_6_start_repair/README.md`](../0099_ddqn_1_6_start_repair/README.md).
