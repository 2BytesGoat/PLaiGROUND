# 0129 — targeted 1-10 start repair

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

## Question

Can a targeted suffix buffer restore the three 0126 failures (`[8,25]`,
`[9,25]`, `[26,20]`) without losing the 22/25 coverage?

## Source

```text
base checkpoint:  runs/ddqn_1_10_feasible_starts/dqn/checkpoints/ddqn_2122840_steps.zip
repair starts:    runs/ddqn_1_10_feasible_starts/repair_starts.json (3 states)
expert suffix:    each exact start through route finish, repeated 4x
buffer:           runs/ddqn_1_10_feasible_starts/expert_1_10/repair_suffix_buffer.pkl
buffer size:      576 transitions (144 matched x 4)
budget:           10,000 steps, checkpoints every 2,500
wall time:        10 s
```

## Result

| Checkpoint | Canonical | Eval pool | Deaths | Timeouts |
|---|---:|---:|---:|---:|
| 2.125336M | 4.7000 | 17/25 | 3 | 5 |
| **2.127832M** | **4.7000** | **20/25** | **5** | **0** |
| 2.130328M | 6.9667 | 21/25 | 4 | 0 |
| 2.132824M | 6.9667 | 20/25 | 4 | 1 |

The repair reached the NEAT-best **4.7000 s** canonical at 2.125336M and
2.127832M, but coverage fell to 17-20/25. The three target starts still fail:
`[8,25]` and `[9,25]` die at 11 steps, and `[26,20]` dies at 31 steps. The
2.127832M model also loses `[17,11]` facing `-1` and `[21,20]` facing `1`.

This is the runbook's over-correction case: the fast route conflicts with
several coverage states, and stacking another repair is not justified. The
checkpoint is recorded as a **speed alternative**, not selected:

```text
runs/ddqn_1_10_start_repair/dqn/checkpoints/ddqn_2127832_steps.zip
canonical 4.7000 s (matches the fastest clean NEAT route)
coverage 20/25 (vs 22/25 for the selected 2.122840M checkpoint)
```

## Output

```text
runs/ddqn_1_10_start_repair/
  dqn/checkpoints/ddqn_2127832_steps.zip
  expert_1_10/repair_suffix_buffer.pkl
  eval/<tag>/summary.json
```
