# 0124 — canonical retention check for the 1-9 single-route expert

Status: complete  
Started: 2026-10-30  
Completed: 2026-10-30

Runs deterministic canonical evaluation for **1-1 through 1-9** with the final
1-9 checkpoint:

```text
runs/ddqn_1_9_single_route/dqn/dqn_final.zip
```

This is a retention curiosity check, not a clean-start gate. It answers whether
the 1-9-only fine-tune still clears each level from canonical start.

## Result

| Level | Result | Time | Gate | Verdict |
|---|---|---:|---:|---|
| 1-1 | finish | 2.5167 s | 2.5944 s | pass |
| 1-2 | timeout | — | 3.2376 s | fail |
| 1-3 | finish | 6.0167 s | 3.3816 s | finish, gate fail |
| 1-4 | timeout | — | 4.4856 s | fail |
| 1-5 | death | — | 3.2088 s | fail |
| 1-6 | death | — | 4.28 s (explicit) | fail |
| 1-7 | timeout | — | 5.2728 s | fail |
| 1-8 | timeout | — | 4.5144 s | fail |
| 1-9 | finish | 2.9500 s | 3.0648 s | pass |

Summary: **3/9 levels finish** (1-1, 1-3, 1-9); **2/9 pass the provisional
gate** (1-1, 1-9). The 1-9 fine-tune does not retain the old-level campaign.
