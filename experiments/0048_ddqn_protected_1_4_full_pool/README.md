# 0048 — DDQN 1-4 with the 0046 pool, lean 188

Status: stopped diagnostic

The 0046 pool improved the 1-4 start coverage, but this branch used `lean` 188,
which has none of the three hold/ceiling signals (`on_ceiling`, `jump_held`,
`stick_active`). It was stopped at ~2.626M and preserved as the 188/new-pool
baseline.

## 2.525M evaluation

```text
1-4 canonical: 0/1, timeout
1-4 train:     1/11 at 6.7833 s
1-4 holdout:   0/2
spawns:        11/11 train, 2/2 holdout
```

This is still weak. The next branch is 0053, which uses `lean_stick` 191 and
loads the 188 model weights into an expanded first layer.
