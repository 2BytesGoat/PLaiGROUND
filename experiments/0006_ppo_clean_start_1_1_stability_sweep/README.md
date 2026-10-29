# 0006 — PPO stability config sweep

Status: complete; `8x1_n600_b480` selected for 100k confirmation  
Started: 2026-10-28
Completed: 2026-10-28

## Question

Which rollout/batch shape is most stable when continuing the selected 500k
clean-start checkpoint for 50,000 timesteps?

All variants resume the same 500k checkpoint and reuse the same preprocessor,
start pool, PPO learning rate, entropy bonus, and target KL. Only the rollout
shape and environment count change.

| Variant | Processes | `n_steps` | `batch_size` | Rollout |
|---|---:|---:|---:|---:|
| `8x1_n150_b120` | 8 | 150 | 120 | 1,200 |
| `8x1_n300_b240` | 8 | 300 | 240 | 2,400 |
| `8x1_n600_b480` | 8 | 600 | 480 | 4,800 |
| `16x1_n150_b240` | 16 | 150 | 240 | 2,400 |
| `16x1_n300_b480` | 16 | 300 | 480 | 4,800 |

Each variant ran for 50k new timesteps and saved its own output directory.
The winner, **8×1 / 600 / 480**, was extended for 100k from the 500k checkpoint
in experiment 0007. That 600k model is now the selected stable policy.

## Result

See [`results.md`](results.md) for deterministic canonical/train/holdout
comparisons. The old 8×1 / 150 / 120 control degraded to 4.9167 s canonical,
8/12 train, and 5/8 holdout. The 8×1 / 600 / 480 variant stayed at 2.15 s
canonical, 12/12 train, and 8/8 holdout, with the strongest training-monitor
stability.

## Outputs

```text
runs/ppo_stability_<variant>/
```

Evaluation is separate per variant:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0006_ppo_clean_start_1_1_stability_sweep/<variant> \
  --model runs/ppo_stability_<variant>/ppo_pilot_550k.zip
```
