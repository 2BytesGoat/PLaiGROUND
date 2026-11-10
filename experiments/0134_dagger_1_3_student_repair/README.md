# 0134 — 1-3 DAgger repair of the shared student

Status: **candidate selected; 13/16 strict, 80.2833 s sum; exported to game working tree (ungated)**  
Date: 2026-10-31  
Base: `runs/distill_campaign_student/students/cand_verified_onehot957_512_false_ddqn2_w11x2_s2/student.pt`  
Candidate: `runs/distill_campaign_student/students/cand_dagger_1_3_13of16/student.pt`  
Game export: `../dragon-jump-remaster/godot/src/resources/ai/student_lean_stick_512.json`  
Parent: [`0130`](../0130_distill_campaign_student/SHIPPING_CANDIDATE.md)

## Question

The selected shared student is 16/16 canonical finish but 12/16 strict. The
largest miss is 1-3: **5.2333 s** against the **3.2667 s** teacher. The audit
showed 93.8% canonical action match on 1-3, so the failure is a small number of
divergent decisions, not a model-capacity problem. Can one DAgger round with the
0073 teacher close it without retraining the campaign?

## Method

1. Collect 698 DAgger transitions on 1-3 with the selected `0073` 1.555M expert
   as the teacher, 8 agents, `epsilon=0.1`, true held-jump semantics.
2. Merge that shard into the verified 0130 dataset.
3. Fine-tune the selected student for 15 epochs at LR `3e-5`, level weight
   `1-3=2`, seed 2, MPS. Best validation epoch was 11.

```bash
.venv/bin/python tools/collect_dagger.py \
  --level 1-3 \
  --student runs/distill_campaign_student/students/cand_verified_onehot957_512_false_ddqn2_w11x2_s2/student.pt \
  --teacher runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip \
  --teacher-kind dqn \
  --preprocessing runs/ddqn_fresh_1_1_lean_stick/preprocessing.pkl \
  --output-dir runs/dagger_1_3_student_canonical \
  --agents 8 --epsilon 0.1 --max-decisions 150 --speedup 12 --port 32500

.venv/bin/python scripts/supervised/03_distill_student.py \
  --data-dir runs/distill_campaign_student/dataset_dagger_1_3 \
  --output-dir runs/distill_campaign_student/students/dagger_1_3_gentle_a \
  --input-kind onehot957 --hidden 512 512 \
  --init-student runs/distill_campaign_student/students/cand_verified_onehot957_512_false_ddqn2_w11x2_s2/student.pt \
  --seed 2 --level-weight 1-3=2 --epochs 15 --learning-rate 0.00003 \
  --patience 5 --device mps
```

## Canonical result

Deterministic canonical runs, `max_decisions=180`; 1-14 uses
`reassert_held_jump=false`, all other levels use true semantics.

| Level | Target | Candidate | Strict |
|---|---:|---:|---|
| 1-1 | 1.5667 | 1.5667 | pass |
| 1-2 | 3.1500 | 3.1500 | pass |
| 1-3 | 3.2667 | **3.2667** | **pass** |
| 1-4 | 4.4667 | 4.4667 | pass |
| 1-5 | 3.1667 | 3.1667 | pass |
| 1-6 | 4.6000 | 4.6000 | pass |
| 1-7 | 5.2333 | 5.2333 | pass |
| 1-8 | 4.4667 | 4.9833 | miss +0.5166 |
| 1-9 | 2.9500 | 2.9500 | pass |
| 1-10 | 5.3667 | 5.3667 | pass |
| 1-11 | 9.8167 | 9.8167 | pass |
| 1-12 | 5.9667 | 6.0167 | miss +0.0500 |
| 1-13 | 6.2667 | 6.2667 | pass |
| 1-14 | 6.9333 | 7.4500 | miss +0.5167 |
| 1-15 | 5.8500 | 5.8500 | pass |
| 1-16 | 6.1333 | 6.1333 | pass |

```text
finish:    16 / 16
strict:    13 / 16   (baseline 12 / 16)
sum:       80.2833 s (baseline 82.4833 s)
```

The 1-3 fix is exact. 1-14 also improved from 8.1833 s to 7.4500 s. The
remaining strict misses are 1-8 (+0.5166), 1-12 (+0.0500), and 1-14 (+0.5167).
1-8 was already a strict miss in the baseline; the fine-tune made it slower but
did not change the strict count there.

## Negative follow-ups

- A 100k canonical DDQN warm start on 1-14 (`0133`) timed out at every
  checkpoint; the Q-head warm start is not behavior-preserving.
- Stronger 1-8 level weights and a frozen-trunk head-only fine-tune did not
  recover the original 1-8 time; the regression lives in the shared trunk.
- A combined 1-3/1-12/1-14 anchored fine-tune fixed 1-3 but regressed 1-14 and
  1-16; it was discarded.
- A one-frame 1-12 anchored fine-tune did not move 1-12.

## Export

```bash
.venv/bin/python tools/export_student_json.py \
  --checkpoint runs/distill_campaign_student/students/cand_dagger_1_3_13of16/student.pt \
  --preprocessing runs/ddqn_fresh_1_1_lean_stick/preprocessing.pkl \
  --output ../dragon-jump-remaster/godot/src/resources/ai/student_lean_stick_512.json \
  --model-name student_lean_stick_512 \
  --reassert-held-jump --reassert-held-jump-false-level 1-14 \
  --golden-data-dir runs/distill_campaign_student/dataset_verified_false_ddqn \
  --golden-output ../dragon-jump-remaster/godot/tests/ai/fixtures/student_lean_stick_512_golden.json
```

The previous export is retained as
`student_lean_stick_512_baseline.json` in the same directory. The exporter
verified encoder parity (`encoder_max_abs=0`) and forward parity
(`forward_max_abs=9.54e-07`, argmax match 1.0). The game contract suite
`tests/ai/test_mlp_policy.gd` passes 8/8 with the new weights and golden case.

## Next

1. Do not overwrite the committed 58-value fallback. The new export is in the
   game working tree only; it has not passed the in-game all-16/powerup gate.
2. The remaining work is the audit track: trace game vs Python, add temporal
   scalars (`hold_age`, `fall_age`, `last_action`), and add deliberate powerup
   demonstrations before another shared retrain.
3. If another quick repair is attempted, use DAgger with a KL anchor and keep
   the 1-3 shard; do not use DDQN warm starts from the student.
