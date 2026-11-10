# 0133 — 1-14 canonical DDQN nudge

Status: **negative; discarded**  
Started: 2026-10-31  
Init: `runs/distill_campaign_student/students/cand_verified_onehot957_512_false_ddqn2_w11x2_s2/student.pt`  
Parent: [`0130`](../0130_distill_campaign_student/SHIPPING_CANDIDATE.md)

## Result

The warm-started DDQN lost the 1-14 finish immediately. The init student
finishes 1-14 at **8.1833 s** under false held-jump semantics; checkpoints
10k–100k all time out or die. The Q-head warm start is not behavior-preserving
because the student's logits are not calibrated action-values. Do not repeat
this setup without a KL/behavior anchor or a frozen trunk.

```text
student init:  8.1833 s, finish
10k .. 100k:   0/1 finish (mixture of deaths and timeouts)
```

The 1-14 repair moved to the supervised DAgger track; see
[`0134`](../0134_dagger_1_3_student_repair/README.md).

## Question

Can a short canonical-only DDQN nudge from the selected shared student close the
1-14 strict miss? The student finishes at **8.1833 s**; the target is
**6.9333 s**. 0131/0132 reached 7.45 s at 90k under legacy semantics but the
final checkpoint timed out.

## Setup

- 8 canonical 1-14 environments, `reassert_held_jump=false`
- 100k DDQN steps, LR 5e-5, exploration 0.1 -> 0.02
- Warm start from the selected student (957 -> 512 -> 512 -> 2)
- Evaluate 50k/70k/90k/100k, keep only a checkpoint <= 6.9333 s that still
  finishes all 16 levels

## Run

```bash
.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0133_ddqn_1_14_student_canonical_nudge
```

## Gate

- 1-14 canonical finish at or below **6.9333 s**
- all-16 canonical finish remains 16/16
- strict count does not drop below 12/16

Discard the run if no checkpoint passes.
