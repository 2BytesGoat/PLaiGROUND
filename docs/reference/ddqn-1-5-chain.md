# 1-5 DDQN chain — selected two-model candidate

Status: selected for evaluation, not installed in the Godot runtime  
Built: 2026-10-29

This is the 1-5 extension of the lean_stick DDQN chain. The 1-5 policy and the
old-level policy are separate checkpoints; the game has no level ID, so the
caller selects the checkpoint by level.

## Models

| Level | Checkpoint | Deterministic time | User time | Delta |
|---|---|---:|---:|---:|
| 1-1 | `runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip` | 1.5667 s | 1.66 s | -0.093 s |
| 1-2 | same | 3.1500 s | 2.98 s | +0.170 s |
| 1-3 | same | 3.2667 s | 3.16 s | +0.107 s |
| 1-4 | same | 4.4667 s | 4.38 s | +0.087 s |
| 1-5 | `runs/ddqn_expert_replay_1_5_4frame/dqn/dqn_final.zip` | 3.1667 s | 3.11 s | +0.057 s |

All five levels finish, and every delta is inside the 0.2 s band. The 1-5
result was repeated twice with the same time.

## Contract

```text
observation:        lean_stick 191 -> 957
action repeat:      4 frames (15 Hz)
max decisions:      150
reassert held jump: true
preprocessing:      runs/ddqn_fresh_1_1_lean_stick/preprocessing.pkl
game:               ../dragon-jump-remaster, prepare-for-expo, fe7821d
```

The 1-5 checkpoint is the 0085 final (1.655M). It was trained from the 1.555M
chain model with the 16-route 4-frame 1-5 expert buffer; the 4.68 s route
cluster was included. The old-level checkpoint is the 1.555M 0073/0079 chain.

## Evaluation

1-5, deterministic:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0086_eval_ddqn_1_5_4frame_0085 \
  --model runs/ddqn_expert_replay_1_5_4frame/dqn/dqn_final.zip \
  --tag chain_1_5
```

The config also evaluates 1-4..1-1; those results are expected to be bad on the
1-5 model and should be ignored. The 1-5 row is the selected result.

Old chain, deterministic:

```bash
.venv/bin/python tools/evaluate.py \
  --config experiments/0079_show_chain_1_1_to_1_4_0073 \
  --model runs/ddqn_expert_replay_1_4_lean_stick/dqn/checkpoints/ddqn_1555000_steps.zip \
  --tag chain_old_levels
```

## Rejected single-model attempt

0088 started from the 1.655M 1-5 checkpoint and fine-tuned for 25k steps with
2 x 1-5 + 2 x 1-4 + 2 x 1-3 + 2 x 1-2 and a 1.555M all-level anchor. Its final
gave 1-5 **3.5000 s** (outside the 3.41 s outer gate), 1-4 timeout, 1-3
3.2000 s, 1-2 3.0167 s, and 1-1 1.5667 s. It is not selected. A future
single-model attempt needs a frozen 1-5 head or a dedicated 1-4 expert buffer;
repeating the same fine-tune is not expected to help.

## Artifacts

```text
runs/ddqn_expert_replay_1_5_4frame/eval/eval_final/summary.json
runs/ddqn_expert_replay_1_5_4frame/eval/eval_final_repeat/summary.json
runs/chain_check_1_1_to_1_4_1555k/eval/watch_1555k_1x/summary.json
runs/ddqn_retention_1_5_4frame/eval/eval_final/summary.json
```

## 1-6 is a separate per-level model

The 1-5 model is not a 1-6 model. The selected 1-6 expert is the 0099
**1.815M** checkpoint:
`runs/ddqn_1_6_start_repair/dqn/checkpoints/ddqn_1814952_steps.zip`. It
finishes **32/32** NEAT feasible starts with 0 deaths/timeouts at a 6.0417 s
median, and is canonical **4.6000 s** (repeat-stable). The 0091 1.705M model
remains the canonical reference at 4.5667 s but covers only 11/32 starts. The
4.28 s speed gate is still open. The final product is one shared student A
distilled from the per-level experts; this chain is the reliable fallback and
teacher set. See
[`../audit/2026-10-29-1-6-ddqn-investigation.md`](../audit/2026-10-29-1-6-ddqn-investigation.md)
and
[`../audit/2026-10-29-expert-portfolio-and-student-decision.md`](../audit/2026-10-29-expert-portfolio-and-student-decision.md).
