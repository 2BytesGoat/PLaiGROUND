# 0088 — 25k retention fine-tune of the 1-5 policy

Status: complete  
Started: 2026-10-29

## Question

Can the 1-5 policy be kept while restoring 1-4 and 1-2 in a **single** model?

The 0085 final model gives 1-5 **3.1667 s**, 1-3 3.1667 s, and 1-1 1.5667 s,
but 1-4 fails and 1-2 regresses to 5.35 s. The 1-4 schedule had only one env and
the old anchor had no 1-4 transitions. This run starts from 1.655M and spends
25k steps on a balanced retention mix.

## Setup

```text
start:          1.655M 0085 final
levels:         2 x 1-5, 2 x 1-4, 2 x 1-3, 2 x 1-2
action repeat:  4
learning rate:  5e-5
exploration:    0.10 -> 0.05 over 50% of the run
expert:         1-5 16-route buffer (4.68 s cluster included)
old anchor:     1.555M all-level 4-frame RareReplayBuffer
                5,105 episodes, median 3.00 s, best 1.5667 s, 6 timeouts
margin weights: 0.2 expert, 0.2 old anchor
budget:         25,000 steps, checkpoints at 12.5k and 25k
```

The 1.555M all-level anchor is the key change: it contains actual 1-3 and 1-4
transitions, unlike the 1.53M anchor used by 0085.

## Gates

```text
1-5:       finish, <= 3.41 s (0085 fallback: 3.1667 s)
1-4:       finish, <= 4.68 s
1-3:       finish, <= 3.46 s
1-2:       finish, <= 3.28 s
1-1:       finish, <= 1.96 s
```

If the fine-tune cannot hold 1-5, the two-model chain stays the deliverable:
0085 final for 1-5 and 1.555M for 1-1..1-4.

## Run

```bash
.venv/bin/python scripts/reinforcement/03_ddqn.py \
  --config experiments/0088_ddqn_retention_1_5_4frame
```

## Output

```text
runs/ddqn_retention_1_5_4frame/
  dqn/checkpoints/ddqn_1667500_steps.zip
  dqn/checkpoints/ddqn_1680000_steps.zip
  dqn/dqn_final.zip
  eval/
```

## Result

The 12.5k and 25k checkpoints were evaluated deterministically at 4-frame
control:

```text
              12.5k      25k (final)
1-5           0/1 fail   3.5000 s
1-4           timeout    timeout
1-3           3.2500 s   3.2000 s
1-2           3.0167 s   3.0167 s
1-1           1.5667 s   1.5667 s
```

The fine-tune restored 1-2 and 1-3 but did **not** recover 1-4, and the 1-5
policy drifted from 3.1667 s to 3.5000 s (0.09 s over the 3.41 s outer gate).
The single-model retention attempt is therefore rejected. The two-model chain
stays the deliverable: 0085 final for 1-5, 1.555M for 1-1..1-4. If a single model
is required later, freeze the 1-5 head or give 1-4 its own expert buffer rather
than repeating this fine-tune.
