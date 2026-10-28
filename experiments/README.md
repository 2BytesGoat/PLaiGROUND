# Experiments

Each experiment gets its own numbered directory:

```text
experiments/0001_ppo_clean_start_1_1/
├── README.md       # question, hypothesis, setup, result, decision
├── config.toml     # exact configuration used
└── results.md      # metrics and interpretation
```

Experiment IDs are independent of script numbers:

- `scripts/reinforcement/01_ppo.py` is the generic PPO entry point.
- `experiments/0001_ppo_clean_start_1_1` is one specific PPO run.
- A later PPO experiment gets the next experiment ID, not a new script.

Generated artifacts stay under `runs/<experiment_id>_<slug>/` and are
gitignored. Record the game commit, branch, and dirty state in both the
experiment config and `results.md`; the run directory additionally contains the
full metadata and any dirty game diff.

## Naming

```text
0001_<area>_<level>_<variant>
```

Use `0000` for a data-collection or baseline experiment when needed. Do not
reuse an experiment ID after it has produced a recorded result.
