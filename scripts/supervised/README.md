# Supervised entry points

Planned one-script-per-workflow layout:

```text
01_behavior_cloning.py
02_dagger.py
03_distillation.py
04_trajectory_training.py
```

Each script will be a thin wrapper over a module in
`src/plaiground/algorithms/supervised/`. There is no active supervised script in
the current clean-start PPO task.
