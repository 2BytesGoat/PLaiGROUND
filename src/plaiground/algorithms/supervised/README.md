# Supervised algorithms

Shared supervised training code lives here:

```text
trainer.py             # optimizer/epoch/checkpoint loop
behavior_cloning.py    # observation -> action
dagger.py              # online teacher relabeling
distillation.py        # teacher logits/probabilities -> student
trajectory.py          # sequence/window training
```

`data/trajectories.py` will own the PyTorch `Dataset`/`DataLoader` adapters for
offline supervised workflows.
