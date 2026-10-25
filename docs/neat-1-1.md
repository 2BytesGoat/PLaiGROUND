# NEAT on Dragon Jump 1-1

Experimental NEAT training for the first campaign level. This is intentionally
separate from the notebooks and the Stable-Baselines training path.

## How it works

`src/07_neat_agent.py` uses the game's existing training environment:

- one Godot process runs `N` isolated agents
- agent `i` is driven by NEAT genome `i`
- every step returns one game reward per agent
- `genome.fitness` is the sum of the rewards already computed by
  `PlayerAITrainingController.get_reward()`

The game reward includes flow progress, jump-arc efficiency, time pressure,
the finish bonus, and the death penalty. There is no second fitness function.

The observation is the existing 58-value training observation; the action is
the existing binary jump action. Training uses the isolated `TrainingWorld`
agents, so it does not touch campaign saves, medals, ghosts, or leaderboards.

## Determinism and robustness

The game has no RNG in a campaign run. The same genome and input timing produce
the same episode, which makes reward fit for population selection.

Two practical details preserve that property:

1. **A fresh Godot environment is created for every NEAT generation.** The
   training controller keeps `_best_run_time` across episodes; a fresh process
   makes each generation use the same `INF` reference for the finish bonus.
2. **Timing variants are deterministic.** `--action-repeat` and
   `--start-delay` provide reproducible variation for validation. There is no
   random seed in the level itself; `--seed` only controls NEAT's population
   RNG.

Train on the primary timing first. Validate the best genome with different
action repeats/start delays before treating it as robust.

## Run

Install the extra dependency once (it is also listed in `pyproject.toml`):

```bash
uv pip install --python .venv/bin/python neat-python
# or, in a normal venv:
python -m pip install neat-python
```

From the `PLaiGROUND` repository:

```bash
python src/07_neat_agent.py \
  --generations 20 \
  --population 32 \
  --max-steps 500 \
  --seed 42 \
  --output-dir runs/neat_1_1
```

Outputs:

```text
runs/neat_1_1/
  best_genome.pkl
  history.csv
  neat-checkpoint-*
```

The examples redirect console output to `runs/neat_1_1/train.log` when a
persistent log is useful.

Play the best genome:

```bash
python src/07_neat_agent.py \
  --play runs/neat_1_1/best_genome.pkl \
  --max-steps 600
```

Add `--show` to open the game window.

## Display the population

Training can render every agent as a ghost in one shared scene:

```bash
# Fresh population: 24 agents, 5 generations, 2x wall-clock speed
python src/07_neat_agent.py \
  --generations 5 \
  --population 24 \
  --max-steps 400 \
  --speedup 2 \
  --show

# Resume the trained 32-agent population (checkpoint from the result below)
python src/07_neat_agent.py \
  --resume runs/neat_1_1/neat-checkpoint-20 \
  --generations 3 \
  --max-steps 400 \
  --speedup 2 \
  --show
```

`--speedup` now scales both the physics tick rate and `Engine.time_scale`
(clamped 1–16), so the simulation stays tick-identical and only wall-clock time
changes. The saved 1-1 genome finishes with the same 2.35 s / 154.90 reward at
1×, 2×, 4×, and 8×. The visible run above showed 14/32 agents finishing in each
generation.

Validation examples:

```bash
# Same policy, different action cadence
python src/07_neat_agent.py --play runs/neat_1_1/best_genome.pkl --action-repeat 2

# Same policy after a delayed first jump
python src/07_neat_agent.py --play runs/neat_1_1/best_genome.pkl --start-delay 10

# Held-out level sanity check (not used for selection)
python src/07_neat_agent.py --play runs/neat_1_1/best_genome.pkl --level 1-3
```

Because the environment is deterministic, repeating the same command should
produce the same run time and reward.

## Result (seed 42, 20 generations, population 32)

- Best fitness: **154.90**
- Best finish: **2.35 s**
- Best generation: **19**
- Finishes in the final generation: **14/32**
- Deaths: **0**
- Training wall time: **268.6 s**

The hand-tuned heuristic baseline for 1-1 is 3.95 s, so the NEAT policy is
faster on this map. Validation results for the saved genome:

| Variant | Result |
|---|---|
| `action_repeat=1` | finish, 2.35 s |
| `action_repeat=2` | finish, 2.35 s |
| `action_repeat=3` | finish, 2.383 s |
| `action_repeat=4` | finish, 2.95 s |
| repeat of `action_repeat=1` | finish, 2.35 s (same reward) |
| `start-delay=10` | finish, 2.35 s |
| `speedup=1/2/4/8` | finish, 2.35 s, 154.90 reward at every speed |
| held-out level 1-3 | timeout |

The 1-3 result is expected: this policy was trained only on 1-1 and is not a
campaign solver.

## All campaign levels

`src/08_neat_all_levels.py` trains one independent NEAT population per
campaign level (`1-1` … `1-16`). Training is headless and can use one Godot /
Python worker per port; checkpoints are saved every five generations. After
training, each checkpoint is replayed for one generation with the window open,
so the population's behaviour is visible as it evolved.

```bash
.venv/bin/python src/08_neat_all_levels.py \
  --levels all \
  --population 100 \
  --generations 10 \
  --max-steps 900 \
  --speedup 8 \
  --workers 4 \
  --checkpoint-interval 5 \
  --show-checkpoints \
  --output-dir runs/neat_all_100x10
```

Output layout:

```text
runs/neat_all_100x10/
  1-1/train.log
  1-1/history.csv
  1-1/best_genome.pkl
  1-1/neat-checkpoint-5
  1-1/showcase/gen-005/showcase.log
  ...
  summary.csv
  learning_curves.png
```

Useful flags:

- `--levels 1-2,1-5,1-16` for a subset
- `--no-show-checkpoints` for a faster headless sweep
- `--skip-training` to rebuild the summary/curves from existing histories
- `--force` to retrain levels that already have a complete history
- `--workers N` and `--base-port` for controlled parallelism

The 100-agent training process is CPU-bound, so `--speedup` above 8× does not
reduce wall-clock time appreciably on this machine. Parallel levels help more
than raising the speedup once the CPU is saturated.

## Environment requirements

- A Dragon Jump build with the training controller and 58-value observation
  contract.
- The `ENV_PATH`/`ENV_NAME` in `src/.config` must point at that build.
- `--action-repeat` is overridden to `1` by default. Do not use the shipped
  `ACTION_REPEAT=0`; the synchronizer computes a modulo by that value.
- `--speedup` requires a game build with the training speedup change. The game
  prints `TRAINING_SPEEDUP xN` when it is active.

On macOS, `environments/DragonJump.app` must be an exported app. For local
experiments without an export, a source-project shim can be created at:

```text
environments/DragonJump.app/Contents/MacOS/DragonJump
```

with a script that runs:

```bash
/Applications/Godot.app/Contents/MacOS/Godot \
  --path /absolute/path/to/dragon-jump-remaster/godot "$@"
```

The `environments/` directory is gitignored, so the shim is local-only.
