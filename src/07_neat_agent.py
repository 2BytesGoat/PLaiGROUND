"""Train a NEAT policy for Dragon Jump level 1-1.

The game's training environment already runs N isolated agents and returns a
reward for every agent at every step. This script evaluates a whole NEAT
population in one Godot process: agent i receives the action from genome i,
and the genome's fitness is the sum of the rewards the game already sends.

Usage:
    python src/07_neat_agent.py --generations 30 --population 32
    python src/07_neat_agent.py --resume runs/neat_1_1/neat-checkpoint-20 \
        --generations 3 --speedup 2 --show
    python src/07_neat_agent.py --play runs/neat_1_1/best_genome.pkl --show

The environment config is read from src/.config. Level, action repeat,
speedup, and show-window values are overridden here so the shared config stays
usable for the other notebooks.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# The other src scripts are run from src/ directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import neat  # noqa: E402

from utils import setup_environment  # noqa: E402

DEFAULT_LEVEL = "1-1"
DEFAULT_CONFIG = Path(__file__).with_name("neat_1_1_config.txt")
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "runs" / "neat_1_1"
DEFAULT_MAX_STEPS = 600  # 10 seconds at 60 physics ticks


class RewardEvaluator:
    """Evaluate every genome as one agent in the same deterministic level."""

    def __init__(
        self,
        level: str,
        max_steps: int,
        output_dir: Path,
        action_repeat: int = 1,
        start_delay: int = 0,
        speedup: float = 1.0,
        show_window: bool = False,
        port: int | None = None,
    ) -> None:
        self.level = level
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.action_repeat = action_repeat
        self.start_delay = start_delay
        self.speedup = speedup
        self.show_window = show_window
        self.port = port
        self.generation = 0
        self.best_fitness = float("-inf")
        self.best_genome = None
        self.best_time = float("inf")
        self.last_improvement_generation = 0
        self.history: list[dict] = []
        self.genome_history: list[dict] = []
        self.node_history: list[dict] = []

    def __call__(self, genomes, config) -> None:
        self.generation += 1
        nets = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]
        population = len(nets)

        print(
            f"\n=== generation {self.generation} | "
            f"population={population} | max_steps={self.max_steps} ===",
            flush=True,
        )

        env = setup_environment(
            nb_agents=population,
            level=self.level,
            action_repeat=self.action_repeat,
            speedup=self.speedup,
            show_window=self.show_window,
            port=self.port,
        )
        try:
            obs = env.reset()
            if env.num_envs != population:
                raise RuntimeError(
                    f"Expected {population} training agents, got {env.num_envs}"
                )
            scores = np.zeros(population, dtype=np.float32)
            done_once = np.zeros(population, dtype=bool)
            finished = np.zeros(population, dtype=bool)
            finish_times = np.full(population, np.inf, dtype=np.float64)
            terminal_rewards = np.zeros(population, dtype=np.float32)
            done_steps = np.zeros(population, dtype=np.int32)
            final_infos = [{} for _ in range(population)]
            last_infos = [{} for _ in range(population)]

            for step in range(self.max_steps):
                actions = []
                for i in range(population):
                    if done_once[i]:
                        actions.append([0])
                        continue

                    # The game starts on the first jump press. A configurable
                    # delay gives us deterministic timing-robustness variants
                    # instead of relying on RNG (the level itself has none).
                    if step < self.start_delay:
                        jump = 0
                    elif step == self.start_delay:
                        jump = 1
                    else:
                        jump = int(nets[i].activate(obs["obs"][i])[0] > 0.5)
                    actions.append([jump])

                obs, rewards, dones, infos = env.step(
                    np.asarray(actions, dtype=np.int64)
                )

                for i in range(population):
                    if done_once[i]:
                        continue
                    scores[i] += float(rewards[i])
                    if dones[i]:
                        done_once[i] = True
                        terminal_rewards[i] = float(rewards[i])
                        finished[i] = terminal_rewards[i] > 0.0
                        done_steps[i] = step + 1
                        final_infos[i] = infos[i]
                        if finished[i]:
                            finish_times[i] = float(infos[i].get("run_time", np.inf))

                last_infos = infos
                if done_once.all():
                    break
        finally:
            env.close()

        # The environment reward is the fitness. No second fitness function is
        # added here: flow progress, efficiency, time pressure, finish, and
        # death are already encoded by PlayerAITrainingController.get_reward().
        for (genome_id, genome), score in zip(genomes, scores):
            genome.fitness = float(score)

        best_index = int(np.argmax(scores))
        generation_best = float(scores[best_index])
        generation_best_time = (
            float(finish_times[best_index])
            if finished[best_index]
            else None
        )
        finish_count = int(finished.sum())
        death_count = int((done_once & ~finished).sum())

        print(
            f"generation {self.generation}: "
            f"best={generation_best:.2f} "
            f"time={generation_best_time if generation_best_time is not None else 'timeout'} "
            f"finishes={finish_count}/{population} "
            f"deaths={death_count} "
            f"mean={float(scores.mean()):.2f}",
            flush=True,
        )

        self.history.append(
            {
                "generation": self.generation,
                "best_fitness": generation_best,
                "mean_fitness": float(scores.mean()),
                "finishes": finish_count,
                "deaths": death_count,
                "best_time": generation_best_time,
                "scores": scores.tolist(),
            }
        )
        self._write_history()

        for i, ((genome_id, genome), score) in enumerate(zip(genomes, scores)):
            info = final_infos[i] or last_infos[i]
            position = info.get("global_position", (None, None))
            if isinstance(position, str):
                try:
                    parts = position.strip("()").split(",")
                    position = (float(parts[0]), float(parts[1]))
                except (ValueError, IndexError):
                    position = (None, None)
            if not isinstance(position, (tuple, list)) or len(position) < 2:
                position = (None, None)
            outcome = "timeout"
            if done_once[i]:
                outcome = "finish" if finished[i] else "death"
            self.genome_history.append(
                {
                    "generation": self.generation,
                    "genome_id": genome_id,
                    "nodes": len(genome.nodes),
                    "enabled_connections": sum(
                        1 for connection in genome.connections.values() if connection.enabled
                    ),
                    "fitness": float(score),
                    "outcome": outcome,
                    "run_time": (
                        float(finish_times[i])
                        if finished[i] and np.isfinite(finish_times[i])
                        else ""
                    ),
                    "steps": int(done_steps[i]) if done_once[i] else self.max_steps,
                    "final_x": position[0],
                    "final_y": position[1],
                }
            )
        self._write_genome_stats()

        if generation_best > self.best_fitness:
            self.best_fitness = generation_best
            self.best_genome = genomes[best_index][1]
            if generation_best_time is not None:
                self.best_time = generation_best_time
            self.last_improvement_generation = self.generation
            self.save_best()

    def is_stable(
        self,
        min_generations: int,
        patience: int,
        min_finish_rate: float,
        min_finish_streak: int,
    ) -> bool:
        """True once the best fitness plateaus with a sustained finish rate."""
        if patience <= 0 or self.generation < min_generations:
            return False
        if not self.history:
            return False
        if self.generation - self.last_improvement_generation < patience:
            return False

        streak = 0
        for row in reversed(self.history):
            population = len(row["scores"])
            if population == 0:
                break
            if row["finishes"] > 0 and row["finishes"] / population >= min_finish_rate:
                streak += 1
            else:
                break
        return streak >= max(1, min_finish_streak)

    def _write_history(self) -> None:
        import csv

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "history.csv"
        rows = self.history
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "generation",
                    "best_fitness",
                    "mean_fitness",
                    "finishes",
                    "deaths",
                    "best_time",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in writer.fieldnames})

    def _write_genome_stats(self) -> None:
        import csv

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "genome_stats.csv"
        fields = [
            "generation",
            "genome_id",
            "nodes",
            "enabled_connections",
            "fitness",
            "outcome",
            "run_time",
            "steps",
            "final_x",
            "final_y",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in self.genome_history:
                writer.writerow({key: row[key] for key in fields})

        buckets = defaultdict(list)
        for row in self.genome_history:
            buckets[(row["generation"], row["nodes"])].append(row)

        bucket_rows = []
        for (generation, nodes), rows in sorted(buckets.items()):
            finish_rows = [row for row in rows if row["outcome"] == "finish"]
            death_rows = [row for row in rows if row["outcome"] == "death"]
            times = [
                float(row["run_time"])
                for row in finish_rows
                if row["run_time"] not in ("", None)
            ]
            bucket_rows.append(
                {
                    "generation": generation,
                    "nodes": nodes,
                    "count": len(rows),
                    "finishes": len(finish_rows),
                    "finish_rate": f"{len(finish_rows) / len(rows):.3f}",
                    "deaths": len(death_rows),
                    "death_rate": f"{len(death_rows) / len(rows):.3f}",
                    "timeouts": sum(1 for row in rows if row["outcome"] == "timeout"),
                    "mean_fitness": f"{sum(float(row['fitness']) for row in rows) / len(rows):.2f}",
                    "best_fitness": f"{max(float(row['fitness']) for row in rows):.2f}",
                    "best_time": f"{min(times):.3f}" if times else "",
                }
            )

        self.node_history = bucket_rows
        bucket_path = self.output_dir / "node_buckets.csv"
        with bucket_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "generation",
                    "nodes",
                    "count",
                    "finishes",
                    "finish_rate",
                    "deaths",
                    "death_rate",
                    "timeouts",
                    "mean_fitness",
                    "best_fitness",
                    "best_time",
                ],
            )
            writer.writeheader()
            writer.writerows(bucket_rows)

    def save_best(self) -> None:
        if self.best_genome is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "best_genome.pkl"
        with path.open("wb") as handle:
            pickle.dump(self.best_genome, handle)
        print(f"saved best genome to {path}", flush=True)


def load_config(config_path: Path, population: int | None = None) -> neat.Config:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    if population is not None:
        config.pop_size = int(population)
    return config


def train(args: argparse.Namespace) -> None:
    if args.action_repeat < 1:
        raise ValueError("--action-repeat must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        population = neat.Checkpointer.restore_checkpoint(str(args.resume))
    else:
        config = load_config(args.config, args.population)
        random.seed(args.seed)
        np.random.seed(args.seed)
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(
        neat.Checkpointer(
            args.checkpoint_interval,
            filename_prefix=str(output_dir / "neat-checkpoint-"),
        )
    )

    evaluator = RewardEvaluator(
        level=args.level,
        max_steps=args.max_steps,
        output_dir=output_dir,
        action_repeat=args.action_repeat,
        start_delay=args.start_delay,
        speedup=args.speedup,
        show_window=args.show,
        port=args.port,
    )

    start = time.time()
    stop_reason = "max generations"
    for _ in range(args.generations):
        population.run(evaluator, 1)
        if evaluator.is_stable(
            args.min_generations,
            args.patience,
            args.min_finish_rate,
            args.min_finish_streak,
        ):
            stop_reason = "stability plateau"
            break
    elapsed = time.time() - start

    if evaluator.best_genome is not None:
        evaluator.save_best()
        print(
            f"\ntraining finished in {elapsed:.1f}s ({stop_reason}) | "
            f"best fitness={evaluator.best_fitness:.2f} | "
            f"best finish time={evaluator.best_time}",
            flush=True,
        )
    else:
        print("\ntraining finished without a saved genome", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate a checkpoint population on one level without evolving it."""
    if args.resume is None:
        raise ValueError("--evaluate requires --resume <checkpoint>")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    population = neat.Checkpointer.restore_checkpoint(str(args.resume))
    evaluator = RewardEvaluator(
        level=args.level,
        max_steps=args.max_steps,
        output_dir=output_dir,
        action_repeat=args.action_repeat,
        start_delay=args.start_delay,
        speedup=args.speedup,
        show_window=args.show,
        port=args.port,
    )
    start = time.time()
    evaluator(list(population.population.items()), population.config)
    elapsed = time.time() - start
    print(
        f"\nevaluation finished in {elapsed:.1f}s | "
        f"best fitness={evaluator.best_fitness:.2f} | "
        f"best finish time={evaluator.best_time}",
        flush=True,
    )


def play(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    genome_path = Path(args.play)
    with genome_path.open("rb") as handle:
        genome = pickle.load(handle)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = setup_environment(
        nb_agents=1,
        level=args.level,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        show_window=args.show,
        port=args.port,
    )
    try:
        obs = env.reset()
        total_reward = 0.0
        for step in range(args.max_steps):
            if step < args.start_delay:
                jump = 0
            elif step == args.start_delay:
                jump = 1
            else:
                jump = int(net.activate(obs["obs"][0])[0] > 0.5)
            obs, rewards, dones, infos = env.step(
                np.asarray([[jump]], dtype=np.int64)
            )
            total_reward += float(rewards[0])
            if dones[0]:
                info = infos[0]
                print(
                    f"finished={rewards[0] > 0} "
                    f"run_time={info.get('run_time')} "
                    f"reward={total_reward:.2f} "
                    f"position={info.get('global_position')}",
                    flush=True,
                )
                return
        print(f"timeout after {args.max_steps} steps reward={total_reward:.2f}")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default=DEFAULT_LEVEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument(
        "--min-generations",
        type=int,
        default=0,
        help="Earliest generation at which stability may stop training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop after N generations without a best-fitness improvement (0 = off).",
    )
    parser.add_argument(
        "--min-finish-rate",
        type=float,
        default=0.0,
        help="Required finish rate when checking stability.",
    )
    parser.add_argument(
        "--min-finish-streak",
        type=int,
        default=1,
        help="Consecutive generations at --min-finish-rate required to stabilize.",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="Physics frames per action. Must be >= 1.",
    )
    parser.add_argument(
        "--start-delay",
        type=int,
        default=0,
        help="Deterministic first-jump delay in physics frames.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port for the Godot training server (default: 11008).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save a NEAT checkpoint every N generations.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume a NEAT population from a neat-checkpoint file.",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="Wall-clock speedup (1-16). Scales physics ticks and time scale.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate every genome in a --resume checkpoint without evolving.",
    )
    parser.add_argument(
        "--play",
        type=Path,
        help="Load a pickled genome and run one episode instead of training.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the game window to display the agents.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.evaluate:
        evaluate(args)
    elif args.play:
        play(args)
    else:
        train(args)
