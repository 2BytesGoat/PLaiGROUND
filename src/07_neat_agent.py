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
import hashlib
import json
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
        parallel: int = 1,
        record_paths: bool = False,
        archive_dir: Path | None = None,
        initial_generation: int = 0,
    ) -> None:
        self.level = level
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.action_repeat = action_repeat
        self.start_delay = start_delay
        self.speedup = speedup
        self.show_window = show_window
        self.port = port
        self.parallel = max(1, int(parallel))
        self.record_paths = bool(record_paths)
        self.archive_dir = archive_dir or (output_dir / "archive")
        self.env = None
        self.population = 0
        self.startup_s = 0.0
        self.generation = int(initial_generation)
        self.best_fitness = float("-inf")
        self.best_genome = None
        self.best_time = float("inf")
        self.best_time_genome = None
        self.best_clean_time = float("inf")
        self.best_clean_genome = None
        self.route_archive: dict[str, dict] = {}
        self.last_improvement_generation = 0
        self.history: list[dict] = []
        self.genome_history: list[dict] = []
        self.node_history: list[dict] = []
        self._load_archive_state()

    def _load_archive_state(self) -> None:
        """Reload route, best, and history state when resuming into a run."""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        routes_path = self.archive_dir / "routes.json"
        if routes_path.exists():
            try:
                payload = json.loads(routes_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self.route_archive = payload
            except (OSError, ValueError):
                self.route_archive = {}

        best_time_path = self.output_dir / "best_time.json"
        if best_time_path.exists():
            try:
                payload = json.loads(best_time_path.read_text(encoding="utf-8"))
                self.best_time = float(payload.get("time", self.best_time))
            except (OSError, TypeError, ValueError):
                pass
        best_time_genome_path = self.output_dir / "best_time_genome.pkl"
        if best_time_genome_path.exists():
            try:
                with best_time_genome_path.open("rb") as handle:
                    self.best_time_genome = pickle.load(handle)
            except (OSError, pickle.UnpicklingError):
                self.best_time_genome = None

        best_clean_path = self.output_dir / "best_clean.json"
        if best_clean_path.exists():
            try:
                payload = json.loads(best_clean_path.read_text(encoding="utf-8"))
                self.best_clean_time = float(
                    payload.get("time", self.best_clean_time)
                )
            except (OSError, TypeError, ValueError):
                pass
        best_clean_genome_path = self.output_dir / "best_clean_genome.pkl"
        if best_clean_genome_path.exists():
            try:
                with best_clean_genome_path.open("rb") as handle:
                    self.best_clean_genome = pickle.load(handle)
            except (OSError, pickle.UnpicklingError):
                self.best_clean_genome = None

        finishers_path = self.archive_dir / "finishers.jsonl"
        if finishers_path.exists() and (
            not np.isfinite(self.best_time) or not np.isfinite(self.best_clean_time)
        ):
            try:
                for line in finishers_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    run_time = float(record.get("run_time", float("inf")))
                    if run_time < self.best_time:
                        self.best_time = run_time
                    if record.get("clean") and run_time < self.best_clean_time:
                        self.best_clean_time = run_time
            except (OSError, TypeError, ValueError):
                pass

        best_genome_path = self.output_dir / "best_genome.pkl"
        if best_genome_path.exists():
            try:
                with best_genome_path.open("rb") as handle:
                    self.best_genome = pickle.load(handle)
            except (OSError, pickle.UnpicklingError):
                self.best_genome = None

        history_path = self.output_dir / "history.csv"
        if history_path.exists():
            try:
                import csv

                with history_path.open(newline="", encoding="utf-8") as handle:
                    self.history = list(csv.DictReader(handle))
                fitness_values = [
                    float(row["best_fitness"])
                    for row in self.history
                    if row.get("best_fitness") not in (None, "")
                ]
                if fitness_values:
                    self.best_fitness = max(fitness_values)
            except (OSError, TypeError, ValueError):
                pass

        genome_stats_path = self.output_dir / "genome_stats.csv"
        if genome_stats_path.exists():
            try:
                import csv

                with genome_stats_path.open(
                    newline="", encoding="utf-8"
                ) as handle:
                    self.genome_history = list(csv.DictReader(handle))
            except OSError:
                pass

    def open(self, population: int) -> float:
        """Create the game processes once and reuse them for every generation.

        The first call pays Godot launch/handshake/level-load cost. Later calls
        keep the same environment and only reset the agents, which is the warm
        generation measured by the throughput probe.
        """
        if self.env is not None and self.population == population:
            return 0.0
        self.close()
        if population % self.parallel != 0:
            raise ValueError(
                f"population {population} must be divisible by --parallel "
                f"({self.parallel})"
            )
        started = time.perf_counter()
        self.env = setup_environment(
            nb_agents=population // self.parallel,
            n_parallel=self.parallel,
            level=self.level,
            action_repeat=self.action_repeat,
            speedup=self.speedup,
            show_window=self.show_window,
            port=self.port,
        )
        self.population = population
        self.startup_s = time.perf_counter() - started
        return self.startup_s

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
        self.population = 0

    def __call__(self, genomes, config) -> None:
        self.generation += 1
        nets = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]
        population = len(nets)
        if population % self.parallel != 0:
            raise ValueError(
                f"population {population} must be divisible by --parallel "
                f"({self.parallel})"
            )
        startup_this = self.open(population)
        env = self.env
        if env is None:
            raise RuntimeError("failed to open the training environment")
        generation_start = time.perf_counter()

        print(
            f"\n=== generation {self.generation} | "
            f"population={population} | parallel={self.parallel} | "
            f"max_steps={self.max_steps} | startup={startup_this:.2f}s ===",
            flush=True,
        )

        try:
            reset_start = time.perf_counter()
            obs = env.reset()
            reset_s = time.perf_counter() - reset_start
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
            cell_paths: list[list[list[int]]] = [[] for _ in range(population)]
            action_paths: list[list[int]] = [[] for _ in range(population)]
            steps_start = time.perf_counter()

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
                    action_paths[i].append(jump)
                    actions.append([jump])

                obs, rewards, dones, infos = env.step(
                    np.asarray(actions, dtype=np.int64)
                )

                for i in range(population):
                    if done_once[i]:
                        continue
                    info = infos[i] if i < len(infos) else {}
                    cell = info.get("cell")
                    if (
                        isinstance(cell, (list, tuple))
                        and len(cell) >= 2
                    ):
                        cell_paths[i].append([int(cell[0]), int(cell[1])])
                    scores[i] += float(rewards[i])
                    if dones[i]:
                        done_once[i] = True
                        terminal_rewards[i] = float(rewards[i])
                        finished[i] = bool(
                            info.get("finished", terminal_rewards[i] > 0.0)
                        )
                        done_steps[i] = step + 1
                        final_infos[i] = info
                        if finished[i]:
                            finish_times[i] = float(info.get("run_time", np.inf))

                last_infos = infos
                if done_once.all():
                    break
            steps_s = time.perf_counter() - steps_start
            generation_s = time.perf_counter() - generation_start
        finally:
            # Intentionally keep the environment alive for the next generation;
            # train()/evaluate() own the final close.
            pass

        # The environment reward is the fitness. No second fitness function is
        # added here: flow progress, efficiency, time pressure, finish, and
        # death are already encoded by PlayerAITrainingController.get_reward().
        for (genome_id, genome), score in zip(genomes, scores):
            genome.fitness = float(score)

        best_index = int(np.argmax(scores))
        generation_best = float(scores[best_index])
        finished_indices = np.flatnonzero(finished)
        generation_best_time = (
            float(finish_times[finished_indices].min())
            if len(finished_indices)
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

        self.history = [
            row
            for row in self.history
            if str(row.get("generation", "")) != str(self.generation)
        ]
        self.history.append(
            {
                "generation": self.generation,
                "best_fitness": generation_best,
                "mean_fitness": float(scores.mean()),
                "finishes": finish_count,
                "deaths": death_count,
                "best_time": generation_best_time,
                "startup_s": startup_this,
                "reset_s": reset_s,
                "steps_s": steps_s,
                "generation_s": generation_s,
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
        self._archive_generation(
            genomes,
            finished,
            finish_times,
            cell_paths,
            action_paths,
            final_infos,
            scores,
        )

        if len(finished_indices):
            best_time_index = int(
                finished_indices[
                    int(np.argmin(finish_times[finished_indices]))
                ]
            )
            if float(finish_times[best_time_index]) < self.best_time:
                self.best_time = float(finish_times[best_time_index])
                self.best_time_genome = genomes[best_time_index][1]
                self.save_best_time()

        if generation_best > self.best_fitness:
            self.best_fitness = generation_best
            self.best_genome = genomes[best_index][1]
            self.last_improvement_generation = self.generation
            self.save_best()

    def _archive_generation(
        self,
        genomes,
        finished,
        finish_times,
        cell_paths,
        action_paths,
        final_infos,
        scores,
    ) -> None:
        """Append successful trajectories and keep the fastest per route.

        The raw JSONL is intentionally unfiltered: it is the state-source for
        prefix-replay PPO initialization. Route deduplication and the
        non-jitter flag are derived metadata, not deletion criteria.
        """
        if not self.record_paths:
            return
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        genome_dir = self.archive_dir / "genomes"
        genome_dir.mkdir(parents=True, exist_ok=True)
        finishers_path = self.archive_dir / "finishers.jsonl"
        routes_path = self.archive_dir / "routes.json"

        for i in np.flatnonzero(finished):
            cells = []
            for cell in cell_paths[int(i)]:
                if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                    cells.append([int(cell[0]), int(cell[1])])
            if len(cells) < 2:
                continue
            actions = [int(item) for item in action_paths[int(i)]]
            genome_id, genome = genomes[int(i)]
            info = final_infos[int(i)] or {}
            run_time = float(finish_times[int(i)])

            compressed = []
            for cell in cells:
                if not compressed or compressed[-1] != cell:
                    compressed.append(cell)
            unique_cells = len({tuple(cell) for cell in compressed})
            revisit_ratio = (
                1.0 - unique_cells / len(compressed) if compressed else 1.0
            )
            backtracks = sum(
                1
                for index in range(2, len(compressed))
                if compressed[index] == compressed[index - 2]
            )
            action_flips = (
                sum(
                    1
                    for index in range(1, len(actions))
                    if actions[index] != actions[index - 1]
                )
                / max(1, len(actions))
            )
            clean = (
                len(compressed) >= 5
                and revisit_ratio <= 0.10
                and backtracks <= 2
                and action_flips <= 0.75
            )
            signature = hashlib.sha1(
                json.dumps(compressed, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            record = {
                "generation": self.generation,
                "genome_id": genome_id,
                "fitness": float(scores[int(i)]),
                "run_time": run_time,
                "path_cells": compressed,
                "actions": actions,
                "route_signature": signature,
                "path_length": len(compressed),
                "unique_cells": unique_cells,
                "revisit_ratio": revisit_ratio,
                "backtracks": backtracks,
                "action_flip_rate": action_flips,
                "clean": clean,
                "final_info": {
                    key: info.get(key)
                    for key in (
                        "state",
                        "global_position",
                        "velocity",
                        "run_time",
                        "best_run_time",
                        "run_active",
                        "started_walking",
                        "facing_direction",
                        "on_floor",
                        "on_wall",
                        "powerup_count",
                    )
                },
            }
            with finishers_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            previous = self.route_archive.get(signature)
            if previous is None or run_time < float(previous["run_time"]):
                genome_path = genome_dir / f"{signature}.pkl"
                with genome_path.open("wb") as handle:
                    pickle.dump(genome, handle)
                self.route_archive[signature] = {
                    "generation": self.generation,
                    "genome_id": genome_id,
                    "run_time": run_time,
                    "path_length": len(compressed),
                    "unique_cells": unique_cells,
                    "revisit_ratio": revisit_ratio,
                    "backtracks": backtracks,
                    "action_flip_rate": action_flips,
                    "clean": clean,
                    "genome_file": str(genome_path.relative_to(self.archive_dir)),
                }
            if clean and run_time < self.best_clean_time:
                self.best_clean_time = run_time
                self.best_clean_genome = genome
                self.save_best_clean()

        if self.route_archive:
            routes_path.write_text(
                json.dumps(self.route_archive, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def save_best_time(self) -> None:
        if self.best_time_genome is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "best_time_genome.pkl"
        with path.open("wb") as handle:
            pickle.dump(self.best_time_genome, handle)
        (self.output_dir / "best_time.json").write_text(
            json.dumps(
                {
                    "time": self.best_time,
                    "generation": self.generation,
                    "level": self.level,
                    "observation": "legacy_58",
                    "action_repeat": self.action_repeat,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved best-time genome to {path}", flush=True)

    def save_best_clean(self) -> None:
        if self.best_clean_genome is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "best_clean_genome.pkl"
        with path.open("wb") as handle:
            pickle.dump(self.best_clean_genome, handle)
        (self.output_dir / "best_clean.json").write_text(
            json.dumps(
                {
                    "time": self.best_clean_time,
                    "generation": self.generation,
                    "level": self.level,
                    "observation": "legacy_58",
                    "action_repeat": self.action_repeat,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved best-clean genome to {path}", flush=True)

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
            population = len(row.get("scores", []))
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
                    "startup_s",
                    "reset_s",
                    "steps_s",
                    "generation_s",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

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
            try:
                generation = int(float(row["generation"]))
                nodes = int(float(row["nodes"]))
            except (KeyError, TypeError, ValueError):
                continue
            buckets[(generation, nodes)].append(row)

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
    initial_generation = int(getattr(population, "generation", 0))

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
        parallel=args.parallel,
        record_paths=args.record_paths,
        archive_dir=args.archive_dir,
        initial_generation=initial_generation,
    )

    start = time.time()
    stop_reason = "max generations"
    try:
        for _ in range(args.generations):
            population.run(evaluator, 1)
            if (
                args.target_time is not None
                and evaluator.best_time <= args.target_time
            ):
                stop_reason = "target time"
                break
            if evaluator.is_stable(
                args.min_generations,
                args.patience,
                args.min_finish_rate,
                args.min_finish_streak,
            ):
                stop_reason = "stability plateau"
                break
    finally:
        evaluator.close()
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
        parallel=args.parallel,
        record_paths=args.record_paths,
        archive_dir=args.archive_dir,
        initial_generation=int(getattr(population, "generation", 0)),
    )
    start = time.time()
    try:
        evaluator(list(population.population.items()), population.config)
    finally:
        evaluator.close()
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
        "--parallel",
        type=int,
        default=1,
        help="Game processes per generation; population must be divisible by it.",
    )
    parser.add_argument(
        "--record-paths",
        action="store_true",
        help="Append every finisher's cell/action path and write route archives.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Path archive directory (default: <output-dir>/archive).",
    )
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
    parser.add_argument(
        "--target-time",
        type=float,
        default=None,
        help="Stop after the generation whose best finish time is <= this value.",
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
        help="Decision step at which the forced start jump is pressed (default: 0, "
        "the first decision). This jump is always injected before the genome acts.",
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
