"""Teacher takeover from random student mid-run states.

This is a closed-loop recovery test that does not need game-side state
injection: the student drives the real game for a seeded random number of
decisions, then the teacher takes over the same live player. If the teacher can
finish from that state, it recovered a state the student actually reached.

Usage:
    .venv/bin/python src/15_recovery_test.py --level 1-11
    .venv/bin/python src/15_recovery_test.py --level 1-11 --min-takeover-step 30 --max-takeover-step 90 --seed 7

The output includes a per-agent CSV with takeover step, outcome, time, and the
teacher's recovery rate. Teacher-only and student-only evals remain the
baselines; this script measures the mixed rollout.
"""

from __future__ import annotations

import argparse
import csv
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from dotenv import load_dotenv

from wrappers.godot_env import GodotEnv


LEVELS = [f"1-{i}" for i in range(1, 17)]
TEACHER_ROOTS = (
    Path("runs/teacher_champions"),
    Path("runs/teacher_repair/final"),
    Path("runs/teacher_repair_post"),
    Path("runs/ppo_lineage/snapshots"),
    Path("runs/ppo_lineage_cont/snapshots"),
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "runs" / "recovery"


def parse_level(value: str) -> str:
    if value not in LEVELS:
        raise ValueError(f"Unknown level {value}; expected one of {LEVELS}")
    return value


def resolve_env_path() -> str:
    import os

    config_path = Path(__file__).resolve().with_name(".config")
    load_dotenv(config_path, override=True)
    env_dir = os.getenv("ENV_PATH", "../environments")
    env_name = os.getenv("ENV_NAME", "DragonJump")
    config_dir = config_path.parent
    repo_dir = config_dir.parent
    if os.path.isabs(env_dir):
        resolved = Path(env_dir)
    else:
        resolved = None
        for base in (Path.cwd(), config_dir, repo_dir):
            candidate = (base / env_dir).resolve()
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            resolved = (Path.cwd() / env_dir).resolve()
    return str(resolved / env_name)


def isolated_home(output_dir: Path, name: str) -> dict[str, str]:
    import os

    root = (output_dir / "userdata" / name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["HOME"] = str(root)
    child_env["XDG_DATA_HOME"] = str(root / "xdg_data")
    child_env["XDG_CONFIG_HOME"] = str(root / "xdg_config")
    return child_env


def model_path(root: Path, value: str | None, default: str) -> Path:
    path = Path(value) if value else root / default
    return path if path.is_absolute() else root / path


def default_teacher_path(level: str) -> Path | None:
    for root in TEACHER_ROOTS:
        candidate = root / level / "ppo_pilot.zip"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="1-11")
    parser.add_argument("--student", default="runs/ppo_distill_post/student_obs58_256.pt")
    parser.add_argument(
        "--teacher",
        default=None,
        help="Teacher checkpoint; defaults to teacher_repair_post/<level>/ppo_pilot.zip.",
    )
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--min-takeover-step", type=int, default=20)
    parser.add_argument("--max-takeover-step", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-port", type=int, default=12000)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    level = parse_level(args.level)
    args.output_dir = (args.output_dir / level).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_path = resolve_env_path()

    student_script = import_module("12_ppo_distill")
    student_path = model_path(Path.cwd(), args.student, "")
    student = student_script.load_student(student_path)
    student.eval()
    obs_encoding = student.obs_encoding
    action_repeat = int(getattr(student, "action_repeat", args.action_repeat))

    teacher_path = (
        Path(args.teacher) if args.teacher else default_teacher_path(level)
    )
    if teacher_path is None or not teacher_path.exists():
        raise FileNotFoundError(teacher_path)
    teacher = PPO.load(str(teacher_path), device="cpu")
    teacher_size = int(teacher.observation_space.shape[0])
    student_size = int(student.trunk[0].in_features)
    if teacher_size != student_size:
        raise ValueError(
            f"teacher observes {teacher_size} values but student observes "
            f"{student_size}; recovery requires matching observation encodings"
        )

    n = args.agents
    if not (0 <= args.min_takeover_step <= args.max_takeover_step < args.max_decisions):
        raise ValueError("takeover range must be inside [0, max_decisions)")
    rng = np.random.default_rng(args.seed)
    takeover_steps = rng.integers(
        args.min_takeover_step,
        args.max_takeover_step + 1,
        size=n,
        endpoint=True,
    )

    env = GodotEnv(
        env_path=env_path,
        port=args.eval_port,
        show_window=args.show,
        seed=args.seed,
        action_repeat=action_repeat,
        speedup=args.speedup,
        nb_agents=n,
        level=level,
        convert_action_space=True,
        log_path=str(args.output_dir / "godot.log"),
        env=isolated_home(args.output_dir, "recovery"),
        obs_encoding=obs_encoding,
    )
    done_once = np.zeros(n, dtype=bool)
    finished = np.zeros(n, dtype=bool)
    finished_before_takeover = np.zeros(n, dtype=bool)
    finish_times = np.full(n, np.inf, dtype=np.float64)
    deaths = np.zeros(n, dtype=np.int32)
    efficiency = np.full(n, np.nan, dtype=np.float64)
    flow = np.full(n, np.nan, dtype=np.float64)
    last_infos: list[dict] = [{} for _ in range(n)]

    try:
        obs, _ = env.reset()
        for step in range(args.max_decisions):
            obs_np = np.asarray(
                [obs[i]["obs"] for i in range(n)],
                dtype=np.float32,
            )
            with torch.no_grad():
                student_actions = (
                    student(torch.from_numpy(obs_np)).argmax(dim=1).numpy()
                )
            teacher_actions = np.zeros(n, dtype=np.int64)
            for i in range(n):
                if done_once[i]:
                    continue
                teacher_action, _ = teacher.predict(
                    obs_np[i], deterministic=True
                )
                teacher_actions[i] = int(teacher_action)

            use_teacher = step >= takeover_steps
            actions = np.where(use_teacher, teacher_actions, student_actions)
            actions[done_once] = 0

            obs, rewards, dones, _timeouts, infos = env.step(
                actions.reshape(-1, 1)
            )
            for i in range(n):
                if done_once[i]:
                    continue
                last_infos[i] = dict(infos[i]) if infos[i] else {}
                if dones[i]:
                    done_once[i] = True
                    finished[i] = float(rewards[i]) > 0.0
                    finished_before_takeover[i] = bool(finished[i] and step < takeover_steps[i])
                    if "deaths" in last_infos[i]:
                        try:
                            deaths[i] = int(last_infos[i]["deaths"])
                        except (TypeError, ValueError):
                            pass
                    if finished[i]:
                        try:
                            finish_times[i] = float(last_infos[i]["run_time"])
                        except (KeyError, TypeError, ValueError):
                            finish_times[i] = float("nan")
                        if not np.isfinite(finish_times[i]) or finish_times[i] <= 0.0:
                            finish_times[i] = (
                                (step + 1) * action_repeat / 60.0
                            )
            if done_once.all():
                break
    finally:
        env.close()

    for i in range(n):
        info = last_infos[i]
        for key, target in (("efficiency", efficiency), ("flow", flow)):
            if key in info:
                try:
                    target[i] = float(info[key])
                except (TypeError, ValueError):
                    pass
        if "deaths" in info:
            try:
                deaths[i] = int(info["deaths"])
            except (TypeError, ValueError):
                pass

    finish_count = int(finished.sum())
    recoveries = int((finished & ~finished_before_takeover).sum())
    clean_times = finish_times[np.isfinite(finish_times)]
    best = float(np.min(clean_times)) if clean_times.size else None
    median = float(np.median(clean_times)) if clean_times.size else None
    efficiency_mean = (
        float(np.nanmean(efficiency)) if np.isfinite(efficiency).any() else None
    )
    flow_mean = float(np.nanmean(flow)) if np.isfinite(flow).any() else None

    rows = []
    for i in range(n):
        rows.append(
            {
                "agent": i,
                "takeover_step": int(takeover_steps[i]),
                "finished": bool(finished[i]),
                "finished_before_takeover": bool(finished_before_takeover[i]),
                "run_time": (
                    "" if not finished[i] else f"{finish_times[i]:.6f}"
                ),
                "deaths": int(deaths[i]),
                "efficiency": (
                    f"{efficiency[i]:.6f}" if np.isfinite(efficiency[i]) else ""
                ),
                "flow": f"{flow[i]:.6f}" if np.isfinite(flow[i]) else "",
            }
        )
    with (args.output_dir / "agents.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "level": level,
        "agents": n,
        "takeover_min": args.min_takeover_step,
        "takeover_max": args.max_takeover_step,
        "seed": args.seed,
        "finishes": finish_count,
        "recoveries_after_takeover": recoveries,
        "finished_before_takeover": int(finished_before_takeover.sum()),
        "deaths": int(deaths.sum()),
        "best_time": f"{best:.6f}" if best is not None else "",
        "median_time": f"{median:.6f}" if median is not None else "",
        "efficiency_mean": (
            f"{efficiency_mean:.6f}" if efficiency_mean is not None else ""
        ),
        "flow_mean": f"{flow_mean:.6f}" if flow_mean is not None else "",
    }
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(
        f"[recovery {level}] finishes={finish_count}/{n} "
        f"after_takeover={recoveries} "
        f"best={summary['best_time'] or 'none'} "
        f"median={summary['median_time'] or 'none'} "
        f"eff={summary['efficiency_mean'] or 'none'} "
        f"flow={summary['flow_mean'] or 'none'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
