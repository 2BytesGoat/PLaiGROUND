"""Distill per-level PPO experts into one level-agnostic student policy.

This is supervised learning, not PPO. For each passing level snapshot we run the
teacher on that level, record observations and teacher action probabilities,
then train one student MLP on:

    observation (legacy 58 or multi-hot grid) -> jump probabilities (2)

The student has no campaign-level input. The same policy must infer what to do
from the semantic grid, exit direction, time-to-exit, and player state alone. It
is evaluated on all levels with 100 deterministic agents. Missing teacher
levels are skipped and reported. The agent chooses every action, including the
first jump that starts the run; the collector and evaluator do not force an
opening action.

Set ``--obs-encoding multi_hot`` on collect/train/dagger/eval to use the new
754-value semantic grid. Teachers must be trained with the same encoding.

Commands:
    python src/12_ppo_distill.py collect --episodes 2
    python src/12_ppo_distill.py train
    python src/12_ppo_distill.py eval --show

The game process user trees stay under runs/ppo_distill/userdata; no real save
or settings are touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from dotenv import load_dotenv

from wrappers.godot_env import GodotEnv
import obs_spec


LEVELS = [f"1-{i}" for i in range(1, 17)]
# Repaired teachers take precedence over the original stage snapshots.
TEACHER_ROOTS = (
    Path("runs/teacher_champions"),
    Path("runs/teacher_repair/final"),
    Path("runs/teacher_repair_post"),
    Path("runs/ppo_lineage/snapshots"),
    Path("runs/ppo_lineage_cont/snapshots"),
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "runs" / "ppo_distill"


def parse_levels(value: str) -> list[str]:
    if not value or value.lower() == "all":
        return list(LEVELS)
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [level for level in levels if level not in LEVELS]
    if unknown:
        raise ValueError(f"Unknown level(s): {', '.join(unknown)}")
    return levels


def resolve_env_path() -> str:
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
    root = (output_dir / "userdata" / name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["HOME"] = str(root)
    child_env["XDG_DATA_HOME"] = str(root / "xdg_data")
    child_env["XDG_CONFIG_HOME"] = str(root / "xdg_config")
    return child_env


def teacher_path(level: str, root: Path | None = None) -> Path | None:
    roots = (root,) if root is not None else TEACHER_ROOTS
    for candidate_root in roots:
        candidate = candidate_root / level / "ppo_pilot.zip"
        if candidate.exists():
            return candidate
    return None


def available_levels() -> list[str]:
    return [level for level in LEVELS if teacher_path(level) is not None]


def level_index(level: str) -> int:
    return int(level.split("-")[1]) - 1


def teacher_probs(teacher: PPO, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(obs, dtype=torch.float32)
        dist = teacher.policy.get_distribution(tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()
    return probs.astype(np.float32)


class StudentPolicy(nn.Module):
    """One level-agnostic student policy over the game observation vector.

    Dropout is a training-only regularizer / MC-dropout exploration knob. The
    exported JSON has no dropout; the runtime always evaluates deterministically.
    """

    def __init__(
        self,
        input_dim: int = obs_spec.LEGACY_OBSERVATION_SIZE,
        hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Keep the original four-module trunk layout so pre-dropout checkpoints
        # keep their `trunk.0` / `trunk.2` keys. Dropout is applied functionally
        # in forward(), not as extra modules.
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.heads = nn.Linear(hidden, 2)
        self.dropout_p = float(dropout)

    def forward(self, x):
        x = self.trunk[1](self.trunk[0](x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.trunk[3](self.trunk[2](x))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.heads(x)


def collect_level(
    level: str,
    model_path: Path,
    args: argparse.Namespace,
    env_path: str,
) -> dict:
    out_dir = args.output_dir / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{level}.npz"
    log_path = args.output_dir / "logs" / f"collect-{level}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    teacher = PPO.load(str(model_path), device="cpu")
    expected_size = obs_spec.expected_obs_size(args.obs_encoding)
    teacher_size = int(teacher.observation_space.shape[0])
    teacher_uses_legacy = teacher_size == obs_spec.LEGACY_OBSERVATION_SIZE
    if teacher_size != expected_size and not (
        args.obs_encoding == obs_spec.ENCODING_MULTI_HOT
        and teacher_uses_legacy
    ):
        raise ValueError(
            f"{model_path} observes {teacher_size} values, but "
            f"--obs-encoding={args.obs_encoding} expects {expected_size}; "
            "train the teacher with either encoding and let the game emit "
            "the legacy vector for a legacy teacher"
        )
    n = args.agents
    idx = level_index(level)
    rng = np.random.default_rng(args.seed + idx)

    obs_all: list[np.ndarray] = []
    prob_all: list[np.ndarray] = []
    action_all: list[np.ndarray] = []
    level_all: list[np.ndarray] = []
    step_all: list[np.ndarray] = []
    episode_all: list[np.ndarray] = []
    agent_all: list[np.ndarray] = []

    env = GodotEnv(
        env_path=env_path,
        port=args.base_port,
        show_window=args.show,
        seed=args.seed + idx,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        nb_agents=n,
        level=level,
        convert_action_space=True,
        log_path=str(log_path),
        env=isolated_home(args.output_dir, f"collect_{level}"),
        obs_encoding=args.obs_encoding,
        emit_legacy_obs=teacher_uses_legacy,
    )
    try:
        obs, _ = env.reset()
        for episode in range(args.episodes):
            active = np.ones(n, dtype=bool)
            for step in range(args.max_decisions):
                student_obs_np = np.asarray(
                    [obs[i]["obs"] for i in range(n)],
                    dtype=np.float32,
                )
                teacher_obs_np = student_obs_np
                if teacher_uses_legacy:
                    teacher_obs_np = np.asarray(
                        [obs[i]["legacy_obs"] for i in range(n)],
                        dtype=np.float32,
                    )
                probs = teacher_probs(teacher, teacher_obs_np)
                actions = (rng.random(n) < probs[:, 1]).astype(np.int64)

                active_idx = np.flatnonzero(active)
                if active_idx.size:
                    obs_all.append(student_obs_np[active_idx].copy())
                    prob_all.append(probs[active_idx].copy())
                    action_all.append(actions[active_idx].copy())
                    level_all.append(np.full(active_idx.size, idx, dtype=np.int64))
                    step_all.append(np.full(active_idx.size, step, dtype=np.int32))
                    episode_all.append(
                        np.full(active_idx.size, episode, dtype=np.int32)
                    )
                    agent_all.append(active_idx.astype(np.int32))

                obs, rewards, dones, _timeouts, _infos = env.step(
                    actions.reshape(-1, 1)
                )
                active &= ~np.asarray(dones, dtype=bool)
                if not active.any():
                    break
            if episode + 1 < args.episodes:
                obs, _ = env.reset()
    finally:
        env.close()

    if not obs_all:
        raise RuntimeError(f"No data collected for {level}")

    arrays = {
        "obs": np.concatenate(obs_all).astype(np.float32),
        "probs": np.concatenate(prob_all).astype(np.float32),
        "actions": np.concatenate(action_all).astype(np.int64),
        "level": np.concatenate(level_all).astype(np.int64),
        "step": np.concatenate(step_all).astype(np.int32),
        "episode": np.concatenate(episode_all).astype(np.int32),
        "agent": np.concatenate(agent_all).astype(np.int32),
    }
    np.savez_compressed(data_path, **arrays)
    result = {
        "level": level,
        "samples": len(arrays["actions"]),
        "teacher": str(model_path),
        "data": str(data_path),
    }
    print(
        f"[collect {level}] samples={result['samples']} teacher={model_path}",
        flush=True,
    )
    return result


def resolve_torch_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_student(args: argparse.Namespace) -> None:
    paths = sorted((args.output_dir / "dataset").glob("*.npz"))
    if not paths:
        raise FileNotFoundError("No dataset/*.npz files; run collect first")

    obs_parts = []
    prob_parts = []
    for path in paths:
        data = np.load(path)
        obs_parts.append(data["obs"])
        prob_parts.append(data["probs"])
    obs = np.concatenate(obs_parts).astype(np.float32)
    probs = np.concatenate(prob_parts)
    obs = obs_spec.validate_obs_batch(obs, args.obs_encoding)
    input_dim = int(obs.shape[1])

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(obs))
    split = int(len(obs) * (1.0 - args.val_fraction))
    train_idx = order[:split]
    val_idx = order[split:]

    train_ds = TensorDataset(
        torch.from_numpy(obs[train_idx]),
        torch.from_numpy(probs[train_idx]),
    )
    val_x = torch.from_numpy(obs[val_idx])
    val_y = torch.from_numpy(probs[val_idx])
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    init_seed = args.init_seed if args.init_seed is not None else args.seed
    torch.manual_seed(init_seed)
    device = resolve_torch_device(args.device)
    print(f"[train] device={device} samples={len(obs)}", flush=True)
    student = StudentPolicy(
        input_dim=input_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_val = float("inf")
    best_state = None
    history = []
    model_path = args.output_dir / args.model_name
    log_path = args.output_dir / f"train_{model_path.stem}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        student.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = student(xb)
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(yb * log_probs).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(xb)
            count += len(xb)

        student.eval()
        with torch.no_grad():
            val_logits = student(val_x)
            val_log_probs = F.log_softmax(val_logits, dim=1)
            val_loss = float(-(val_y * val_log_probs).sum(dim=1).mean())
            val_acc = float(
                (val_logits.argmax(dim=1) == val_y.argmax(dim=1))
                .float()
                .mean()
            )
        train_loss = total / max(1, count)
        history.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_accuracy": f"{val_acc:.6f}",
            }
        )
        print(
            f"[train] epoch={epoch} train={train_loss:.4f} "
            f"val={val_loss:.4f} acc={val_acc:.3f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in student.state_dict().items()
            }

    if best_state is not None:
        student.load_state_dict(best_state)
    available = sorted(
        {
            p.stem.removeprefix("dagger_")
            for p in paths
        }
    )
    missing = [level for level in LEVELS if level not in available]
    save_state = {
        key: value.detach().cpu()
        for key, value in student.state_dict().items()
    }
    torch.save(
        {
            "state_dict": save_state,
            "input_dim": input_dim,
            "obs_encoding": args.obs_encoding,
            "observation_size": input_dim,
            "action_repeat": args.action_repeat,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "init_seed": init_seed,
            "trained_levels": available,
            "missing_levels": missing,
        },
        model_path,
    )
    with log_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"],
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"[train] saved {model_path}", flush=True)
    print(
        f"[train] samples={len(obs)} levels={available} missing={missing}",
        flush=True,
    )


def dagger_collect(
    level: str,
    student: StudentPolicy,
    teacher: PPO,
    args: argparse.Namespace,
    env_path: str,
) -> int:
    """Run the student and label its visited states with the level teacher."""
    out_dir = args.output_dir / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"dagger_{level}.npz"
    log_path = args.output_dir / "logs" / f"dagger-{level}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    expected_size = obs_spec.expected_obs_size(args.obs_encoding)
    if int(student.trunk[0].in_features) != expected_size:
        raise ValueError(
            f"student observes {student.trunk[0].in_features} values, but "
            f"--obs-encoding={args.obs_encoding} expects {expected_size}"
        )
    student_action_repeat = int(getattr(student, "action_repeat", 4))
    if student_action_repeat != args.action_repeat:
        raise ValueError(
            f"student was trained with action_repeat={student_action_repeat}, "
            f"not {args.action_repeat}"
        )
    teacher_size = int(teacher.observation_space.shape[0])
    teacher_uses_legacy = teacher_size == obs_spec.LEGACY_OBSERVATION_SIZE
    if teacher_size != expected_size and not (
        args.obs_encoding == obs_spec.ENCODING_MULTI_HOT
        and teacher_uses_legacy
    ):
        raise ValueError(
            f"teacher observes {teacher_size} values, but "
            f"--obs-encoding={args.obs_encoding} expects {expected_size}; "
            "train the teacher with either encoding and let the game emit "
            "the legacy vector for a legacy teacher"
        )
    n = args.agents
    idx = level_index(level)
    rng = np.random.default_rng(args.seed + 100 + idx)

    obs_all: list[np.ndarray] = []
    prob_all: list[np.ndarray] = []
    action_all: list[np.ndarray] = []
    level_all: list[np.ndarray] = []
    step_all: list[np.ndarray] = []
    episode_all: list[np.ndarray] = []
    agent_all: list[np.ndarray] = []

    env = GodotEnv(
        env_path=env_path,
        port=args.base_port,
        show_window=args.show,
        seed=args.seed + 100 + idx,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        nb_agents=n,
        level=level,
        convert_action_space=True,
        log_path=str(log_path),
        env=isolated_home(args.output_dir, f"dagger_{level}"),
        obs_encoding=args.obs_encoding,
        emit_legacy_obs=teacher_uses_legacy,
    )
    try:
        obs, _ = env.reset()
        for episode in range(args.episodes):
            active = np.ones(n, dtype=bool)
            for step in range(args.max_decisions):
                student_obs_np = np.asarray(
                    [obs[i]["obs"] for i in range(n)],
                    dtype=np.float32,
                )
                teacher_obs_np = student_obs_np
                if teacher_uses_legacy:
                    teacher_obs_np = np.asarray(
                        [obs[i]["legacy_obs"] for i in range(n)],
                        dtype=np.float32,
                    )
                # Student rollouts supply the states; the teacher supplies
                # the labels at exactly those states.
                with torch.no_grad():
                    logits = student(torch.from_numpy(student_obs_np))
                    temperature = max(float(args.rollout_temperature), 1e-3)
                    student_probs = torch.softmax(
                        logits / temperature, dim=1
                    ).numpy()
                teacher_probs_np = teacher_probs(teacher, teacher_obs_np)
                actions = (
                    rng.random(n) < student_probs[:, 1]
                ).astype(np.int64)

                active_idx = np.flatnonzero(active)
                if active_idx.size:
                    obs_all.append(student_obs_np[active_idx].copy())
                    prob_all.append(teacher_probs_np[active_idx].copy())
                    action_all.append(actions[active_idx].copy())
                    level_all.append(np.full(active_idx.size, idx, dtype=np.int64))
                    step_all.append(np.full(active_idx.size, step, dtype=np.int32))
                    episode_all.append(
                        np.full(active_idx.size, episode, dtype=np.int32)
                    )
                    agent_all.append(active_idx.astype(np.int32))

                obs, rewards, dones, _timeouts, _infos = env.step(
                    actions.reshape(-1, 1)
                )
                active &= ~np.asarray(dones, dtype=bool)
                if not active.any():
                    break
            if episode + 1 < args.episodes:
                obs, _ = env.reset()
    finally:
        env.close()

    if not obs_all:
        return 0
    np.savez_compressed(
        data_path,
        obs=np.concatenate(obs_all).astype(np.float32),
        probs=np.concatenate(prob_all).astype(np.float32),
        actions=np.concatenate(action_all).astype(np.int64),
        level=np.concatenate(level_all).astype(np.int64),
        step=np.concatenate(step_all).astype(np.int32),
        episode=np.concatenate(episode_all).astype(np.int32),
        agent=np.concatenate(agent_all).astype(np.int32),
    )
    print(f"[dagger {level}] samples={len(np.concatenate(action_all))}", flush=True)
    return len(np.concatenate(action_all))


def resolve_model_path(args: argparse.Namespace, model: str | None = None) -> Path:
    path = Path(model or args.model)
    return path if path.is_absolute() else args.output_dir / path


def load_student(path: Path, obs_encoding: str | None = None) -> StudentPolicy:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    input_dim = int(payload["input_dim"])
    payload_encoding = str(
        payload.get("obs_encoding", obs_spec.ENCODING_LEGACY)
    )
    expected_size = obs_spec.expected_obs_size(payload_encoding)
    if input_dim != expected_size:
        raise ValueError(
            f"{path} declares {payload_encoding} observations but has "
            f"input_dim={input_dim}, expected {expected_size}"
        )
    if obs_encoding is not None and obs_encoding != payload_encoding:
        raise ValueError(
            f"{path} was trained with --obs-encoding={payload_encoding}, "
            f"not {obs_encoding}"
        )
    student = StudentPolicy(
        input_dim=input_dim,
        hidden=int(payload["hidden"]),
    )
    student.load_state_dict(payload["state_dict"])
    student.eval()
    student.obs_encoding = payload_encoding
    student.input_dim = input_dim
    student.action_repeat = int(payload.get("action_repeat", 4))
    return student


def export_student(args: argparse.Namespace) -> None:
    """Export a legacy or multi-hot student to the versioned JSON runtime."""
    model_path = resolve_model_path(args)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    input_dim = int(payload["input_dim"])
    obs_encoding = str(
        payload.get("obs_encoding", obs_spec.ENCODING_LEGACY)
    )
    expected_size = obs_spec.expected_obs_size(obs_encoding)
    if input_dim != expected_size:
        raise ValueError(
            f"{model_path} declares {obs_encoding} observations but has "
            f"input_dim={input_dim}, expected {expected_size}"
        )
    if payload.get("head_mode") not in (None, "shared"):
        raise ValueError(f"{model_path} uses a level-conditioned head; not exportable")

    state = payload["state_dict"]
    required = {
        "trunk.0.weight",
        "trunk.0.bias",
        "trunk.2.weight",
        "trunk.2.bias",
        "heads.weight",
        "heads.bias",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"{model_path} is missing state keys: {sorted(missing)}")

    w1 = state["trunk.0.weight"].detach().cpu().numpy().astype(np.float32)
    b1 = state["trunk.0.bias"].detach().cpu().numpy().astype(np.float32)
    w2 = state["trunk.2.weight"].detach().cpu().numpy().astype(np.float32)
    b2 = state["trunk.2.bias"].detach().cpu().numpy().astype(np.float32)
    w3 = state["heads.weight"].detach().cpu().numpy().astype(np.float32)
    b3 = state["heads.bias"].detach().cpu().numpy().astype(np.float32)
    hidden_sizes = [int(w1.shape[0]), int(w2.shape[0])]
    if w1.shape[1] != input_dim or w2.shape[1] != hidden_sizes[0] or w3.shape[1] != hidden_sizes[1]:
        raise ValueError(f"{model_path} has inconsistent layer shapes")

    weights = {
        "w1": w1.reshape(-1).tolist(),
        "b1": b1.reshape(-1).tolist(),
        "w2": w2.reshape(-1).tolist(),
        "b2": b2.reshape(-1).tolist(),
        "w3": w3.reshape(-1).tolist(),
        "b3": b3.reshape(-1).tolist(),
    }

    def json_forward(x: np.ndarray) -> np.ndarray:
        h1 = np.maximum(x @ w1.T + b1, 0.0)
        h2 = np.maximum(h1 @ w2.T + b2, 0.0)
        return h2 @ w3.T + b3

    rng = np.random.default_rng(args.verify_seed)
    samples = rng.standard_normal((args.verify_samples, input_dim), dtype=np.float32)
    student = StudentPolicy(input_dim=input_dim, hidden=int(payload["hidden"]))
    student.load_state_dict(state)
    student.eval()
    with torch.no_grad():
        torch_logits = student(torch.from_numpy(samples)).numpy()
    json_logits = json_forward(samples)
    max_abs = float(np.max(np.abs(torch_logits - json_logits)))
    argmax_match = float(
        (torch_logits.argmax(axis=1) == json_logits.argmax(axis=1)).mean()
    )
    if max_abs > 1e-4 or argmax_match < 1.0:
        raise RuntimeError(
            f"export verification failed: max_abs={max_abs:.6g} "
            f"argmax_match={argmax_match:.4f}"
        )

    output_path = (
        Path(args.output)
        if args.output
        else args.output_dir / "export" / f"{model_path.stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "format": "dragon-jump-mlp",
        "spec_version": obs_spec.spec_version(obs_encoding),
        "model_name": model_path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": model_path.name,
        "obs_encoding": obs_encoding,
        "observation_size": expected_size,
        "input_size": input_dim,
        "hidden_sizes": hidden_sizes,
        "activation": "relu",
        "output_size": 2,
        "action_names": ["idle", "jump"],
        "action_repeat": int(payload.get("action_repeat", 4)),
        "level_conditioning": "none",
        "trained_levels": payload.get("trained_levels", []),
        "missing_levels": payload.get("missing_levels", []),
        "weights": weights,
    }
    with output_path.open("w") as handle:
        json.dump(document, handle, separators=(",", ":"))
    print(
        f"[export] wrote {output_path} "
        f"({output_path.stat().st_size} bytes, hidden={hidden_sizes}, "
        f"verify_max_abs={max_abs:.3g}, argmax_match={argmax_match:.3f})",
        flush=True,
    )


def eval_student(args: argparse.Namespace, env_path: str) -> None:
    model_path = resolve_model_path(args)
    student = load_student(model_path)
    obs_encoding = student.obs_encoding
    action_repeat = int(getattr(student, "action_repeat", args.action_repeat))
    levels = parse_levels(args.levels)
    summary = []
    run_name = args.eval_name or model_path.stem
    out_dir = args.output_dir / "eval" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for level in levels:
        idx = level_index(level)
        log_path = out_dir / f"godot-{level}.log"
        env = GodotEnv(
            env_path=env_path,
            port=args.eval_port,
            show_window=args.show,
            seed=args.seed + idx,
            action_repeat=action_repeat,
            speedup=args.speedup,
            nb_agents=args.agents,
            level=level,
            convert_action_space=True,
            log_path=str(log_path),
            env=isolated_home(args.output_dir, f"eval_{level}"),
            obs_encoding=obs_encoding,
        )
        n = args.agents
        done_once = np.zeros(n, dtype=bool)
        finished = np.zeros(n, dtype=bool)
        finish_times = np.full(n, np.inf, dtype=np.float64)
        rewards_sum = np.zeros(n, dtype=np.float64)
        steps_run = np.zeros(n, dtype=np.int32)
        efficiency = np.full(n, np.nan, dtype=np.float64)
        flow = np.full(n, np.nan, dtype=np.float64)
        deaths = np.zeros(n, dtype=np.int32)
        last_infos: list[dict] = [{} for _ in range(n)]
        try:
            obs, _ = env.reset()
            for step in range(args.max_decisions):
                obs_np = np.asarray(
                    [obs[i]["obs"] for i in range(n)],
                    dtype=np.float32,
                )
                with torch.no_grad():
                    logits = student(torch.from_numpy(obs_np))
                    actions = logits.argmax(dim=1).cpu().numpy()
                actions[done_once] = 0

                obs, rewards, dones, _timeouts, infos = env.step(
                    actions.reshape(-1, 1)
                )
                for i in range(n):
                    if done_once[i]:
                        continue
                    last_infos[i] = dict(infos[i]) if infos[i] else {}
                    rewards_sum[i] += float(rewards[i])
                    steps_run[i] = step + 1
                    if dones[i]:
                        done_once[i] = True
                        finished[i] = float(rewards[i]) > 0.0
                        if finished[i]:
                            try:
                                finish_times[i] = float(infos[i]["run_time"])
                            except (KeyError, TypeError, ValueError):
                                finish_times[i] = float("nan")
                            if (
                                not np.isfinite(finish_times[i])
                                or finish_times[i] <= 0.0
                            ):
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
        death_count = int(deaths.sum())
        timeout_count = int((~done_once).sum())
        clean_times = finish_times[np.isfinite(finish_times)]
        best = float(np.min(clean_times)) if clean_times.size else None
        median = float(np.median(clean_times)) if clean_times.size else None
        mean_time = float(np.mean(clean_times)) if clean_times.size else None
        time_cv = (
            float(np.std(clean_times) / mean_time)
            if clean_times.size and mean_time
            else None
        )
        efficiency_mean = (
            float(np.nanmean(efficiency))
            if np.isfinite(efficiency).any()
            else None
        )
        flow_mean = (
            float(np.nanmean(flow))
            if np.isfinite(flow).any()
            else None
        )
        row = {
            "level": level,
            "agents": n,
            "finishes": finish_count,
            "deaths": death_count,
            "timeouts": timeout_count,
            "best_time": f"{best:.6f}" if best is not None else "",
            "median_time": f"{median:.6f}" if median is not None else "",
            "mean_time": (
                f"{mean_time:.6f}" if mean_time is not None else ""
            ),
            "time_cv": f"{time_cv:.6f}" if time_cv is not None else "",
            "efficiency_mean": (
                f"{efficiency_mean:.6f}" if efficiency_mean is not None else ""
            ),
            "flow_mean": (
                f"{flow_mean:.6f}" if flow_mean is not None else ""
            ),
            "mean_reward": f"{float(rewards_sum.mean()):.2f}",
            "teacher": "yes" if teacher_path(level) is not None else "no",
        }
        summary.append(row)
        print(
            f"[eval {level}] finishes={finish_count}/{n} "
            f"deaths={death_count} timeouts={timeout_count} "
            f"best={row['best_time'] or 'none'} "
            f"median={row['median_time'] or 'none'} "
            f"eff={row['efficiency_mean'] or 'none'} "
            f"flow={row['flow_mean'] or 'none'} teacher={row['teacher']}",
            flush=True,
        )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "level",
                "agents",
                "finishes",
                "deaths",
                "timeouts",
                "best_time",
                "median_time",
                "mean_time",
                "time_cv",
                "efficiency_mean",
                "flow_mean",
                "mean_reward",
                "teacher",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    print(f"[eval] summary saved to {summary_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Run teachers and save datasets")
    collect.add_argument("--levels", default="all")
    collect.add_argument("--episodes", type=int, default=2)
    collect.add_argument("--agents", type=int, default=100)
    collect.add_argument("--max-decisions", type=int, default=150)
    collect.add_argument("--action-repeat", type=int, default=4)
    collect.add_argument("--speedup", type=float, default=8.0)
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--base-port", type=int, default=11008)
    collect.add_argument("--show", action="store_true")
    collect.add_argument(
        "--obs-encoding",
        default=obs_spec.ENCODING_LEGACY,
        choices=obs_spec.SUPPORTED_ENCODINGS,
        help="Observation contract emitted by the game.",
    )
    collect.add_argument(
        "--teacher-root",
        type=Path,
        default=None,
        help="Optional root containing <level>/ppo_pilot.zip teachers.",
    )
    collect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    train = sub.add_parser("train", help="Train the student from saved data")
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--hidden", type=int, default=128)
    train.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Training-only dropout between hidden layers; export is always off.",
    )
    train.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Model initialization seed (defaults to --seed).",
    )
    train.add_argument("--model-name", default="student.pt")
    train.add_argument("--val-fraction", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="Student training device; auto prefers CUDA, then MPS, then CPU.",
    )
    train.add_argument(
        "--action-repeat",
        type=int,
        default=4,
        help="Control cadence used to collect the dataset and exported JSON.",
    )
    train.add_argument(
        "--obs-encoding",
        default=obs_spec.ENCODING_LEGACY,
        choices=obs_spec.SUPPORTED_ENCODINGS,
        help="Observation contract used by the collected dataset.",
    )
    train.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    dagger = sub.add_parser(
        "dagger",
        help="Run student rollouts and label student-visited states with teachers",
    )
    dagger.add_argument("--levels", default="all")
    dagger.add_argument("--model", default="student.pt")
    dagger.add_argument("--episodes", type=int, default=2)
    dagger.add_argument("--agents", type=int, default=100)
    dagger.add_argument("--max-decisions", type=int, default=150)
    dagger.add_argument("--action-repeat", type=int, default=4)
    dagger.add_argument("--speedup", type=float, default=8.0)
    dagger.add_argument(
        "--rollout-temperature",
        type=float,
        default=1.0,
        help="Student rollout temperature; >1 explores more for DAgger coverage.",
    )
    dagger.add_argument("--seed", type=int, default=42)
    dagger.add_argument("--base-port", type=int, default=11008)
    dagger.add_argument("--show", action="store_true")
    dagger.add_argument(
        "--obs-encoding",
        default=obs_spec.ENCODING_LEGACY,
        choices=obs_spec.SUPPORTED_ENCODINGS,
        help="Observation contract shared by student, teacher, and game.",
    )
    dagger.add_argument(
        "--teacher-root",
        type=Path,
        default=None,
        help="Optional root containing <level>/ppo_pilot.zip teachers.",
    )
    dagger.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    eval_parser = sub.add_parser("eval", help="Evaluate the student on all levels")
    eval_parser.add_argument("--levels", default="all")
    eval_parser.add_argument("--model", default="student.pt")
    eval_parser.add_argument("--eval-name")
    eval_parser.add_argument("--agents", type=int, default=100)
    eval_parser.add_argument("--max-decisions", type=int, default=150)
    eval_parser.add_argument("--action-repeat", type=int, default=4)
    eval_parser.add_argument("--speedup", type=float, default=8.0)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--eval-port", type=int, default=12000)
    eval_parser.add_argument("--show", action="store_true")
    eval_parser.add_argument(
        "--obs-encoding",
        default=obs_spec.ENCODING_LEGACY,
        choices=obs_spec.SUPPORTED_ENCODINGS,
        help="Observation contract; ignored when the checkpoint records one.",
    )
    eval_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    export_parser = sub.add_parser(
        "export",
        help="Export a legacy or multi-hot student checkpoint to versioned JSON",
    )
    export_parser.add_argument("--model", default="student.pt")
    export_parser.add_argument("--output", type=Path)
    export_parser.add_argument("--verify-samples", type=int, default=256)
    export_parser.add_argument("--verify-seed", type=int, default=0)
    export_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir = args.output_dir.resolve()
    (args.output_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval").mkdir(parents=True, exist_ok=True)
    env_path = resolve_env_path()

    if args.command == "collect":
        levels = parse_levels(args.levels)
        results = []
        for level in levels:
            path = teacher_path(level, args.teacher_root)
            if path is None:
                print(f"[collect {level}] no teacher snapshot; skipped", flush=True)
                continue
            results.append(collect_level(level, path, args, env_path))
        print(f"[collect] {len(results)} levels", flush=True)
    elif args.command == "train":
        train_student(args)
    elif args.command == "dagger":
        student = load_student(resolve_model_path(args))
        levels = parse_levels(args.levels)
        total = 0
        for level in levels:
            path = teacher_path(level, args.teacher_root)
            if path is None:
                print(f"[dagger {level}] no teacher snapshot; skipped", flush=True)
                continue
            teacher = PPO.load(str(path), device="cpu")
            total += dagger_collect(level, student, teacher, args, env_path)
        print(f"[dagger] {total} labelled student states", flush=True)
    elif args.command == "eval":
        eval_student(args, env_path)
    elif args.command == "export":
        export_student(args)


if __name__ == "__main__":
    main()
