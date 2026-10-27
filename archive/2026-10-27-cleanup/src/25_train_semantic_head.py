"""Train a 64x64 behavior-cloned head on the frozen semantic MAE latent.

The autoencoder reconstructs the 7x7x15 semantic grid. This script freezes it,
concatenates the exact 19-value tail, and trains a small MLP on 1-11 teacher
probabilities. It is the downstream test of whether the semantic latent carries
enough information for control.

Usage:
    python src/25_train_semantic_head.py \
      --autoencoder-path runs/legacy_autoencoder/semantic_mae64_all_s42/autoencoder.pt \
      --teacher-model runs/topology/multi_hot_8x8_w512_s42/best_model.zip \
      --output-dir runs/legacy_autoencoder/semantic_mae64_head_1-11_s42
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset

import obs_spec
from wrappers.semantic_autoencoder import (
    GRID_VALUES,
    TAIL_VALUES,
    SemanticGridEncoder,
)

REPO_DIR = Path(__file__).resolve().parent.parent
LEVELS = [f"1-{index}" for index in range(1, 17)]


def find_semantic_data(data_root: Path, level: str):
    for path in sorted(data_root.glob("**/dataset/*.npz")):
        if path.stem not in {level, f"dagger_{level}"}:
            continue
        try:
            payload = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            continue
        if "obs" not in payload.files or "probs" not in payload.files:
            continue
        if payload["obs"].shape[1] != obs_spec.MULTI_HOT_OBSERVATION_SIZE:
            continue
        yield path, payload


def load_states(
    data_root: Path,
    level: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    observations: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    sources = 0
    for path, payload in find_semantic_data(data_root, level):
        observations.append(np.asarray(payload["obs"], dtype=np.float32))
        probabilities.append(np.asarray(payload["probs"], dtype=np.float32))
        sources += 1
        print(f"loaded {path}: {len(payload['obs']):,}", flush=True)
    if not observations:
        raise FileNotFoundError(f"no semantic {level} data under {data_root}")
    obs = np.concatenate(observations, axis=0)
    probs = np.concatenate(probabilities, axis=0)
    unique, inverse, counts = np.unique(
        obs, axis=0, return_inverse=True, return_counts=True
    )
    averaged = np.zeros((len(unique), probs.shape[1]), dtype=np.float32)
    np.add.at(averaged, inverse, probs)
    averaged /= counts[:, None]
    row_sum = averaged.sum(axis=1, keepdims=True)
    averaged = np.divide(
        averaged,
        np.maximum(row_sum, 1e-8),
        out=np.zeros_like(averaged),
        where=row_sum > 0,
    )
    print(
        f"unique states={len(unique):,} from {len(obs):,} rows; "
        f"mean P(jump)={averaged[:, 1].mean():.3f}",
        flush=True,
    )
    return unique, averaged, {"sources": sources, "rows": int(len(obs))}


def relabel_with_teacher(
    observations: np.ndarray,
    teacher_path: Path,
    device: torch.device,
) -> np.ndarray:
    teacher = PPO.load(str(teacher_path), device=device)
    if tuple(teacher.observation_space.shape) != (
        obs_spec.MULTI_HOT_OBSERVATION_SIZE,
    ):
        raise ValueError(
            f"{teacher_path} observes {teacher.observation_space.shape}; "
            "expected 754-value multi_hot teacher"
        )
    teacher.policy.eval()
    inputs = torch.from_numpy(observations.astype(np.float32, copy=False))
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), 4096):
            batch = inputs[start : start + 4096].to(device)
            distribution = teacher.policy.get_distribution(batch)
            chunks.append(distribution.distribution.probs.cpu().numpy())
    probabilities = np.concatenate(chunks, axis=0).astype(np.float32)
    print(
        f"relabelled {len(observations):,} semantic states from {teacher_path}; "
        f"mean P(jump)={probabilities[:, 1].mean():.3f}",
        flush=True,
    )
    return probabilities


def load_encoder(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[SemanticGridEncoder, dict]:
    payload = torch.load(
        checkpoint_path.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    encoder = SemanticGridEncoder(
        latent_dim=int(payload["latent_dim"]),
        conv_channels=tuple(int(v) for v in payload["conv_channels"]),
    )
    encoder.load_state_dict(payload["encoder"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder.to(device), payload


def make_features(
    encoder: SemanticGridEncoder,
    observations: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    inputs = torch.from_numpy(observations.astype(np.float32, copy=False))
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(inputs), 4096):
            batch = inputs[start : start + 4096].to(device)
            latent = encoder(batch)
            chunks.append(
                torch.cat((latent, batch[:, GRID_VALUES:]), dim=1).cpu()
            )
    return torch.cat(chunks, dim=0)


def evaluate_head(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    head.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = head(features)
            total_loss += float(
                F.cross_entropy(logits, targets.argmax(dim=1)).item()
            ) * len(features)
            correct += int(
                (logits.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            )
            total += len(features)
    return total_loss / max(1, total), correct / max(1, total)


def train_head(
    features: torch.Tensor,
    probabilities: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    targets = torch.from_numpy(probabilities.astype(np.float32, copy=False))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(features))
    val_count = max(1, int(round(len(features) * args.val_fraction)))
    val_idx = order[:val_count]
    train_idx = order[val_count:]
    train_loader = DataLoader(
        TensorDataset(features[train_idx], targets[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(features[val_idx], targets[val_idx]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    input_dim = int(features.shape[1])
    hidden = list(args.net_arch)
    if len(hidden) != 2:
        raise ValueError("--net-arch must contain exactly two hidden sizes")
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.Tanh()))
        previous = width
    layers.append(nn.Linear(previous, targets.shape[1]))
    head = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    history_path = args.output_dir / "head_history.csv"
    history_file = history_path.open("w", newline="")
    writer = csv.DictWriter(
        history_file,
        fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"],
    )
    writer.writeheader()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    stale = 0
    try:
        for epoch in range(1, args.epochs + 1):
            head.train()
            total_loss = 0.0
            total = 0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = head(batch_features)
                loss = -(
                    batch_targets * F.log_softmax(logits, dim=1)
                ).sum(dim=1).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_features)
                total += len(batch_features)
            scheduler.step()
            train_loss = total_loss / max(1, total)
            val_loss, val_acc = evaluate_head(head, val_loader, device)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_accuracy": f"{val_acc:.6f}",
                }
            )
            history_file.flush()
            print(
                f"[head {epoch:03d}] train={train_loss:.4f} "
                f"val={val_loss:.4f} acc={val_acc:.4f}",
                flush=True,
            )
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_metrics = {"val_loss": val_loss, "val_accuracy": val_acc}
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(
                        f"[head] early stop at epoch {epoch}; best={best_loss:.4f}",
                        flush=True,
                    )
                    break
    finally:
        history_file.close()
    if best_state is None:
        raise RuntimeError("semantic head produced no checkpoint")
    checkpoint = {
        "state_dict": best_state,
        "features_dim": input_dim,
        "net_arch": hidden,
        "activation": "tanh",
        "obs_encoding": obs_spec.ENCODING_MULTI_HOT,
        **best_metrics,
    }
    torch.save(checkpoint, args.output_dir / "policy_head.pt")
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_DIR / "runs")
    parser.add_argument("--level", default="1-11")
    parser.add_argument(
        "--autoencoder-path",
        type=Path,
        default=REPO_DIR
        / "runs"
        / "legacy_autoencoder"
        / "semantic_mae64_all_s42"
        / "autoencoder.pt",
    )
    parser.add_argument(
        "--teacher-model",
        type=Path,
        default=REPO_DIR
        / "runs"
        / "topology"
        / "multi_hot_8x8_w512_s42"
        / "best_model.zip",
        help="Optional 754-input teacher used to relabel the states.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_DIR
        / "runs"
        / "legacy_autoencoder"
        / "semantic_mae64_head_1-11_s42",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"device={device} output={args.output_dir}", flush=True)
    observations, probabilities, metadata = load_states(args.data_root, args.level)
    if args.teacher_model is not None:
        probabilities = relabel_with_teacher(
            observations,
            args.teacher_model.expanduser().resolve(),
            device,
        )
    np.savez_compressed(
        args.output_dir / "semantic_bc_dataset.npz",
        obs=observations,
        probs=probabilities,
    )
    encoder, payload = load_encoder(args.autoencoder_path, device)
    features = make_features(encoder, observations, device)
    print(f"features={tuple(features.shape)}", flush=True)
    head_checkpoint = train_head(features, probabilities, args, device)
    summary = {
        "autoencoder": str(args.autoencoder_path.expanduser().resolve()),
        "autoencoder_latent": int(payload["latent_dim"]),
        "policy_head": str(args.output_dir / "policy_head.pt"),
        "features_dim": head_checkpoint["features_dim"],
        "net_arch": list(args.net_arch),
        "unique_states": int(len(observations)),
        **{
            key: head_checkpoint[key]
            for key in ("val_loss", "val_accuracy")
        },
        "teacher_model": (
            str(args.teacher_model.expanduser().resolve())
            if args.teacher_model is not None
            else None
        ),
        **metadata,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
