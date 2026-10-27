"""Train a grid autoencoder and a behavior-cloned PPO head from saved states.

The autoencoder sees only unique 7x7 tile grids collected from the existing
legacy-58 distillation datasets. The PPO head is trained on frozen encoder
features plus the exact nine scalar tail values, using the teacher action
probabilities as soft labels. The resulting head is a warm start for PPO.

Usage:
    python src/22_train_legacy_autoencoder.py
    python src/22_train_legacy_autoencoder.py --latent-dim 64 --epochs 120
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

from wrappers.legacy_autoencoder import (
    ENCODER_VOCAB,
    GRID_VALUES,
    TAIL_VALUES,
    LegacyGridAutoencoder,
    LegacyGridEncoder,
    OneHotLegacyGridAutoencoder,
    build_grid_encoder,
)

REPO_DIR = Path(__file__).resolve().parent.parent


def find_legacy_npz(root: Path):
    """Yield ``(path, payload)`` for every 58-value observation dataset."""
    for path in sorted(root.glob("**/dataset/*.npz")):
        try:
            payload = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            continue
        if "obs" not in payload.files:
            continue
        obs = payload["obs"]
        if obs.ndim != 2 or obs.shape[1] != GRID_VALUES + TAIL_VALUES:
            continue
        yield path, payload


def canonical_grids(observations: np.ndarray) -> np.ndarray:
    raw = np.rint(observations[:, :GRID_VALUES]).astype(np.int16)
    return np.where(raw < 0, ENCODER_VOCAB - 1, raw).astype(np.uint8)


def load_unique_grids(
    data_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    observations: list[np.ndarray] = []
    sources: list[dict] = []
    for path, payload in find_legacy_npz(data_root):
        obs = np.asarray(payload["obs"], dtype=np.float32)
        observations.append(obs)
        sources.append(
            {
                "path": str(path.relative_to(REPO_DIR)),
                "rows": int(len(obs)),
                "unique_grids": int(len(np.unique(canonical_grids(obs), axis=0))),
            }
        )
        print(f"loaded {path.relative_to(REPO_DIR)}: {len(obs):,} rows", flush=True)

    if not observations:
        raise FileNotFoundError(f"no 58-value obs datasets under {data_root}")

    all_obs = np.concatenate(observations, axis=0)
    grids = canonical_grids(all_obs)
    unique, counts = np.unique(grids, axis=0, return_counts=True)
    print(
        f"candidates={len(all_obs):,} unique_grids={len(unique):,} "
        f"unique_full_states={len(np.unique(all_obs, axis=0)):,}",
        flush=True,
    )
    coverage = {
        "candidate_rows": int(len(all_obs)),
        "unique_grids": int(len(unique)),
        "unique_full_states": int(len(np.unique(all_obs, axis=0))),
        "grid_value_counts": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(unique, return_counts=True))
        },
        "cell_value_counts": {
            str(label): int(count)
            for label, count in enumerate(
                np.bincount(unique.ravel(), minlength=ENCODER_VOCAB)
            )
        },
        "sources": sources,
    }
    return unique, counts, coverage


def class_weights(unique_grids: np.ndarray) -> torch.Tensor:
    counts = np.bincount(
        unique_grids.ravel(), minlength=ENCODER_VOCAB
    ).astype(np.float64)
    present = counts > 0
    weights = np.zeros(ENCODER_VOCAB, dtype=np.float32)
    if present.any():
        # Inverse-square-root balancing keeps rare void/hazard cells visible
        # without making the loss explode for a handful of examples.
        weights[present] = (
            counts[present].sum() / (present.sum() * counts[present])
        ) ** 0.5
        weights = np.clip(weights, 0.05, 10.0).astype(np.float32)
    return torch.from_numpy(weights)


def make_loader(
    grids: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    inputs = np.zeros((len(grids), GRID_VALUES + TAIL_VALUES), dtype=np.float32)
    inputs[:, :GRID_VALUES] = grids.astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(grids.astype(np.int64)).view(-1, 7, 7),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_autoencoder(
    model: LegacyGridAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    exact = 0
    total = 0
    cells = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _latent = model(inputs)
            total_loss += float(criterion(logits, targets).item()) * len(inputs)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == targets).sum().item())
            exact += int(
                (predictions == targets).reshape(len(inputs), -1).all(dim=1).sum()
            )
            total += len(inputs)
            cells += targets.numel()
    return (
        total_loss / max(1, total),
        correct / max(1, cells),
        exact / max(1, total),
    )


def train_autoencoder(
    unique_grids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    holdout_grids: np.ndarray | None = None,
) -> tuple[LegacyGridEncoder, dict]:
    rng = np.random.default_rng(args.seed)
    if holdout_grids is not None and len(holdout_grids):
        holdout_keys = {row.tobytes() for row in holdout_grids}
        train_mask = np.asarray(
            [row.tobytes() not in holdout_keys for row in unique_grids],
            dtype=bool,
        )
        train_grids = unique_grids[train_mask]
        val_grids = holdout_grids
        if len(train_grids) == 0:
            raise ValueError("holdout removed every training grid")
    else:
        order = rng.permutation(len(unique_grids))
        val_count = max(1, int(round(len(unique_grids) * args.val_fraction)))
        val_idx = order[:val_count]
        train_idx = order[val_count:]
        if len(train_idx) == 0:
            raise ValueError("not enough unique grids to train")
        train_grids = unique_grids[train_idx]
        val_grids = unique_grids[val_idx]

    train_loader = make_loader(train_grids, args.batch_size, True)
    val_loader = make_loader(val_grids, args.batch_size, False)

    if args.input_encoding == "one_hot":
        model = OneHotLegacyGridAutoencoder(
            latent_dim=args.latent_dim,
            conv_channels=tuple(args.conv_channels),
        ).to(device)
    else:
        model = LegacyGridAutoencoder(
            latent_dim=args.latent_dim,
            embedding_dim=args.embedding_dim,
            conv_channels=tuple(args.conv_channels),
        ).to(device)
    weights = class_weights(unique_grids).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    history_path = args.output_dir / "autoencoder_history.csv"
    history_file = history_path.open("w", newline="")
    history_writer = csv.DictWriter(
        history_file,
        fieldnames=["epoch", "train_loss", "val_loss", "val_cell_acc", "val_exact_acc"],
    )
    history_writer.writeheader()

    best_loss = float("inf")
    best_encoder: dict[str, torch.Tensor] | None = None
    best_decoder: dict[str, torch.Tensor] | None = None
    best_metrics = {
        "val_loss": float("inf"),
        "val_cell_accuracy": 0.0,
        "val_exact_grid_accuracy": 0.0,
    }
    stale = 0
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, _latent = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                total_loss += float(loss.item()) * len(inputs)
                total += len(inputs)
            scheduler.step()

            train_loss = total_loss / max(1, total)
            val_loss, val_cell_acc, val_exact_acc = evaluate_autoencoder(
                model, val_loader, criterion, device
            )
            history_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_cell_acc": f"{val_cell_acc:.6f}",
                    "val_exact_acc": f"{val_exact_acc:.6f}",
                }
            )
            history_file.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"[ae {epoch:03d}] train={train_loss:.4f} "
                    f"val={val_loss:.4f} cell_acc={val_cell_acc:.4f} "
                    f"exact={val_exact_acc:.4f}",
                    flush=True,
                )

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_metrics = {
                    "val_loss": val_loss,
                    "val_cell_accuracy": val_cell_acc,
                    "val_exact_grid_accuracy": val_exact_acc,
                }
                best_encoder = {
                    key: value.detach().cpu().clone()
                    for key, value in model.encoder.state_dict().items()
                }
                best_decoder = {
                    key: value.detach().cpu().clone()
                    for key, value in model.decoder.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= args.autoencoder_patience:
                    print(
                        f"[ae] early stop at epoch {epoch}; "
                        f"best val={best_loss:.4f}",
                        flush=True,
                    )
                    break
    finally:
        history_file.close()

    if best_encoder is None or best_decoder is None:
        raise RuntimeError("autoencoder produced no checkpoint")

    checkpoint = {
        "encoder": best_encoder,
        "decoder": best_decoder,
        "latent_dim": args.latent_dim,
        "embedding_dim": args.embedding_dim,
        "conv_channels": list(args.conv_channels),
        "input_encoding": args.input_encoding,
        "tile_vocab": ENCODER_VOCAB,
        "grid_size": 7,
        "unique_grids": int(len(unique_grids)),
        "train_grids": int(len(train_grids)),
        "val_grids": int(len(val_grids)),
        "seed": args.seed,
        **best_metrics,
    }
    torch.save(checkpoint, args.output_dir / "autoencoder.pt")
    best_encoder_module = build_grid_encoder(
        input_encoding=args.input_encoding,
        latent_dim=args.latent_dim,
        embedding_dim=args.embedding_dim,
        conv_channels=tuple(args.conv_channels),
    )
    best_encoder_module.load_state_dict(best_encoder)
    best_encoder_module.eval()
    return best_encoder_module, checkpoint


def load_bc_data(
    data_root: Path,
    level: str,
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for path, payload in find_legacy_npz(data_root):
        if path.stem not in {level, f"dagger_{level}"}:
            continue
        if "probs" not in payload.files:
            continue
        probabilities.append(np.asarray(payload["probs"], dtype=np.float32))
        observations.append(np.asarray(payload["obs"], dtype=np.float32))
        print(f"loaded BC {path}: {len(payload['obs']):,}", flush=True)

    if not observations:
        raise FileNotFoundError(f"no {level} behavior-cloning data")

    obs = np.concatenate(observations, axis=0)
    probs = np.concatenate(probabilities, axis=0)
    unique, inverse, counts = np.unique(
        obs, axis=0, return_inverse=True, return_counts=True
    )
    averaged = np.zeros((len(unique), probs.shape[1]), dtype=np.float32)
    np.add.at(averaged, inverse, probs)
    averaged /= counts[:, None]
    row_sums = averaged.sum(axis=1, keepdims=True)
    averaged = np.divide(
        averaged,
        np.maximum(row_sums, 1e-8),
        out=np.zeros_like(averaged),
        where=row_sums > 0,
    )
    print(
        f"BC unique states={len(unique):,} from {len(obs):,} rows; "
        f"mean P(jump)={averaged[:, 1].mean():.3f}",
        flush=True,
    )
    return unique, averaged


def relabel_with_teacher(
    observations: np.ndarray,
    teacher_path: Path,
    device: torch.device,
) -> np.ndarray:
    """Replace BC soft labels with a stronger 58-input PPO teacher's probs."""
    teacher = PPO.load(str(teacher_path), device=device)
    if tuple(teacher.observation_space.shape) != (GRID_VALUES + TAIL_VALUES,):
        raise ValueError(
            f"{teacher_path} observes {teacher.observation_space.shape}; "
            "expected a 58-value legacy teacher"
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
        f"relabelled {len(observations):,} BC states with {teacher_path}; "
        f"mean P(jump)={probabilities[:, 1].mean():.3f}",
        flush=True,
    )
    return probabilities


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
            loss = F.cross_entropy(logits, targets.argmax(dim=1))
            total_loss += float(loss.item()) * len(features)
            correct += int(
                (logits.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            )
            total += len(features)
    return total_loss / max(1, total), correct / max(1, total)


def train_head(
    encoder: LegacyGridEncoder,
    observations: np.ndarray,
    probabilities: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    inputs = torch.from_numpy(observations.astype(np.float32, copy=False))
    encoder = encoder.to(device).eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(inputs), 4096):
            batch = inputs[start : start + 4096].to(device)
            latent = encoder(batch)
            chunks.append(
                torch.cat((latent, batch[:, GRID_VALUES:]), dim=1).cpu()
            )
    features = torch.cat(chunks, dim=0)
    targets = torch.from_numpy(probabilities.astype(np.float32, copy=False))
    input_dim = int(features.shape[1])
    print(f"BC feature dimension: {input_dim}", flush=True)

    rng = np.random.default_rng(args.seed + 1)
    order = rng.permutation(len(features))
    val_count = max(1, int(round(len(features) * args.val_fraction)))
    val_idx = order[:val_count]
    train_idx = order[val_count:]
    train_loader = DataLoader(
        TensorDataset(features[train_idx], targets[train_idx]),
        batch_size=args.bc_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(features[val_idx], targets[val_idx]),
        batch_size=args.bc_batch_size,
        shuffle=False,
    )

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
        head.parameters(), lr=args.bc_learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.bc_epochs)
    )

    history_path = args.output_dir / "policy_head_history.csv"
    history_file = history_path.open("w", newline="")
    history_writer = csv.DictWriter(
        history_file, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"]
    )
    history_writer.writeheader()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics = {"val_loss": float("inf"), "val_accuracy": 0.0}
    stale = 0
    try:
        for epoch in range(1, args.bc_epochs + 1):
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
            val_loss, val_accuracy = evaluate_head(head, val_loader, device)
            history_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_accuracy": f"{val_accuracy:.6f}",
                }
            )
            history_file.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == args.bc_epochs:
                print(
                    f"[head {epoch:03d}] train={train_loss:.4f} "
                    f"val={val_loss:.4f} acc={val_accuracy:.4f}",
                    flush=True,
                )
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_metrics = {
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in head.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= args.head_patience:
                    print(
                        f"[head] early stop at epoch {epoch}; "
                        f"best val={best_loss:.4f}",
                        flush=True,
                    )
                    break
    finally:
        history_file.close()

    if best_state is None:
        raise RuntimeError("behavior-cloned head produced no checkpoint")

    checkpoint = {
        "state_dict": best_state,
        "features_dim": input_dim,
        "net_arch": hidden,
        "activation": "tanh",
        "obs_encoding": "legacy",
        "bc_unique_states": int(len(features)),
        **best_metrics,
    }
    torch.save(checkpoint, args.output_dir / "policy_head.pt")
    print(
        f"saved {args.output_dir / 'policy_head.pt'} "
        f"(val acc={best_metrics['val_accuracy']:.4f})",
        flush=True,
    )
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_DIR / "runs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_DIR / "runs" / "legacy_autoencoder" / "unique_ae64_s42",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument(
        "--input-encoding",
        choices=("embedding", "one_hot"),
        default="embedding",
        help="Grid encoder input: learned tile embedding or explicit 7x7x17 one-hot.",
    )
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--conv-channels", type=int, nargs=2, default=[16, 32])
    parser.add_argument(
        "--load-autoencoder",
        type=Path,
        help="Skip autoencoder training and reuse this checkpoint.",
    )
    parser.add_argument(
        "--grids-path",
        type=Path,
        help="Optional .npy of unique 49-cell grids to train on instead of scanning datasets.",
    )
    parser.add_argument(
        "--holdout-grids-path",
        type=Path,
        help="Optional .npy of real grids held out from training and used for validation.",
    )
    parser.add_argument(
        "--teacher-model",
        type=Path,
        help="Optional 58-input PPO teacher used to relabel the BC states.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--autoencoder-patience", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--bc-level", default="1-11")
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-patience", type=int, default=15)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-bc", action="store_true")
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

    if args.grids_path is not None:
        loaded = np.load(args.grids_path.expanduser().resolve())
        loaded = np.rint(loaded).astype(np.int16)
        loaded = np.where(loaded < 0, ENCODER_VOCAB - 1, loaded)
        unique_grids, grid_counts = np.unique(
            loaded.astype(np.uint8), axis=0, return_counts=True
        )
        coverage = {
            "grids_path": str(args.grids_path.expanduser().resolve()),
            "unique_grids": int(len(unique_grids)),
            "cell_value_counts": {
                str(label): int(count)
                for label, count in enumerate(
                    np.bincount(unique_grids.ravel(), minlength=ENCODER_VOCAB)
                )
            },
        }
    else:
        unique_grids, grid_counts, coverage = load_unique_grids(args.data_root)
    np.save(args.output_dir / "unique_grids.npy", unique_grids)
    np.save(args.output_dir / "grid_counts.npy", grid_counts)
    holdout_grids: np.ndarray | None = None
    if args.holdout_grids_path is not None:
        holdout_grids = np.load(
            args.holdout_grids_path.expanduser().resolve()
        )
        holdout_grids = np.rint(holdout_grids).astype(np.int16)
        holdout_grids = np.unique(
            np.where(
                holdout_grids < 0,
                ENCODER_VOCAB - 1,
                holdout_grids,
            ).astype(np.uint8),
            axis=0,
        )
        np.save(args.output_dir / "holdout_grids.npy", holdout_grids)
        coverage["holdout_real_grids"] = int(len(holdout_grids))
    with (args.output_dir / "coverage.json").open("w") as handle:
        json.dump(coverage, handle, indent=2)
    print(
        f"covered grid values: {sorted(int(v) for v in np.unique(unique_grids))}",
        flush=True,
    )

    if args.load_autoencoder is not None:
        autoencoder_path = args.load_autoencoder.expanduser().resolve()
        checkpoint = torch.load(
            autoencoder_path, map_location="cpu", weights_only=False
        )
        args.latent_dim = int(checkpoint["latent_dim"])
        args.embedding_dim = int(checkpoint.get("embedding_dim", 8))
        args.conv_channels = [
            int(value) for value in checkpoint.get("conv_channels", [16, 32])
        ]
        args.input_encoding = str(
            checkpoint.get("input_encoding", "embedding")
        )
        encoder = build_grid_encoder(
            input_encoding=args.input_encoding,
            latent_dim=args.latent_dim,
            embedding_dim=args.embedding_dim,
            conv_channels=tuple(args.conv_channels),
        )
        encoder.load_state_dict(checkpoint["encoder"])
        encoder.eval()
        print(f"loaded autoencoder: {autoencoder_path}", flush=True)
    else:
        autoencoder_path = args.output_dir / "autoencoder.pt"
        encoder, checkpoint = train_autoencoder(
            unique_grids, args, device, holdout_grids=holdout_grids
        )
    print(
        f"autoencoder: cell_acc={checkpoint['val_cell_accuracy']:.4f} "
        f"exact={checkpoint['val_exact_grid_accuracy']:.4f} "
        f"val_loss={checkpoint['val_loss']:.4f}",
        flush=True,
    )

    if args.skip_bc:
        return
    bc_observations, bc_probabilities = load_bc_data(args.data_root, args.bc_level)
    if args.teacher_model is not None:
        teacher_path = args.teacher_model.expanduser().resolve()
        bc_probabilities = relabel_with_teacher(
            bc_observations, teacher_path, device
        )
    np.savez_compressed(
        args.output_dir / "bc_dataset.npz",
        obs=bc_observations,
        probs=bc_probabilities,
    )
    head_checkpoint = train_head(
        encoder,
        bc_observations,
        bc_probabilities,
        args,
        device,
    )
    summary = {
        "autoencoder": {
            "path": str(autoencoder_path),
            "latent_dim": args.latent_dim,
            "unique_grids": int(len(unique_grids)),
            "holdout_real_grids": (
                int(len(holdout_grids)) if holdout_grids is not None else 0
            ),
            **{
                key: checkpoint[key]
                for key in (
                    "val_loss",
                    "val_cell_accuracy",
                    "val_exact_grid_accuracy",
                )
            },
        },
        "policy_head": {
            "path": str(args.output_dir / "policy_head.pt"),
            "features_dim": head_checkpoint["features_dim"],
            "bc_unique_states": head_checkpoint["bc_unique_states"],
            "val_loss": head_checkpoint["val_loss"],
            "val_accuracy": head_checkpoint["val_accuracy"],
        },
        "teacher_model": (
            str(args.teacher_model.expanduser().resolve())
            if args.teacher_model is not None
            else None
        ),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
