# %%
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from processing.feature_engineering import compute_can_jump, compute_raycast_sensors
from processing.frame_visualizer import FrameVisualizer
from utils import load_observations_by_session, setup_environment

# %%
DEFAULT_DATA_DIR = "../data"


def engineer_features(observation, session_info: dict) -> list[float]:
    """Convert one raw observation into a flat feature vector (same as notebooks/05_linear_regression.ipynb)."""
    visualizer = FrameVisualizer()
    parsed_obs = visualizer.parse_observation(observation)
    grid = parsed_obs["grid"]
    sensor_values = parsed_obs["sensor_values"]

    can_jump_feature = compute_can_jump(sensor_values, session_info)
    sensor_features = list(sensor_values.values())

    raycasts, detected_object_types = compute_raycast_sensors(grid)

    tracked_object_types = [
        visualizer.get_class_index("Empty"),
        visualizer.get_class_index("Wall"),
        visualizer.get_class_index("Spikes"),
    ]

    object_type_to_col = {
        object_type: col for col, object_type in enumerate(tracked_object_types)
    }
    object_type_one_hot = np.zeros(
        (len(detected_object_types), len(tracked_object_types)), dtype=np.float32
    )

    for row, object_type in enumerate(detected_object_types):
        col = object_type_to_col.get(object_type)
        if col is not None:
            object_type_one_hot[row, col] = 1.0

    one_hot_features = object_type_one_hot.flatten().tolist()
    return [can_jump_feature] + raycasts + one_hot_features + sensor_features


def aggregate_conflicting_examples(
    features,
    actions,
    groups,
    *,
    deduplicate: bool = True,
):
    """If deduplicate=True, merge rows with identical (rounded) features and majority vote labels.

    Weights are the number of merged rows. If deduplicate=False, keep every frame as its own row
    with weight 1.0 (no merging / majority resolution).
    """
    if not deduplicate:
        return (
            np.asarray(features, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.ones(len(actions), dtype=np.float32),
            np.asarray(groups),
        )

    grouped_labels = defaultdict(list)
    grouped_group = {}
    for feature_vector, action, group in zip(features, actions, groups, strict=True):
        key = tuple(np.round(feature_vector, 5))
        grouped_labels[key].append(int(action))
        grouped_group[key] = group

    aggregated_features = []
    aggregated_actions = []
    sample_weights = []
    aggregated_groups = []

    for key, labels in grouped_labels.items():
        action_counts = Counter(labels)
        majority_action = action_counts.most_common(1)[0][0]
        aggregated_features.append(np.array(key, dtype=np.float32))
        aggregated_actions.append(majority_action)
        sample_weights.append(len(labels))
        aggregated_groups.append(grouped_group[key])

    return (
        np.asarray(aggregated_features, dtype=np.float32),
        np.asarray(aggregated_actions, dtype=np.int64),
        np.asarray(sample_weights, dtype=np.float32),
        np.asarray(aggregated_groups),
    )


def load_training_engineered_xy(
    data_dir: str = DEFAULT_DATA_DIR,
    *,
    deduplicate: bool = True,
):
    all_data = {}
    for i, file in enumerate(sorted(Path(data_dir).glob("*.jsonl"))):
        observations_by_session = load_observations_by_session(file)
        for session_id, all_steps in observations_by_session.items():
            all_data[f"{i}_{session_id}"] = all_steps

    engineered_features = []
    actions = []
    groups = []

    for session_id, all_steps in all_data.items():
        session_info = {"prev_action": 0}
        for step in all_steps:
            engineered_features.append(engineer_features(step["state"], session_info))
            actions.append(step["action"])
            groups.append(session_id)
            session_info["prev_action"] = int(step["action"])

    if not engineered_features:
        return None, None, None, None

    X, y, sample_weights, sample_groups = aggregate_conflicting_examples(
        engineered_features, actions, groups, deduplicate=deduplicate
    )
    return X, y, sample_weights, sample_groups


class JumpMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropouts: float | Sequence[float] | None = None
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer size.")
        n_hidden = len(hidden_dims)
        if dropouts is None:
            dropout_ps = [0.0] * n_hidden
        elif isinstance(dropouts, (int, float)):
            dropout_ps = [float(dropouts)] * n_hidden
        else:
            if len(dropouts) != n_hidden:
                raise ValueError(
                    f"dropouts length ({len(dropouts)}) must match hidden_dims length ({n_hidden})."
                )
            dropout_ps = [float(p) for p in dropouts]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h, p in zip(hidden_dims, dropout_ps, strict=True):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if p > 0.0:
                layers.append(nn.Dropout(p))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _weighted_accuracy(logits, targets, weights) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).float()
    return float((correct * weights).sum() / weights.sum())


@torch.no_grad()
def _weighted_eval_loss(model, x, y, w, criterion) -> float:
    model.eval()
    logits = model(x)
    loss_per_row = criterion(logits, y)
    return float((loss_per_row * w).sum() / w.sum())


def plot_training_loss(
    train_loss_history: list[float],
    val_loss_history: list[float],
    output_path: str = "training_loss.png",
    show_plot: bool = False,
):
    epochs = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        train_loss_history,
        marker="o",
        linewidth=1.5,
        label="Train",
    )
    plt.plot(
        epochs,
        val_loss_history,
        marker="s",
        linewidth=1.5,
        label="Validation",
    )
    plt.title("Train vs Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    print(f"Saved training loss plot to: {output_path}")
    if show_plot:
        plt.show()
    plt.close()


def train_pytorch_classifier(
    data_dir: str = DEFAULT_DATA_DIR,
    deduplicate: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-2,
    weight_decay: float = 1e-2,
    lr_decay: bool = True,
    lr_decay_factor: float = 0.8,
    lr_decay_patience: int = 15,
    mlp_hidden_dims: tuple[int, ...] = (128, 64),
    mlp_dropouts: float | Sequence[float] | None = 0.2,
    early_stopping: bool = True,
    early_stopping_patience: int = 40,
    early_stopping_min_delta: float = 0.0,
    loss_plot_path: str = "training_loss.png",
    show_loss_plot: bool = True,
):
    result = load_training_engineered_xy(data_dir, deduplicate=deduplicate)
    if result[0] is None:
        raise FileNotFoundError(f"No training rows found under {data_dir} (*.jsonl).")
    X, y, sample_weights, sample_groups = result

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=sample_groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    w_train, w_test = sample_weights[train_idx], sample_weights[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(np.max(y)) + 1

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(w_train),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    x_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)
    w_test_t = torch.from_numpy(w_test).to(device)

    # Keep the same class-balance behavior idea as the random-forest script.
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()

    model = JumpMLP(
        input_dim=X.shape[1],
        output_dim=num_classes,
        hidden_dims=mlp_hidden_dims,
        dropouts=mlp_dropouts,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device), reduction="none"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler: ReduceLROnPlateau | None = None
    if lr_decay:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_decay_factor,
            patience=lr_decay_patience,
            min_lr=1e-7,
        )

    model.train()
    loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        running_weighted_loss = 0.0
        running_weight = 0.0
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            logits = model(xb)
            loss_per_row = criterion(logits, yb)
            loss = (loss_per_row * wb).sum() / wb.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_weighted_loss += float((loss_per_row * wb).sum().item())
            running_weight += float(wb.sum().item())

        avg_loss = running_weighted_loss / max(running_weight, 1e-8)
        loss_history.append(avg_loss)

        val_loss = _weighted_eval_loss(model, x_test_t, y_test_t, w_test_t, criterion)
        val_loss_history.append(val_loss)
        model.train()

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} - train loss: {avg_loss:.4f}, "
                f"val loss: {val_loss:.4f}, lr: {current_lr:.2e}"
            )

        if early_stopping:
            improved = val_loss < best_val_loss - early_stopping_min_delta
            if improved:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no val improvement for {early_stopping_patience} epochs; "
                    f"best val loss: {best_val_loss:.4f})."
                )
                break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test_t)
        accuracy = _weighted_accuracy(test_logits, y_test_t, w_test_t)
    majority_label = int(np.argmax(np.bincount(y_train, minlength=num_classes)))
    baseline_acc = float(
        ((y_test == majority_label).astype(np.float32) * w_test).sum() / w_test.sum()
    )
    print(f"PyTorch DNN accuracy (hold-out, session split): {accuracy:.3f}")
    print(
        f"Majority-class baseline accuracy (hold-out, same split): {baseline_acc:.3f}"
    )
    plot_training_loss(
        train_loss_history=loss_history,
        val_loss_history=val_loss_history,
        output_path=loss_plot_path,
        show_plot=show_loss_plot,
    )

    return model, device


def agent_brain(observation, session_info, step_count, model, device):
    if step_count == 0:
        return [1]

    # Inference must run in eval mode so dropout (if any) is disabled.
    model.eval()
    engineered = np.asarray(engineer_features(observation, session_info), dtype=np.float32)
    x = torch.from_numpy(engineered).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
    return [pred]


def main():
    model, device = train_pytorch_classifier()

    env = setup_environment()
    obs = env.reset()
    nb_agents = len(obs["obs"])
    session_infos = [{"prev_action": 0} for _ in range(nb_agents)]

    step_count = 0
    while True:
        actions = [
            agent_brain(obs["obs"][i], session_infos[i], step_count, model, device)
            for i in range(nb_agents)
        ]
        actions = np.array(actions, dtype=np.int64)

        for i in range(nb_agents):
            session_infos[i]["prev_action"] = int(np.asarray(actions[i]).flat[0])

        obs, reward, done, info = env.step(actions)
        if any(done):
            break
        step_count += 1

    env.close()


if __name__ == "__main__":
    main()

# %%
