import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeduplicationStats:
    total_samples: int
    unique_samples: int
    duplicate_samples: int


def hash_state(state: list[float], *, decimals: int | None = None) -> str:
    arr = np.asarray(state)
    if decimals is not None and np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr, decimals=decimals)
    payload = arr.tolist()
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def deduplicate_rows_by_majority_label(
    rows: list[dict[str, list[float]]],
    *,
    state_key: str = "state",
    label_key: str = "action",
    group_key: str = "group",
    state_hash_fn = None,
) -> tuple[list[dict[str, list[float]]], DeduplicationStats]:
    hash_fn = state_hash_fn or hash_state
    grouped = defaultdict(
        lambda: {"state": None, "labels": [], "group": None}
    )

    total_samples = 0
    for row in rows:
        state = row[state_key]
        label = row[label_key]
        group = row.get(group_key)

        state_digest = hash_fn(state)
        grouped[state_digest]["state"] = state
        grouped[state_digest]["labels"].append(label)
        grouped[state_digest]["group"] = group
        total_samples += 1

    deduplicated_rows = []
    for item in grouped.values():
        counts = Counter(item["labels"])
        majority_label = counts.most_common(1)[0][0]
        deduplicated_rows.append(
            {
                state_key: item["state"],
                label_key: majority_label,
                group_key: item["group"],
                "sample_weight": int(sum(counts.values())),
                "label_counts": dict(counts),
            }
        )

    unique_samples = len(deduplicated_rows)
    stats = DeduplicationStats(
        total_samples=total_samples,
        unique_samples=unique_samples,
        duplicate_samples=max(0, total_samples - unique_samples),
    )
    return deduplicated_rows, stats


def build_training_arrays(
    rows: list[dict[str, list[float]]],
    *,
    feature_key = "state",
    label_key = "action",
    group_key = "group",
    weight_key = "sample_weight",
    feature_transform = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y, groups, sample_weights = [], [], [], []
    transform = feature_transform or (lambda x: x)

    for row in rows:
        X.append(transform(row[feature_key]))
        y.append(row[label_key])
        groups.append(row.get(group_key))
        sample_weights.append(row.get(weight_key, 1.0))

    return (
        np.asarray(X),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups),
        np.asarray(sample_weights, dtype=np.float32),
    )


def compute_binary_class_weights(
    labels: list[int],
    *,
    negative_label = 0,
    positive_label = 1,
) -> dict[int, float]:
    labels_arr = np.asarray(list(labels), dtype=np.int64)
    if labels_arr.size == 0:
        return {negative_label: 1.0, positive_label: 1.0}

    positive_count = int(np.sum(labels_arr == positive_label))
    negative_count = int(np.sum(labels_arr == negative_label))
    # If labels aren't only these two, the rest were ignored — avoid silent wrong counts.
    other = int(labels_arr.size - positive_count - negative_count)
    if other != 0:
        raise ValueError(
            f"Expected only labels {negative_label} and {positive_label}; "
            f"found {other} other label(s)."
        )

    if positive_count == 0 or negative_count == 0:
        return {negative_label: 1.0, positive_label: 1.0}

    if positive_count > negative_count:
        return {negative_label: positive_count / negative_count, positive_label: 1.0}
    if negative_count > positive_count:
        return {negative_label: 1.0, positive_label: negative_count / positive_count}
    return {negative_label: 1.0, positive_label: 1.0}
