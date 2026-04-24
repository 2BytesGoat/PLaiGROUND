from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from processing.feature_engineering import compute_can_jump, compute_object_sensors
from processing.frame_visualizer import FrameVisualizer
from utils import load_observations_by_session, setup_environment


DEFAULT_DATA_DIR = "./data"


def engineer_features(observation, session_info: dict) -> list[float]:
    visualizer = FrameVisualizer()
    parsed_obs = visualizer.parse_observation(observation)
    grid = parsed_obs["grid"]
    sensor_values = parsed_obs["sensor_values"]
    sensors = list(sensor_values.values())

    can_jump = compute_can_jump(sensor_values, session_info)
    wall_index = visualizer.get_class_index("Wall")
    wall_distances = compute_object_sensors(grid, wall_index)

    return [can_jump] + wall_distances + sensors


def aggregate_conflicting_examples(features, actions, groups):
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


def load_training_engineered_xy(data_dir: str = DEFAULT_DATA_DIR):
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
            engineered_features.append(
                engineer_features(step["state"], session_info)
            )
            actions.append(step["action"])
            groups.append(session_id)
            session_info["prev_action"] = int(step["action"])

    if not engineered_features:
        return None, None, None, None

    X, y, sample_weights, sample_groups = aggregate_conflicting_examples(
        engineered_features, actions, groups
    )
    return X, y, sample_weights, sample_groups


def train_random_forest_classifier(
    data_dir: str = DEFAULT_DATA_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int | None = 12,
):
    result = load_training_engineered_xy(data_dir)
    if result[0] is None:
        raise FileNotFoundError(
            f"No training rows found under {data_dir} (*.jsonl)."
        )
    X, y, sample_weights, sample_groups = result

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups=sample_groups)
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    w_train, w_test = sample_weights[train_idx], sample_weights[test_idx]

    # Match notebook: ratio from training fold labels only (after session split).
    nb_ones = int(y_train.sum())
    nb_zeros = int(len(y_train) - nb_ones)
    ratio = 1.0 if nb_ones == 0 else nb_zeros / nb_ones
    class_weights = {0: 1.0, 1: ratio}

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weights,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    accuracy = model.score(X_test, y_test, sample_weight=w_test)
    print(f"Random forest accuracy (hold-out, session split): {accuracy:.3f}")
    return model


def agent_brain(observation, session_info, step_count, model):
    if step_count == 0:
        return [1]

    engineered = engineer_features(observation, session_info)
    pred = model.predict([engineered])[0]
    return [int(pred)]


def main():
    model = train_random_forest_classifier()

    env = setup_environment()
    obs = env.reset()
    nb_agents = len(obs["obs"])
    session_infos = [{"prev_action": 0} for _ in range(nb_agents)]

    step_count = 0
    while True:
        actions = [
            agent_brain(obs["obs"][i], session_infos[i], step_count, model)
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
