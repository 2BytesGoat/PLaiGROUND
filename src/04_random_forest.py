from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from ml_forge.game.cleaning import aggregate_conflicting_examples
from ml_forge.game.features import FeatureEngineer
from utils import setup_environment


DEFAULT_DATA_DIR = "./data"


def engineer_features(observation, session_info: dict) -> list[float]:
    engineer = FeatureEngineer()
    return engineer.engineer_features(observation, session_info)


def load_training_engineered_xy(data_dir: str = DEFAULT_DATA_DIR):
    from ml_forge.game.recording import load_all_sessions

    all_data = load_all_sessions(data_dir)

    engineer = FeatureEngineer()
    engineered_features = []
    actions = []
    groups = []

    for session_id, all_steps in all_data.items():
        session_info = engineer.new_session_info()
        for step in all_steps:
            engineered_features.append(
                engineer.engineer_features(step["state"], session_info)
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


def agent_brain(observation, session_info, step_count, model, engineer=None):
    if step_count == 0:
        return [1]

    if engineer is None:
        engineer = FeatureEngineer()
    engineered = engineer.engineer_features(observation, session_info)
    pred = model.predict([engineered])[0]
    return [int(pred)]


def main():
    model = train_random_forest_classifier()

    env = setup_environment()
    obs = env.reset()
    nb_agents = len(obs["obs"])
    engineer = FeatureEngineer()
    session_infos = [engineer.new_session_info() for _ in range(nb_agents)]

    step_count = 0
    while True:
        actions = [
            agent_brain(obs["obs"][i], session_infos[i], step_count, model, engineer)
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