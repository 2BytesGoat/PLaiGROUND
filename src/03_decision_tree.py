from pathlib import Path

import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split

from utils import load_observations_by_session, setup_environment


DEFAULT_DATA_DIR = "./data"


def load_training_xy(data_dir: str = DEFAULT_DATA_DIR):
    all_data = {}
    for i, file in enumerate(sorted(Path(data_dir).glob("*.jsonl"))):
        observations_by_session = load_observations_by_session(file)
        for session_id, all_steps in observations_by_session.items():
            all_data[f"{i}_{session_id}"] = all_steps

    X, y = [], []
    for _session_id, all_steps in all_data.items():
        for step in all_steps:
            X.append(step["state"])
            y.append(step["action"])
    return X, y


def train_decision_tree_classifier(
    data_dir: str = DEFAULT_DATA_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = 3,
):
    X, y = load_training_xy(data_dir)
    if not X:
        raise FileNotFoundError(
            f"No training rows found under {data_dir} (*.jsonl)."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = tree.DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced")
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Decision tree accuracy (hold-out): {accuracy:.2f}")
    return model


def agent_brain(observation, step_count, model):
    # Important: You need to press jump for the first frame to start walking.
    if step_count == 0:
        return [1]

    pred = model.predict([observation])[0]
    return [int(pred)]


def main():
    model = train_decision_tree_classifier()

    env = setup_environment()
    obs = env.reset()
    nb_agents = len(obs["obs"])

    step_count = 0
    while True:
        actions = [agent_brain(obs["obs"][i], step_count, model) for i in range(nb_agents)]
        actions = np.array(actions, dtype=np.int64)
        obs, reward, done, info = env.step(actions)
        if any(done):
            break
        step_count += 1

    env.close()


if __name__ == "__main__":
    main()
