# %%
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import plot_tree

from utils import load_observations_by_session, setup_environment

""" Featrues to engineer
- distance to wall
- distance to spike
"""



def plot_action_distribution(actions):
    plt.hist(actions, bins=np.arange(-0.5, max(actions) + 1.5, 1), rwidth=0.8)
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title("Distribution of Actions in the Dataset")
    plt.xticks(np.arange(max(actions) + 1))
    plt.show()


def engineer_features(observation, prev_observation=None):
    obs = np.asarray(observation, dtype=np.float32)
    grid = obs[:-8]
    sensors = obs[-8:]

    side = int(np.sqrt(grid.size))
    if side * side != grid.size:
        raise ValueError(f"Grid is not square. Got {grid.size} grid cells.")
    grid_2d = grid.reshape(side, side)

    player_positions = np.argwhere(grid_2d == 2)
    if player_positions.size == 0:
        player_row, player_col = side // 2, side // 2
    else:
        player_row, player_col = player_positions[0]

    right_grid = grid_2d[:, player_col + 1 :] if player_col + 1 < side else np.empty((side, 0))
    left_grid = grid_2d[:, :player_col] if player_col > 0 else np.empty((side, 0))

    def nearest_distance(target_id):
        target_positions = np.argwhere(grid_2d == target_id)
        if target_positions.size == 0:
            return float(side * 2)
        distances = np.abs(target_positions[:, 0] - player_row) + np.abs(target_positions[:, 1] - player_col)
        return float(distances.min())

    current_sensor_features = np.array(
        [
            sensors[0],  # dir_x
            sensors[1],  # dir_y
            sensors[2],  # vel_x
            sensors[3],  # vel_y
            sensors[4],  # on_floor
            sensors[5],  # on_wall
            sensors[6],  # perc_to_peak
            sensors[7],  # has_powerup
            float(sensors[3] < 0),  # is_falling
            float(sensors[3] > 0),  # is_rising
            float((sensors[4] > 0.5) or (sensors[7] > 0.5)),  # can_jump_now
            float(abs(sensors[6] - 1.0) < 0.1),  # near jump apex
        ],
        dtype=np.float32,
    )

    if prev_observation is None:
        prev_sensors = np.zeros(8, dtype=np.float32)
    else:
        prev_obs = np.asarray(prev_observation, dtype=np.float32)
        prev_sensors = prev_obs[-8:]

    temporal_features = np.array(
        [
            prev_sensors[2],  # prev vel_x
            prev_sensors[3],  # prev vel_y
            sensors[2] - prev_sensors[2],  # delta vel_x
            sensors[3] - prev_sensors[3],  # delta vel_y
            prev_sensors[4],  # prev on_floor
            prev_sensors[5],  # prev on_wall
        ],
        dtype=np.float32,
    )

    grid_features = np.array(
        [
            float(np.sum(right_grid == 3)),  # spikes right
            float(np.sum(left_grid == 3)),  # spikes left
            float(np.sum(right_grid == 1)),  # walls right
            float(np.sum(right_grid == 4)),  # exits right
            nearest_distance(3),  # nearest spike
            nearest_distance(1),  # nearest wall
            nearest_distance(4),  # nearest exit
            float(np.sum(grid_2d[player_row + 1 :, max(0, player_col - 1) : player_col + 2] == 0)),  # safe cells below
        ],
        dtype=np.float32,
    )

    return np.concatenate([current_sensor_features, temporal_features, grid_features], dtype=np.float32)


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


# %%
data_path = "../data/"
all_data = {}

for i, data_file in enumerate(Path(data_path).glob("*.jsonl")):
    observations_by_session = load_observations_by_session(data_file)
    for session_id, all_steps in observations_by_session.items():
        all_data[f"{i}_{session_id}"] = all_steps

features, actions, groups = [], [], []

for session_id, all_steps in all_data.items():
    for step in all_steps:
        features.append(step["state"])
        actions.append(step["action"])
        groups.append(session_id)

plot_action_distribution(actions)

X, y, sample_weights, sample_groups = aggregate_conflicting_examples(features, actions, groups)
plot_action_distribution(y)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=sample_groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
weights_train = sample_weights[train_idx]

clf = tree.DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
)
clf.fit(X_train, y_train, sample_weight=weights_train)

y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(f"Balanced accuracy: {balanced_acc:.3f}")
print(classification_report(y_test, y_pred, digits=3))

plt.figure(figsize=(36, 16))
plot_tree(
    clf,
    filled=True,
    feature_names=[f"feature_{i}" for i in range(X.shape[1])],
    class_names=[f"action_{i}" for i in sorted(np.unique(y))],
    max_depth=4,
)
plt.show()


def agent_brain(observation, prev_observation, step_count):
    # The first step should jump to kickstart movement.
    if step_count == 0:
        return [1]

    action = clf.predict([observation])
    return action

# %%
def main():
    # SETUP THE ENVIRONMENT
    env = setup_environment()

    # GET THE INITIAL STATE OF THE GAME
    obs = env.reset()

    # GET NUMBER OF CONCURRENT AGENTS IN ONE ENVIRONMENT
    nb_agents = len(obs["obs"])
    prev_observations = [None] * nb_agents

    step_count = 0
    while True:
        # TAKE AN ACTION FOR EACH AGENT
        actions = [
            agent_brain(obs["obs"][i], prev_observations[i], step_count)
            for i in range(nb_agents)
        ]

        # FORMAT THE ACTIONS AS A NUMPY ARRAY
        actions = np.array(actions, dtype=np.int64)

        # KEEP THE LAST OBSERVATION FOR TEMPORAL FEATURES
        prev_observations = [obs["obs"][i] for i in range(nb_agents)]

        # EXECUTE THE ACTIONS INSIDE THE ENVIRONMENT
        obs, reward, done, info = env.step(actions)

        # IF ANY OF THE AGENTS FINISHES END THE LOOP
        if any(done):
            break

        step_count += 1

    env.close()


if __name__ == "__main__":
    main()

# %%
