# %%
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import plot_tree

from processing.frame_visualizer import FrameVisualizer
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


def compute_can_jump(sensor_values, session_info):
    """
    The player can jump if:
      - After jumping, it released the jump button and touched the floor
      - After jumping, it released the jump button and has a powerup
    """
    on_floor = bool(sensor_values["on_floor"])
    has_powerup = bool(sensor_values["has_powerup"])
    jump_pressed = int(session_info["prev_action"]) == 1  # jump action id
    
    return float(has_powerup or (on_floor and not jump_pressed))


def compute_wall_sensors(grid, wall_index):
    wall_distance_sensors = []
    
    right_wall_distance = 1
    for i, right_cells in enumerate([grid[3][3], grid[3][4], grid[3][5], grid[3][6]]):
        if right_cells == wall_index:
            right_wall_distance = i * 0.25
            break
    wall_distance_sensors.append(right_wall_distance)

    left_wall_distance = 1
    for i, left_cells in enumerate([grid[3][3], grid[3][2], grid[3][1], grid[3][0]]):
        if left_cells == wall_index:
            left_wall_distance = i * 0.25
            break
    wall_distance_sensors.append(left_wall_distance)
    
    up_wall_distance = 1
    for i, up_cells in enumerate([grid[3][3], grid[2][3], grid[1][3], grid[0][3]]):
        if up_cells == wall_index:
            up_wall_distance = i * 0.25
            break
    wall_distance_sensors.append(up_wall_distance)

    down_wall_distance = 1
    for i, down_cells in enumerate([grid[3][3], grid[4][3], grid[5][3], grid[6][3]]):
        if down_cells == wall_index:
            down_wall_distance = i * 0.25
            break
    wall_distance_sensors.append(down_wall_distance)

    diagonal_up_left_wall_distance = 1
    for i, up_left_cells in enumerate([grid[3][3], grid[2][2], grid[1][1], grid[0][0]]):
        if up_left_cells == wall_index:
            diagonal_up_left_wall_distance = i * 0.25
            break
    wall_distance_sensors.append(diagonal_up_left_wall_distance)

    diagonal_up_right_wall_distance = 1
    for i, up_right_cells in enumerate([grid[3][3], grid[2][4], grid[1][5], grid[0][6]]):
        if up_right_cells == wall_index:
            diagonal_up_right_wall_distance = i * 0.25
            break
    wall_distance_sensors.append(diagonal_up_right_wall_distance)

    diagonal_down_left_wall_distance = 1
    for i, down_left_cells in enumerate([grid[3][3], grid[4][2], grid[5][1], grid[6][0]]):
        if down_left_cells == wall_index:
            diagonal_down_left_wall_distance = i * 0.25
            break
    wall_distance_sensors.append(diagonal_down_left_wall_distance)

    diagonal_down_right_wall_distance = 1
    for i, down_right_cells in enumerate([grid[3][3], grid[4][4], grid[5][5], grid[6][6]]):
        if down_right_cells == wall_index:
            diagonal_down_right_wall_distance = i * 0.25
            break
    wall_distance_sensors.append(diagonal_down_right_wall_distance)

    return wall_distance_sensors


def engineer_features(observation, prev_observation=None, session_info=None):
    visualizer = FrameVisualizer()
    parsed_obs = visualizer.parse_observation(observation)
    grid = parsed_obs["grid"]
    sensor_values = parsed_obs["sensor_values"]
    sensors = list(sensor_values.values())

    # Create features for the player's ability to jump
    can_jump = compute_can_jump(sensor_values, session_info)

    # Create features for walls around the agent
    wall_index = visualizer.get_class_index("Wall")
    wall_distance_sensors = compute_wall_sensors(grid, wall_index)

    return [can_jump] + wall_distance_sensors + sensors


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

actions = []

for session_id, all_steps in all_data.items():
    for step in all_steps:
        actions.append(step["action"])

plot_action_distribution(actions)

engineered_features, groups = [], []
for session_id, all_steps in all_data.items():
    prev_observation = None
    session_info = {"prev_action": 0}
    for step in all_steps:
        engineered_features.append(
            engineer_features(
                observation=step["state"],
                prev_observation=prev_observation,
                session_info=session_info,
            )
        )
        session_info["prev_action"] = step["action"]
        prev_observation = step["state"]
        groups.append(session_id)

X, y, sample_weights, sample_groups = aggregate_conflicting_examples(engineered_features, actions, groups)
plot_action_distribution(y)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=sample_groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
weights_train = sample_weights[train_idx]

clf = tree.DecisionTreeClassifier(
    max_depth=10,
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


def agent_brain(observation, prev_observation, session_info, step_count):
    # The first step should jump to kickstart movement.
    if step_count == 0:
        return [1]

    engineered_observation = engineer_features(
        observation=observation,
        prev_observation=prev_observation,
        session_info=session_info,
    )
    action = clf.predict([engineered_observation])
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
    session_infos = [{"prev_action": 0} for _ in range(nb_agents)]

    step_count = 0
    while True:
        # TAKE AN ACTION FOR EACH AGENT
        actions = [
            agent_brain(
                obs["obs"][i],
                prev_observations[i],
                session_infos[i],
                step_count,
            )
            for i in range(nb_agents)
        ]

        # FORMAT THE ACTIONS AS A NUMPY ARRAY
        actions = np.array(actions, dtype=np.int64)

        # UPDATE THE PREVIOUS OBSERVATION FOR EACH AGENT
        prev_observations = [obs["obs"][i] for i in range(nb_agents)]

        # UPDATE THE SESSION INFO FOR EACH AGENT
        for i, action in enumerate(actions):
            session_infos[i]["prev_action"] = action[0]
        
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
