# %% 
"""
TODO
- make this into a jupyter notebook
- explain train test splits
- explain decision tree parameters (gini)
- view accuracy on test set
- visualize the decision tree
- test AI on environment
- show overfitting by showing how big the tree is and that decreasing size keeps accuracy
- show how after you prune the tree it become more interpretable but biased
- show that's important to how you build test set (histogram, balanced labels)
"""

# %%
from pathlib import Path

from sklearn import tree
from sklearn.model_selection import train_test_split

from utils import load_observations_by_session
# %%
data_path = "../data/"
all_data = {}

for i, file in enumerate(Path(data_path).glob("*.jsonl")):
    observations_by_session = load_observations_by_session(file)
    for session_id, all_steps in observations_by_session.items():
        all_data[f"{i}_{session_id}"] = all_steps
# %%
X, y = [], []
for session_id, all_steps in all_data.items():
    for step in all_steps:
        X.append(step["state"])
        y.append(step["action"])
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=10)
clf = clf.fit(X_train, y_train)
# %%
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
# %%
# Visualize the decision tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(100,40))
plot_tree(clf, filled=True, feature_names=[f"feature_{i}" for i in range(len(X[0]))], class_names=[f"action_{i}" for i in range(len(set(y)))])
plt.show()
# %%
import numpy as np

from utils import setup_environment


def agent_brain(observation, step_count):
    # Important: You need to press jump for the firest frame to start walking.
    if step_count == 0:
        return [1] # press jumpt to start walking
    
    action = clf.predict([observation])
    return action


def main():
    # SETUP THE ENVIRONMENT
    env = setup_environment()

    # GET THE INITIAL STATE OF THE GAME
    obs = env.reset()

    # GET NUMBER OF CONCURENT AGENTS IN ONE ENVIRONMENT
    nb_agents = len(obs["obs"])
    
    step_count = 0
    while True:
        # TAKE AN ACTION FOR EACH AGENT
        actions = [agent_brain(obs["obs"][i], step_count) for i in range(nb_agents)]

        # FORMAT THE ACTIONS AS A NUMPY ARRAY
        actions = np.array(actions, dtype=np.int64)

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
