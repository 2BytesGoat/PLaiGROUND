import time
import numpy as np
from utils import setup_environment
from processing.frame_visualizer import FrameVisualizer


def agent_brain(observation, step_count):
    # TODO: define this object once, outside the function
    visualizer = FrameVisualizer()

    parsed = visualizer.parse_observation(observation)
    grid = parsed["grid"]

    # Example heuristic: jump whenyou see a wall on your right.
    # First is the row, second is the column.
    # Important: You need to press jump for the firest frame to start walking.
    if grid[3, 6] != 0 or step_count == 0:
        return [1] # jump

    return [0] # do nothing


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

