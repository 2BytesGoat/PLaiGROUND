import numpy as np
from utils import setup_environment


def if_else_agent(observation):
    action = 0

    distance, goal_x, goal_y, velocity_x, velocity_y = observation[:5]
    signal_shift = 5 # the signal info starts at index 5
    objec_type_shift = signal_shift + 32 # the object type info starts at index 32

    sensors = observation[signal_shift : objec_type_shift]
    forward_sensor = sensors[8]

    if forward_sensor > 0.78:
        action = 1

    return [action]


def main():
    # SETUP THE ENVIRONMENT
    env = setup_environment()

    # GET THE INITIAL STATE OF THE GAME
    obs = env.reset()

    # GET NUMBER OF CONCURENT AGENTS IN ONE ENVIRONMENT
    nb_agents = len(obs["obs"])
    
    while True:
        # TAKE AN ACTION FOR EACH AGENT
        actions = [if_else_agent(obs["obs"][i]) for i in range(nb_agents)]

        # FORMAT THE ACTIONS AS A NUMPY ARRAY
        actions = np.array(actions, dtype=np.int64)

        # EXECUTE THE ACTIONS INSIDE THE ENVIRONMENT
        obs, reward, done, info = env.step(actions)

        # IF ANY OF THE AGENTS FINISHES END THE LOOP
        if any(done):
            break

    env.close()


if __name__ == "__main__":
    main()

