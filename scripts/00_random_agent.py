import numpy as np
from utils import setup_environment


def random_agent(observation):
    action = np.random.randint(0, 2)
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
        actions = [random_agent(obs["obs"][i]) for i in range(nb_agents)]

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
