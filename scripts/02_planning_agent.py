import yaml
import numpy as np

from utils import setup_environment
from planning_agent.agent import Agent


def load_plan(plan_path):
    with open(plan_path, 'r') as f:
        plan = yaml.safe_load(f)["plan"]
    return plan


def main():
    env = setup_environment()

    # GET THE INITIAL STATE OF THE GAME
    reward = 0
    done = False
    obs = env.reset()

    # GET NUMBER OF CONCURENT AGENTS IN ONE ENVIRONMENT
    nb_agents = len(obs["obs"])
    
    plan = load_plan('scripts/planning_agent/agent_plan.yml')
    planning_agent = Agent(plan)
    
    while True:
        # TAKE AN ACTION FOR EACH AGENT
        actions = [planning_agent.act(obs["obs"][i], reward, done) for i in range(nb_agents)]
        
        # FORMAT THE ACTIONS AS A NUMPY ARRAY
        actions = np.array(actions, dtype=np.int64)

        # EXECUTE THE ACTIONS INSIDE THE ENVIRONMENT
        obs, reward, done, info = env.step(actions)

        # IF ANY OF THE AGENTS FINISHES OR TIME EXPIRES END THE LOOP
        if any(done):
            break

    env.close()


if __name__ == "__main__":
    main()
