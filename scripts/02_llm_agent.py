import os
import argparse
import yaml

import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

from llm_agent.agent import Agent

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "--env_path",
    default="/Users/gianistatie/Documents/personal/DragonJump/builds/osx",
    type=str,
    help="The path to the Godot game directory (get this by typing 'path' in the game's developer console)",
)
parser.add_argument(
    "--env_name",
    default="DragonJump",
    type=str,
    help="The Godot binary to use, do not include for in editor training",
)
parser.add_argument("--seed", type=int, default=42, help="seed of the experiment")
parser.add_argument(
    "--nb_agents",
    default=1,
    type=int,
    help="How many agents to launch in the environment. "
    "Requires --nb_agents to be set if > 1.",
)
parser.add_argument(
    "--speedup", default=1, type=int, help="Whether to speed up the physics in the env"
)
parser.add_argument(
    "--level", default="1-1", type=str, help="The level to play"
)


def main():
    args, extras = parser.parse_known_args()

    env_path = None
    if args.env_path:
        env_path = os.path.join(args.env_path, args.env_name)

    env = StableBaselinesGodotEnv(
        env_path=env_path,
        show_window=True, # can't get 2d observation without rendering
        seed=args.seed,
        speedup=args.speedup,
        nb_agents=args.nb_agents,
        level=args.level,
        action_repeat=3, # the number of times an action is repeated
        max_steps=1000,
    )

    # GET THE INITIAL STATE OF THE GAME
    obs = env.reset()
    reward = 0
    done = False

    # GET NUMBER OF CONCURENT AGENTS IN ONE ENVIRONMENT
    nb_agents = len(obs["obs"])
    nb_steps = 255

    with open('scripts/llm_agent/agent_plan.yml', 'r') as f:
        plan = yaml.safe_load(f)["plan"]
    llm_agent = Agent(plan)
    
    for i in range(nb_steps):
        # TAKE AN ACTION FOR EACH AGENT
        actions = [llm_agent.act(obs["obs"][i], reward, done) for i in range(nb_agents)]
        
        # FORMAT THE ACTIONS AS A NUMPY ARRAY
        actions = np.array(actions, dtype=np.int64)

        # EXECUTE THE ACTIONS INSIDE THE ENVIRONMENT
        obs, reward, done, info = env.step(actions)

        # IF ANY OF THE AGENTS FINISHES OR TIME EXPIRES END THE LOOP
        if any(done):
            break
    
    if not done:
        report = llm_agent.generate_report()
        print(report)

    env.close()


if __name__ == "__main__":
    main()
