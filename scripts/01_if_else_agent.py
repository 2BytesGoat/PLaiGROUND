import os
import argparse

import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "--env_path",
    default=None,
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
    default=10,
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


def if_else_agent(observation, step_number):
    action = 0
    
    # JUMP FOR FIRST 3 STEPS TO START THE GAME
    if step_number < 3:
        action = step_number % 2

    # JUMP IF YOU ARE CLOSER THAN 1.2 UNITS TO A WALL
    else:
        # split the observation into the different components
        distance, goal_x, goal_y, velocity_x, velocity_y = observation[:5]
        sensors = observation[5:37]
        sensors_collision = observation[37:]

        forward_sensor = sensors[8]

        if forward_sensor > 0.8:
            action = 1

    return np.array([action])


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
        level=args.level
    )

    # GET THE INITIAL STATE OF THE GAME
    obs = env.reset()

    # GET NUMBER OF CONCURENT AGENTS IN ONE ENVIRONMENT
    nb_agents = len(obs["obs"])
    
    step_number = 0
    while True:
        # TAKE AN ACTION FOR EACH AGENT
        actions = [if_else_agent(obs["obs"][0], step_number) for _ in range(nb_agents)]
        
        # FORMAT THE ACTIONS AS A NUMPY ARRAY
        actions = np.array(actions, dtype=np.int64)

        # EXECUTE THE ACTIONS INSIDE THE ENVIRONMENT
        obs, reward, done, info = env.step(actions)

        # IF ANY OF THE AGENTS FINISHES OR TIME EXPIRES END THE LOOP
        if any(done):
            break

        step_number += 1

    env.close()


if __name__ == "__main__":
    main()
