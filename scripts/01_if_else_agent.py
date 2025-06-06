import argparse

import numpy as np
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "--env_path",
    default=None,
    type=str,
    help="The Godot binary to use, do not include for in editor training",
)
parser.add_argument(
    "--viz",
    action="store_true",
    help="If set, the simulation will be displayed in a window during training. Otherwise "
    "training will run without rendering the simulation. This setting does not apply to in-editor training.",
    default=True,
)
parser.add_argument("--seed", type=int, default=42, help="seed of the experiment")
parser.add_argument(
    "--n_parallel",
    default=1,
    type=int,
    help="How many instances of the environment executable to "
    "launch - requires --env_path to be set if > 1.",
)
parser.add_argument(
    "--speedup", default=1, type=int, help="Whether to speed up the physics in the env"
)


def if_else_agent(ovservation):
    # THE BIGGER THE VALUE, THE CLOSER YOU ARE TO A WALL
    if ovservation["sensors"][8] + ovservation["sensors"][9] > 1.2:
        return 1
    return 0


def main():
    args, extras = parser.parse_known_args()

    env = StableBaselinesGodotEnv(
        env_path=args.env_path,
        show_window=args.viz,
        seed=args.seed,
        n_parallel=args.n_parallel,
        speedup=args.speedup,
    )

    # GET THE INITIAL STATE OF THE GAME
    obs = env.reset()

    # GET NUMBER OF CONCURENT AGENTS IN ONE ENVIRONMENT
    nb_agents = len(obs["obs"])

    while True:
        # CALCULATE BEST ACTION TO TAKE
        action = if_else_agent(obs["obs"][0])

        # TELL ALL AGENTS TO TAKE THE SAME ACTION
        actions = [[action] for _ in range(nb_agents)]

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
