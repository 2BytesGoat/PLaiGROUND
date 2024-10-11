import argparse

import cv2
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


def main():
    args, extras = parser.parse_known_args()

    # INITIALIZE THE ENVIRONMENT USED TO TRAIN THE AI
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
        # TAKE A RANDOM ACTION FOR EACH AGENT
        actions = [env.action_space.sample() for _ in range(nb_agents)]

        # FORMAT THE ACTIONS AS A NUMPY ARRAY
        actions = np.array(actions, dtype=np.int64)

        # EXECUTE THE ACTIONS INSIDE THE ENVIRONMENT
        obs, reward, done, info = env.step(actions)

        width = info[0]["frame"]["width"]
        height = info[0]["frame"]["height"]

        for i, image_as_str in enumerate(info):
            image = np.fromstring(
                image_as_str["frame"]["data"][1:-1], dtype=int, sep=", "
            ).reshape(height, width, 4)
            cv2.imwrite(f"test_{i}.png", image)

        # IF ANY OF THE AGENTS FINISHES OR TIME EXPIRES END THE LOOP
        if any(done):
            break

    env.close()


if __name__ == "__main__":
    main()
