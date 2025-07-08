import os
import dotenv

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


def setup_environment(nb_agents=None, level=None):
    dotenv.load_dotenv(dotenv_path="scripts/.config", override=True)
    
    env_path = None
    env_dir = os.getenv("ENV_PATH")
    env_name = os.getenv("ENV_NAME")

    if env_dir:
        env_path = os.path.join(env_dir, env_name)
    
    seed = os.getenv("SEED", 42)
    speedup = os.getenv("SPEEDUP", 1)
    nb_agents = nb_agents if nb_agents else os.getenv("NB_AGENTS", 1)
    level = level if level else os.getenv("LEVEL", "1-1")
    action_repeat = os.getenv("ACTION_REPEAT", 5)

    env = StableBaselinesGodotEnv(
        env_path=env_path,
        show_window=True, # can't get 2d observation without rendering
        seed=seed,
        speedup=speedup,
        nb_agents=nb_agents,
        level=level,
        action_repeat=action_repeat,
    )

    return env
