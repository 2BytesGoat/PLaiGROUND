import os
from dotenv import load_dotenv

from wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


def setup_environment(nb_agents=None, level=None):
    config_path = os.path.join(os.getcwd(), ".config")
    load_dotenv(config_path, override=True)
    
    env_path = None
    env_dir = os.getenv("ENV_PATH", "../environments")
    env_name = os.getenv("ENV_NAME", "DragonJump")

    if env_dir:
        env_path = os.path.join(env_dir, env_name)
    
    seed = os.getenv("SEED", 42)
    speedup = os.getenv("SPEEDUP", 1)
    nb_agents = nb_agents if nb_agents else os.getenv("NB_AGENTS", 1)
    level = level if level else os.getenv("LEVEL", "1-1")
    action_repeat = os.getenv("ACTION_REPEAT", 5)
    show_window = os.getenv("SHOW_WINDOW", "False").lower() == "true"

    env = StableBaselinesGodotEnv(
        env_path=env_path,
        show_window=show_window,
        seed=seed,
        speedup=speedup,
        nb_agents=nb_agents,
        level=level,
        action_repeat=action_repeat,
    )

    return env
