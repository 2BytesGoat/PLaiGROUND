import os
import json
from collections import defaultdict

from dotenv import load_dotenv

from plaiground.envs.stable_baselines_wrapper import StableBaselinesGodotEnv


def setup_environment(
    nb_agents=None,
    level=None,
    action_repeat=None,
    speedup=None,
    show_window=None,
    port=None,
    n_parallel=1,
    obs_encoding=None,
    reassert_held_jump=False,
):
    repo_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    config_path = os.path.join(repo_directory, "src", ".config")
    load_dotenv(config_path, override=True)
    print(f"Loaded environment variables from {config_path}")
    
    env_path = None
    env_dir = os.getenv("ENV_PATH")
    env_name = os.getenv("ENV_NAME")

    if env_dir:
        if os.path.isabs(env_dir):
            resolved_env_dir = env_dir
        else:
            config_directory = os.path.dirname(config_path)
            repo_directory = os.path.dirname(config_directory)
            search_dirs = [
                os.getcwd(),
                config_directory,
                repo_directory,
            ]
            resolved_env_dir = env_dir
            for base_dir in search_dirs:
                candidate = os.path.abspath(os.path.join(base_dir, env_dir))
                if os.path.exists(candidate):
                    resolved_env_dir = candidate
                    break

        env_path = os.path.join(resolved_env_dir, env_name)
    
    seed = os.getenv("SEED", 42)
    speedup = speedup if speedup is not None else os.getenv("SPEEDUP", 1)
    nb_agents = nb_agents if nb_agents else os.getenv("NB_AGENTS", 1)
    level = level if level else os.getenv("LEVEL", "1-1")
    action_repeat = (
        action_repeat
        if action_repeat is not None
        else os.getenv("ACTION_REPEAT", 5)
    )
    show_window = (
        show_window
        if show_window is not None
        else os.getenv("SHOW_WINDOW", "False").lower() == "true"
    )

    env_kwargs = dict(
        env_path=env_path,
        show_window=show_window,
        seed=seed,
        speedup=speedup,
        nb_agents=nb_agents,
        level=level,
        action_repeat=action_repeat,
        n_parallel=n_parallel,
    )
    if port is not None:
        env_kwargs["port"] = port
    if obs_encoding is not None:
        env_kwargs["obs_encoding"] = obs_encoding
    if reassert_held_jump:
        env_kwargs["reassert_held_jump"] = "true"

    env = StableBaselinesGodotEnv(**env_kwargs)

    return env

def load_observations_by_session(data_path: str) -> dict[int, list[dict]]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    frames_by_session = defaultdict(list)
    for frame in data:
        frames_by_session[int(frame["session"])].append(frame)
    return dict(frames_by_session)
