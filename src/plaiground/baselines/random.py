"""Random-action baseline for the legacy wrapper path."""

from __future__ import annotations

import numpy as np

from plaiground.legacy.utils import setup_environment


def agent_brain(observation) -> list[int]:
    return [int(np.random.randint(0, 2))]


def main() -> None:
    env = setup_environment()
    obs = env.reset()
    nb_agents = len(obs["obs"])
    try:
        while True:
            actions = np.array(
                [agent_brain(obs["obs"][i]) for i in range(nb_agents)],
                dtype=np.int64,
            )
            obs, _reward, done, _info = env.step(actions)
            if any(done):
                break
    finally:
        env.close()
