"""Course-era if/else baseline for the legacy 58 observation."""

from __future__ import annotations

import numpy as np

from plaiground.legacy.processing.frame_visualizer import FrameVisualizer
from plaiground.legacy.utils import setup_environment


def agent_brain(observation, step_count: int) -> list[int]:
    parsed = FrameVisualizer().parse_observation(observation)
    grid = parsed["grid"]
    if grid[3, 6] != 0 or step_count == 0:
        return [1]
    return [0]


def main() -> None:
    env = setup_environment()
    obs = env.reset()
    nb_agents = len(obs["obs"])
    step_count = 0
    try:
        while True:
            actions = np.array(
                [agent_brain(obs["obs"][i], step_count) for i in range(nb_agents)],
                dtype=np.int64,
            )
            obs, _reward, done, _info = env.step(actions)
            if any(done):
                break
            step_count += 1
    finally:
        env.close()
