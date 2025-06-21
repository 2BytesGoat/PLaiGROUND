from .game_description import PROMPTS as GAME_DESCRIPTION_PROMPTS
from .game_planner import PROMPTS as GAME_PLANNER_PROMPTS


GAME_PLANNER_PROMPTS["game_planner"] = GAME_PLANNER_PROMPTS["game_planner"].replace("{environment_description}", GAME_DESCRIPTION_PROMPTS["environment_description"])
GAME_PLANNER_PROMPTS["game_planner"] = GAME_PLANNER_PROMPTS["game_planner"].replace("{goal_description}", GAME_DESCRIPTION_PROMPTS["goal_description"])
GAME_PLANNER_PROMPTS["game_planner"] = GAME_PLANNER_PROMPTS["game_planner"].replace("{environment_checks}", GAME_DESCRIPTION_PROMPTS["environment_checks"])


PROMPTS = {
    **GAME_PLANNER_PROMPTS,
}