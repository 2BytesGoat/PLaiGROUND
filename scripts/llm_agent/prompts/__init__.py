from .game_description import PROMPTS as GAME_DESCRIPTION_PROMPTS
from .planner_ai import PROMPTS as GAME_PLANNER_PROMPTS
from .critic_ai import PROMPTS as GAME_CRITIC_PROMPTS


GAME_PLANNER_PROMPTS["game_planner"] = GAME_PLANNER_PROMPTS["game_planner"].replace("{environment_description}", GAME_DESCRIPTION_PROMPTS["environment_description"])
GAME_PLANNER_PROMPTS["game_planner"] = GAME_PLANNER_PROMPTS["game_planner"].replace("{goal_description}", GAME_DESCRIPTION_PROMPTS["goal_description"])
GAME_PLANNER_PROMPTS["game_planner"] = GAME_PLANNER_PROMPTS["game_planner"].replace("{environment_checks}", GAME_DESCRIPTION_PROMPTS["environment_checks"])

GAME_CRITIC_PROMPTS["game_critic"] = GAME_CRITIC_PROMPTS["game_critic"].replace("{environment_description}", GAME_DESCRIPTION_PROMPTS["environment_description"])
GAME_CRITIC_PROMPTS["game_critic"] = GAME_CRITIC_PROMPTS["game_critic"].replace("{goal_description}", GAME_DESCRIPTION_PROMPTS["goal_description"]) 
GAME_CRITIC_PROMPTS["game_critic"] = GAME_CRITIC_PROMPTS["game_critic"].replace("{environment_checks}", GAME_DESCRIPTION_PROMPTS["environment_checks"])


PROMPTS = {
    **GAME_PLANNER_PROMPTS,
    **GAME_CRITIC_PROMPTS,
}