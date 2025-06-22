from .game_description import PROMPTS as GAME_DESCRIPTION_PROMPTS
from .ai_planner import PROMPTS as GAME_PLANNER_PROMPTS
from .ai_critic import PROMPTS as GAME_CRITIC_PROMPTS
from .ai_observer import PROMPTS as GAME_OBSERVER_PROMPTS


GAME_CRITIC_PROMPTS["game_critic"] = GAME_CRITIC_PROMPTS["game_critic"].replace("{environment_checks}", GAME_DESCRIPTION_PROMPTS["environment_checks"])

GAME_PLANNER_PROMPTS["game_planner"] = GAME_PLANNER_PROMPTS["game_planner"].replace("{environment_checks}", GAME_DESCRIPTION_PROMPTS["environment_checks"])

PROMPTS = {
    **GAME_OBSERVER_PROMPTS,
    **GAME_CRITIC_PROMPTS,
    **GAME_PLANNER_PROMPTS
}