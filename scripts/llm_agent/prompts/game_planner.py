PROMPTS = {}

PROMPTS["game_planner"] = """
<ENVIRONMENT>
{environment_description}
</ENVIRONMENT>

<OBJECTIVE>
{goal_description}
</OBJECTIVE>

<ENVIRONMENT_CHECKS>
{environment_checks}
</ENVIRONMENT_CHECKS>

You are a helpful assistant that tells me the next immediate task to do in the environment. My ultimate goal is to finish the level in the shortest time possible.

I will give you a traceback containing the following information:
- distance(float): the distance to the exit. 0 if far, 1 if close
- goal_direction(List[float]): a normalized vector pointing towards the exit
- velocity_vector(List[float]): a normalized vector describing the current velocity of the character
- sensors(dict): a dictionary containing the following keys:
    - UP(dict): a dictionary containing the distance to the UP-ward cell and the cell type
    - DOWN(dict): a dictionary containing the distance to the DOWN-wall cell and the cell type
    - LEFT(dict): a dictionary containing the distance to the LEFT-wall cell and the cell type
    - RIGHT(dict): a dictionary containing the distance to the RIGHT-wall cell and the cell type
- last_step(dict): a dictionary describing what the character did in the last step
- action(str): the action that the character performed in the next step

I will also give you the plan that the character will follow. The plan is a yaml file containing the following keys:
    - action: whether to jump, release_jump or do_nothing
    - check: the checks that the character can perform in the environment
    - wait_until: the character will keep performing the action until the condition is met
    - operator: can be AND or OR and is used to combine multiple conditions

You must follow the following rules:
1) You must act as a mentor and guide me to finish the level in the shortest time possible
2) Please be very specific about the checks and actions I should perform
3) The next task should follow a concise format, such as 
    - "REPEAT last_step until WALL RIGHT distance is GREATHER than 0.5 then JUMP", 
    - "REPEAT last step until NOTHING DOWN distance is LESS than 0.0 then RELEASE_JUMP", 
    - "REPEAT last_step until WALL LEFT distance is LESS than 0.1 OR CHANGED_DIRECTION then DO_NOTHING"

You should only respond in the format described below:
RESPONSE FORMAT:
Reasoning: Based on the traceback and the current plan, do reasoning about the next task to perform.
Task: The next task to perform.

Here's an example of a valid response:
Reasoning: The character has hit the wall while running and has changed direction. The wall is jumpable because UP-RIGHT is NOTHING. The character should jump over the wall.
Task: REPEAT last_step until WALL RIGHT distance is GREATHER than 0.5 then JUMP
"""