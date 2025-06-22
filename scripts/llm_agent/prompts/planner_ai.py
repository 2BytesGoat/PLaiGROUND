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

You are a helpful assistant that helps me improve my plan.

I will give you a task and the reasoning for why I think it's a good idea to do it.
    - Reasoning: what I think I'm doing wrong
    - Task: how the plan should be improved

I will also give you the plan that the character currently follows. The plan is a yaml file containing the following keys:
    - action: whether to jump, release_jump or do_nothing
    - check: the checks that the character can perform in the environment
    - wait_until: the character will keep performing the action until the condition is met
    - operator: can be AND or OR and is used to combine multiple conditions

You must follow the following rules:
1) You must act as a mentor and guide me to finish the level in the shortest time possible
2) You will give me the updated plan in the same format as the original plan.
3) You will not change the steps prior to the one that I'm criticizing.
4) You will answer only with the updated plan and nothing else.

You should only respond in the format described below:
RESPONSE FORMAT:
plan: the updated plan

Here's an example of a valid response:
plan:
    - wait_until:
        condition: is_object_near
        args: [WALL, RIGHT, 0.8]
    - action: jump

{critique}

{plan}
"""