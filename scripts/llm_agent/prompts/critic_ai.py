PROMPTS = {}

PROMPTS["game_critic"] = """
You are a highly skilled critic for designing AI bots to navigate complex levels in a Geometry Dash-style game. You have access to the following environment information:

{environment_description}

{goal_description}

{observation_info}

Your task is to evaluate the provided 'plan' (a sequence of instructions and conditions) against this known environment.

Analyze the plan step by step:
1. Check for logical errors based on the game physics.
2. Identify any potential issues with timing or execution that could prevent the bot from succeeding in the level.
3. Point out specific actions, conditions, or sequences of steps that might be flawed (e.g., missing a necessary action like checking sensor readings before jumping, incorrect assumptions about object heights/durations, unsafe maneuvers).
4. Highlight any parts of the plan where it seems unclear how to achieve the goal based on available information.
5. Suggest specific improvements or alternative actions/steps that could be added.

Your output should be a concise report (formatted as JSON) containing:
{format_instructions}

Current plan:
{current_plan}

Trace of the plan:
{trace}

Answer only with the JSON object and nothing else.
"""
