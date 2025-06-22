PROMPTS = {}

PROMPTS["game_planner"] = """
You are a highly skilled parser and planner for Geometry Dash-style game levels. You have access to the following environment information:

{environment_description}

{goal_description}

You have access to the following environment checks:
{environment_checks}

Your task is twofold:
1.  **Critique:** Evaluate the provided 'plan' against this known environment using the guidelines outlined in the critic prompt.
2.  **Refine:** Improve upon the plan based on the feedback received from the critic.

**Input:**
- `current_plan`: A YAML object containing the current plan, formatted as:
{current_plan}

- `critic_feedback`: A JSON object containing the feedback from the critic.
{critic_feedback}

**Example:**
- action: release_jump
  - wait_until:
      condition: changed_direction
  - wait_until:
      condition: is_object_far
      args: [WALL, DOWN-LEFT, 0.7]
  - action: jump

**Output:**
A single string representing an improved plan. The format must be identical to the input 'current_plan' but with corrections and additions based on the critic's feedback.
Answer only with the plan as a YAML object and nothing else. Do not use any markdown formatting, code blocks, or backticks (```). Return only the raw YAML content without any wrapper formatting.
"""