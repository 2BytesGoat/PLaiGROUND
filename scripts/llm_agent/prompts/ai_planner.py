PROMPTS = {}

PROMPTS["game_planner"] = """
Your role is to create a check and action to the plan based on the critic's feedback. You must checks and actions from the list bellow. You cannot add new checks or actions. You must follow the critic instructions to the letter, without adding any other chekcs or actions which were not mentioned.

{environment_checks}

**Input:**
- `critic_feedback`: A list of checks and actions to add to the plan.
{critic_feedback}

**Here's an example of a simple plan:**
plan:
  - wait_until:
      condition: is_object_near
      args: [WALL, RIGHT, 0.5]
  - action: jump

**Here's an example of how you can combine multiple checks:**
plan:
  - action: do_nothing
  - wait_until:
      operator: OR
      conditions:
        - operator: OR
          conditions:
            - condition: is_object_near
              args: [WALL, RIGHT, 0.5]
            - condition: is_object_far
              args: [WALL, UP, 0.2]
        - condition: is_sliding_on_wall
  - action: jump

**Output:**
A single string representing the checks and action. Do not add any other checks that were not mentioned in the critic's feedback.
Answer only with the plan as a YAML object and nothing else. Do not use any markdown formatting, code blocks, or backticks (```). Return only the raw YAML content without any wrapper formatting.
"""