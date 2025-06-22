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

You are a helpful assistant that updates an existing plan for a character to navigate through the level based on the CRITIQUE.

I will give you a CRITIQUE that identifies which step in the plan is incorrect and what it should be corrected to, along with the current PLAN.
    - Reasoning: analysis of what's wrong with the current plan
    - Incorrect Step: which specific step in the plan needs to be updated
    - Correction: what the incorrect step should be changed to

You must follow the following rules:
1) You must update the existing plan **based only on the incorrect step and correction provided in the critique**, with no additional assumptions.
2) You must identify the specified incorrect step in the current plan and replace it with the correction provided.
3) Each task follows an action-condition-action structure: start with an action, wait for condition(s), then perform end action.
4) You will answer only with the updated plan and nothing else.

You should only respond in the format described below:
RESPONSE FORMAT:
plan: the updated plan formatted as a yaml file that can only contain the following keys:
    - action: whether to jump, release_jump or do_nothing
    - wait_until: the character will keep performing the action until the condition is met
    - condition: the checks that the character can perform in the environment
    - args: the arguments for the check
    - operator: can be AND or OR and is used to combine multiple conditions
    - conditions: a list of conditions that must be met - only used if the condition is a compound condition

Here are examples of valid responses following the action-condition-action structure:

Example 1 - Single condition:
plan:
  - action: jump
  - wait_until:
      condition: is_object_near
      args: [WALL, DOWN, 0.8]
  - action: release_jump

Example 2 - Multiple conditions with operator:
plan:
  - action: release_jump
  - wait_until:
      operator: AND
      conditions:
        - condition: changed_direction
        - condition: is_object_far
          args: [WALL, DOWN-LEFT, 0.7]
  - action: jump

CRITIQUE:
{critique}

PLAN:
{plan}
"""