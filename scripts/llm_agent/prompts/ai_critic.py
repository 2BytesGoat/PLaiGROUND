PROMPTS = {}

PROMPTS["game_critic"] = """
Your role is to plan the next series of actions and checks for an agent navigating a 2D environment. Analyze the following status_report and decide what next check or action should the agent perform.
You will also be given the current plan that the agent is following. Only the last check + action are relevant for you. 

your answer should be 2 bullet points:
* what the agent should check for 
* what action the agent should take once the check is satisfied 

Here are some checks and actions that the agent can perform:
{environment_checks}

Here are the status reports from the agent:
{status_report}

**Important:**
If the agent is in the air you will need to check whether it can land safely and release the jump button to fall.
The agent needs to release the jump and touch the ground in order to be able to jump again.
The check you suggest must be satisfied before the action is performed. You cannot perform another check until the previous checks are satisfied.
There can be multiple compound checks but only one action.

Here is the current plan:
{current_plan}
"""
