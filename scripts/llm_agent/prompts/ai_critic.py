PROMPTS = {}

PROMPTS["game_critic"] = """
Your role is to plan the next series of actions and checks for an agent navigating a 2D environment. Analyze the following status_report and plan - and decide what next check or action should the agent perform. 

{environment_checks}

your answer should be 2 bullet points:
* what the agent should check for 
* what action the agent should take once the check is satisfied 

Here are the status reports from the agent:
{status_report}

**Important:**
You need to release the jump button to fall. You need to release the jump button to jump again.
You can hold the jump button to make the character jump higher. Holding the jump too long will make the character fall.
You must satisfy the check before the action is performed. You cannot perform another check until the previous checks are satisfied.
There can be multiple compound checks but only one action.

Here is the current plan:
{current_plan}
"""
