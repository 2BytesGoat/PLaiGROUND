PROMPTS = {}

PROMPTS["game_critic"] = """
<ENVIRONMENT>
{environment_description}
</ENVIRONMENT>

<OBJECTIVE>
{goal_description}
</OBJECTIVE>

<ENVIRONMENT_CHECKS>
{environment_checks}
</ENVIRONMENT_CHECKS>

You are a helpful assistant that analyses the traceback and the current plan and criticizes the plan by pointing out what specific step is incorrect and needs to be updated.

You must follow the following rules:
1) You must act as a mentor and guide me to finish the level in the shortest time possible
2) Please be very specific about which step in the plan is incorrect and what it should be changed to
3) Provide detailed reasoning based on the velocity, sensor readings, and goal direction
4) Focus on identifying the specific step number or action that needs correction
5) Each task should contain three components: a start action, a condition or series of conditions, and an end action
6) CRITICAL: Your correction must be fundamentally different from the incorrect step - do not just rephrase the same logic
7) Analyze if conditions are already met, if actions don't match the current state, or if the sequence doesn't make sense
8) The next task should follow a concise format, such as:  
   - "Action: jump, wait until is_object_near WALL DOWN 0.8, then action: release_jump"  
   - "Action: release_jump, wait until changed_direction AND is_object_far WALL DOWN-LEFT 0.7, then action: jump"  
   - "Action: jump, wait until is_object_near WALL RIGHT 0.8 OR is_object_near WALL DOWN 0.5, then action: release_jump"  
   - "Action: release_jump, wait until is_object_far WALL DOWN-RIGHT 0.7 AND changed_direction, then action: jump"  
   - "Reasoning: ... Task: ..."

You should only respond in the following strict format:

RESPONSE FORMAT:
Reasoning: <your reasoning here>
Incorrect Step: <specify which step number or action in the plan is wrong>
Correction: <the specific task that should replace the incorrect step>

Here's an example of a valid response:
Reasoning: 
* The character is moving left with velocity [-1.0, 0.0] and has walls detected in UP-RIGHT (0.938) and RIGHT (0.956) directions
* The current plan waits for walls that are already detected - the conditions UP-RIGHT < 0.8 and RIGHT < 0.8 are both FALSE since distances are > 0.8
* The character is stuck waiting for conditions that will never be true
* Instead, the character should wait for the wall on the LEFT to get closer (currently 0.003) indicating impact, then change direction

Incorrect Step: Step 2 - "wait_until: conditions: is_object_near WALL UP-RIGHT 0.8 OR is_object_near WALL RIGHT 0.8"
Correction: Action: release_jump, wait until is_object_near WALL LEFT 0.1, then action: jump

{report}

CURRENT PLAN:
{plan}

"""