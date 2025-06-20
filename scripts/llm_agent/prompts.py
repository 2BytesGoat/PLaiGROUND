PROMPTS = {}

PROMPTS["update_plan"] = """
You are an agent controlling a CHARACTER in a 2D platformer game using symbolic planning.

The character can perform the following actions:
* jump
* do_nothing
* release_jump

The environment is a 2D grid with the following objects:
* NOTHING       - the cell is empty
* WALL          - the cell is a wall
* SPIKES        - the cell is a spike that kills the character
* DISOLVE-BLOCK - the cell is a dissolve block
* BOUNCE-PAD    - the cell is a bounce pad that allows the character to jump higher
* DOUBLE-JUMP   - the cell is a double jump pad that allows the character to jump twice

The directions are:
* UP - above the character
* UP-RIGHT - above and to the right of the character    
* RIGHT - to the right of the character
* DOWN-RIGHT - below and to the right of the character
* DOWN - below the character
* DOWN-LEFT - below and to the left of the character
* LEFT - to the left of the character
* UP-LEFT - above and to the left of the character

You have access to the following tools:
* is_object_near(object_type: str [see objects above], direction: str [see directions above], distance: float [0.0 - 1.0]) -> bool - checks if there is any object closer than the threshold
* is_object_far(object_type: str [see objects above], direction: str [see directions above], distance: float [0.0 - 1.0]) -> bool - checks if there is any object further than the threshold
* is_on_wall() -> bool - checks if the character is on a wall
* changed_direction() -> bool - checks if the character changed direction

To beat the level the agents needs to:
- Wait until you're near the first wall.
- Jump to start climbing the wall.
- Hold the jump until you're above or on the first platform.
- When you're near the edge of the platform jump to the next platform.
- Hold the jump until you land on the second platform.
- When you're near the edge again and there's no wall down-right, jump to the third platform.
- Hold the jump until you're on the third platform.
- Wait until the character turns around near the platform edge.
- Once there's no wall down-left, jump to reach the final goal.

Produce your plan in the following YAML format:

plan:
  - wait_until:
      condition: is_object_near
      args: [WALL, RIGHT, 0.8]
  - action: jump
"""

PROMPTS["next_steps"] = """
This is a 2D precision platformer controlled with a single input.

Core Mechanics:
Jumping: Pressing space makes the character jump. You need to release the jump button in order to be able to jump again.

Wall Interaction:
When touching a wall, the character turns to face away from it.
Once the character touches the wall, it slides slowly downward.
Pressing space while sliding on a wall triggers a wall jump in the opposite direction.

Auto-Fall: 
If space is held too long mid-air, the character will stop jumping and fall automatically.
No directional control: The player cannot move left/right directly; direction changes only by wall contact.

Objective:
Navigate through levels using well-timed jumps and wall interactions to reach the goal. Precision and timing are key.

Your role is to analyze the state of the game and come up with a series of checks and actions to perform in order for an AI agent to get to the end.

Your answer should be in the following format, with nothing else:

plan:
- do_nothing 
- wait_until there is a wall close to your right
- jump

plan:
- jump
- wait_until there is a wall far from your down
- do_nothing

{trace}
"""