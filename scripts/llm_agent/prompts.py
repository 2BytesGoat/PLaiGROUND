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