PROMPTS = {}

PROMPTS["environment_description"] = """
This is a 2D precision platformer controlled with a single input.

Core Controls
Single Input: The game is controlled entirely by pressing or holding the Spacebar.

Jumping Mechanics
Jump Initiation:
Tapping Space causes the character to jump.

Jump Height:
Holding Space increases the jump height, up to a limit.

Mid-Air Cancel:
Holding Space for too long while airborne causes the character to stop ascending and begin falling.

Jump Lockout:
The character cannot jump again until the Spacebar is released and pressed anew.

Directional Movement
No Manual Movement:
The character cannot move left or right directly. Horizontal direction is changed only through interactions with walls.
"""

PROMPTS["goal_description"] = """
Navigate through the level in order to touch the exit.
"""

PROMPTS["environment_checks"] = """
The character can perform the following checks:
- is_object_near(object_type: str [see objects above], direction: str [see directions above], distance: float [0.0 - 1.0]) -> bool - checks if there is any object closer than the threshold
- is_object_far(object_type: str [see objects above], direction: str [see directions above], distance: float [0.0 - 1.0]) -> bool - checks if there is any object further than the threshold
- changed_direction() -> bool - checks if the character changed direction

The objects are:
- NOTHING
- WALL
- SPIKES
- DISSOLVE-BLOCK
- BOUNCE-PAD
- DOUBLE-JUMP

The directions are:
- UP
- UP-RIGHT
- RIGHT
- DOWN-RIGHT
- DOWN
- DOWN-LEFT
- LEFT
- UP-LEFT
"""