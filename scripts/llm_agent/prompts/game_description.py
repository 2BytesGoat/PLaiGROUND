PROMPTS = {}

PROMPTS["environment_description"] = """
This is a 2D precision platformer controlled with a single input.

Core Mechanics:
Jumping: Pressing space makes the character jump. Holding space makes the character jump higher. If space is held too long mid-air, the character will stop jumping and fall automatically. You need to release the jump button in order to be able to jump again.
No directional control: The player cannot move left/right directly; direction changes only by wall contact.

Running into a Wall:
When touching a wall, the character turns to face away from it.

Jumping on a Wall:
While jumping, if a character touches a wall,it slides slowly downward.
Pressing space while sliding on a wall triggers a wall jump in the opposite direction.
The character can only jump on a wall if it is facing the wall.
"""

PROMPTS["goal_description"] = """
Navigate through the level in order to touch the exit.
"""

PROMPTS["environment_checks"] = """
The character can perform the following checks:
- is_object_near(object_type: str [see objects above], direction: str [see directions above], distance: float [0.0 - 1.0]) -> bool - checks if there is any object closer than the threshold
- is_object_far(object_type: str [see objects above], direction: str [see directions above], distance: float [0.0 - 1.0]) -> bool - checks if there is any object further than the threshold
- is_on_wall() -> bool - checks if the character is on a wall
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