PROMPTS = {}

PROMPTS["environment_description"] = """
This environment simulates a fast-paced obstacle course game similar to Geometry Dash. 
You control a character whose movement depends on holding the space bar: pressing it initiates a jump, releasing it ends the jump (or causes fall if held too long). 
Obstacles include walls of varying heights. Crucially:
- You can only stick to a wall's surface when your jump action collides with a wall that is **taller than you**.
- If you press space and collide with a wall exactly as tall as the character (both 16x16 cells), you will simply jump over it, unaffected by gravity during the jump phase relative to its height.
Additionally, there are dissolving walls which require specific interaction timing to pass safely, spikes that kill instantly on contact, powerups like double-jump granting extra abilities in air, and bounce pads providing upward momentum. The game is built on a 16x16 cell grid system.
"""


PROMPTS["goal_description"] = """
Reach the exit portal located at the end of each level without dying or getting stuck indefinitely
"""

PROMPTS["observation_info"] = """
You are controlling a character in a grid-based (16x16 cell) obstacle course game similar to Geometry Dash. The state of the world and feedback on your actions are provided via observation data.

Each observation provides information about the current situation or result of an action:

1.  **Action:** Indicates the last action you performed, such as 'release_jump'.
2.  **Observations:**
    *   `distance`: Represents the distance traveled by the character since the previous observation (likely in a horizontal direction).
    *   `goal_direction`: A vector `[x, y]` indicating the relative position of the exit portal to your current location. Positive x means it's generally to the right or ahead; negative x means left/behind; similarly for y.
    *   `velocity_vector`: A vector `[vx, vy]` representing the character's velocity (speed and direction) in that time step since the last observation.
3.  **Sensors:** Provides distance readings (`distance`) and object type (`object_type`) information for eight directions relative to your character:
    *   The sensor names describe the direction you are looking ('UP', 'RIGHT', etc.) or scanning ('UP-RIGHT' scans diagonally).
    *   `distance`: A numerical value representing how far away an obstacle (or lack thereof) is in that specific direction. Closer objects have a lower distance.
    *   `object_type`: A string indicating the type of object detected by the sensor, if any. Possible values include 'NOTHING', 'WALL', 'SPIKE', 'DISSOLVING_WALL', 'POWERUP', etc.

This observation format allows you to track your movement, know where the goal is relative to you, understand your current velocity, and crucially, perceive nearby obstacles or features (like walls) in different directions. Use this information strategically to navigate the environment, avoid hazards, interact with objects (walls, powerups), manage jumps, and reach the exit portal.
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