from typing import List

DIRECTIONS = [
    "UP", "UP-RIGHT", "RIGHT", "DOWN-RIGHT", 
    "DOWN", "DOWN-LEFT", "LEFT", "UP-LEFT"
]

OBJECT_TYPES = [
    "NOTHING", "WALL", "SPIKES", "DISOLVE-BLOCK", "BOUNCE-PAD", "DOUBLE-JUMP"
]


def describe_observation(observation, previous_observation=None):
    closeness_to_goal, goal_x, goal_y, velocity_x, velocity_y = observation[:5]
    signal_shift = 5 # the signal info starts at index 5
    objec_type_shift = signal_shift + 32 # the object type info starts at index 32
    
    return {
        "closeness_to_goal": 1.0 - closeness_to_goal,
        "goal_direction": direction_to_string([goal_x, goal_y]),
        "movement_direction": direction_to_string([velocity_x, velocity_y]),
        "is_sliding_on_wall": _is_sliding_on_wall_internal(observation),
        "is_in_air": _is_in_air_internal(observation),
        "changed_direction": _changed_direction_internal(observation, previous_observation),
        "sensors": {
            direction: {
                "closeness": observation[signal_shift + (i * 4)],
                "object_type": OBJECT_TYPES[int(observation[objec_type_shift + (i * 4)])],
                "is_close": observation[signal_shift + (i * 4)] >= 0.5,
                "is_far": observation[signal_shift + (i * 4)] < 0.5,
            }
            for i, direction in enumerate(DIRECTIONS)
        }
    }


def _is_in_air_internal(observation: list[float]) -> bool:
    """Internal helper to check if character is in the air using raw observation data."""
    velocity_y = observation[4]
    return velocity_y != 0 and not _is_sliding_on_wall_internal(observation)


def _is_sliding_on_wall_internal(observation: list[float]) -> bool:
    """Internal helper to check if character is sliding on wall using raw observation data."""
    closeness_to_goal, goal_x, goal_y, velocity_x, velocity_y = observation[:5]
    
    # Character is sliding on wall if touching left/right wall and not moving
    return velocity_x < 0.1 and velocity_y != 0


def _changed_direction_internal(observation: list[float], previous_observation: list[float] | None) -> bool:
    """Internal helper to check if character changed direction using raw observation data."""
    if previous_observation is None:
        return False
    
    velocity_x, velocity_y = observation[3], observation[4]
    prev_velocity_x, prev_velocity_y = previous_observation[3], previous_observation[4]
    
    current_direction = direction_to_string([velocity_x, velocity_y])
    previous_direction = direction_to_string([prev_velocity_x, prev_velocity_y])
    
    return previous_direction != current_direction


def is_object_near(observation: list[float], previous_observation: list[float], object_type: str, direction: str, threshold: float, **args) -> bool:
    """Checks if there is an object of the specified type in the given direction within a distance threshold."""
    description = describe_observation(observation, previous_observation)
    closeness = description["sensors"][direction]["closeness"]
    is_close = closeness >= threshold  # Object is near if distance value is high (1=close, 0=far)
    is_same_type = description["sensors"][direction]["object_type"] == object_type
    return is_close and is_same_type


def is_object_far(observation: list[float], previous_observation: list[float], object_type: str, direction: str, threshold: float, **args) -> bool:
    """Checks if there is an object of the specified type in the given direction beyond a distance threshold."""
    description = describe_observation(observation, previous_observation)
    closeness = description["sensors"][direction]["closeness"]
    is_far = closeness <= threshold  # Object is far if distance value is low (1=close, 0=far)
    is_same_type = description["sensors"][direction]["object_type"] == object_type
    return is_far or not is_same_type


def is_sliding_on_wall(observation: list[float], previous_observation: list[float], **args) -> bool:
    """Checks if the character is on a wall."""
    return _is_sliding_on_wall_internal(observation)


def is_in_air(observation: list[float], previous_observation: list[float], **args) -> bool:
    """Checks if the character is in the air."""
    return _is_in_air_internal(observation)


def changed_direction(observation: list[float], previous_observation: list[float] | None, **args) -> bool:
    """Checks if the character changed direction."""
    return _changed_direction_internal(observation, previous_observation)


def direction_to_string(direction: List[float]) -> str:
    direction_components = []
    if direction[1] > 0:
        direction_components.append("UP")
    elif direction[1] < 0:
        direction_components.append("DOWN")
    
    if direction[0] > 0:
        direction_components.append("RIGHT")
    elif direction[0] < 0:
        direction_components.append("LEFT")

    if len(direction_components) == 0:
        return "NONE"

    return "-".join(direction_components)