DIRECTIONS = [
    "UP", "UP-RIGHT", "RIGHT", "DOWN-RIGHT", 
    "DOWN", "DOWN-LEFT", "LEFT", "UP-LEFT"
]

OBJECT_TYPES = [
    "WALL", "SPIKES", "DISOLVE-BLOCK", "BOUNCE-PAD", "DOUBLE-JUMP"
]

def describe_observation(observation):
    distance, goal_x, goal_y, velocity_x, velocity_y = observation[:5]
    signal_shift = 5 # the signal info starts at index 5
    objec_type_shift = 32 # the object type info starts at index 32
    
    return {
        "distance": distance,
        "goal_direction": [goal_x, goal_y],
        "velocity_vector": [velocity_x, velocity_y],
        "sensors": {
            direction: {
            "distance": observation[signal_shift + (i * 4)],
            "object_type": observation[objec_type_shift + (i * 4)],
            }
            for i, direction in enumerate(DIRECTIONS)
        }, 
    }

def is_object_near(observation: list[float], previous_observation: list[float], object_type: str, direction: str, threshold: float, **args) -> bool:
    """Checks if there is a wall in the given direction within a distance threshold."""
    description = describe_observation(observation)
    object_type_index = OBJECT_TYPES.index(object_type)
    print(description["sensors"][direction]["distance"], threshold)
    print(description["sensors"][direction]["object_type"], object_type_index)
    return description["sensors"][direction]["distance"] > threshold and \
        description["sensors"][direction]["object_type"] == object_type_index

def is_on_wall(observation: list[float], previous_observation: list[float], **args) -> bool:
    """Checks if the character is on a wall."""
    description = describe_observation(observation)
    return (description["sensors"]["LEFT"]["distance"] > 0.8 or \
        description["sensors"]["RIGHT"]["distance"] > 0.8) and \
        description["velocity_vector"][0] < 0.1

def changed_direction(observation: list[float], previous_observation: list[float], **args) -> bool:
    """Checks if the character changed direction."""
    # returns the sign of the number
    sign = lambda x: x and (1, -1)[x<0]

    if previous_observation is None:
        return False
    
    previous_description = describe_observation(previous_observation)
    description = describe_observation(observation)
    return sign(previous_description["velocity_vector"][0]) != sign(description["velocity_vector"][0])
