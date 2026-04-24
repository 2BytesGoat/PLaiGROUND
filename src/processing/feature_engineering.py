import numpy as np


def compute_can_jump(sensor_values: dict, session_info: dict) -> float:
    """
    The player can jump if:
      - After jumping, it released the jump button and touched the floor
      - After jumping, it released the jump button and has a powerup
    """
    on_floor = bool(sensor_values["on_floor"])
    has_powerup = bool(sensor_values["has_powerup"])
    jump_pressed = int(session_info["prev_action"]) == 1  # jump action id
    
    return float(has_powerup or (on_floor and not jump_pressed))


def compute_object_sensors(
    grid: np.ndarray,
    object_index: int,
    player_cell: tuple[int, int] = (3, 3),
    max_steps: int = 4,
) -> list[float]:
    directions = [
        (0, 1),   # right
        (0, -1),  # left
        (-1, 0),  # up
        (1, 0),   # down
        (-1, -1), # up-left
        (-1, 1),  # up-right
        (1, -1),  # down-left
        (1, 1),   # down-right
    ]
    return [
        _distance_to_object_in_direction(
            grid=grid,
            object_index=object_index,
            player_cell=player_cell,
            direction=direction,
            max_steps=max_steps,
        )
        for direction in directions
    ]


def _distance_to_object_in_direction(
    grid: np.ndarray,
    object_index: int,
    player_cell: tuple[int, int],
    direction: tuple[int, int],
    max_steps: int = 4,
) -> float:
    row, col = player_cell
    d_row, d_col = direction

    for step in range(0, max_steps):
        next_row = row + d_row * step
        next_col = col + d_col * step
        if not (0 <= next_row < grid.shape[0] and 0 <= next_col < grid.shape[1]):
            break
        if grid[next_row, next_col] == object_index:
            return step * 0.25
    return 1.0
