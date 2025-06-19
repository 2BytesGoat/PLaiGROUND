import numpy as np

from processing.tools import is_object_near, is_on_wall, changed_direction

CONDITIONS = {
    'is_object_near': is_object_near,   
    'is_on_wall': is_on_wall,
    'changed_direction': changed_direction
}

ACTIONS = [
    'jump',
    'release_jump',
    'do_nothing'
]


class Agent:
    def __init__(self, plan: list[dict]):
        self.plan = plan

        self.action = 0 # 0: do nothing, 1: jump
        self.previous_observation = None
        self.current_step_index = 0


    def act(self, observation: list[float], reward: float, done: bool) -> int:
        if done:
            return False
        
        self.check_plan(observation)
        self.previous_observation = observation

        return np.array([self.action])
        
    
    def wait_until(self, observation: list[float], condition_name: str, *args):
        condition = CONDITIONS[condition_name]
        if condition(observation, self.previous_observation, *args):
            self.current_step_index += 1


    def update_action(self, action_name: str):
        if action_name not in ACTIONS:
            raise ValueError(f"Unknown action: {action_name}")
        
        match action_name:
            case 'jump':
                self.action = 1
            case _:
                self.action = 0

    
    def check_plan(self, observation: list[float]):
        step = self.plan[self.current_step_index]

        if 'wait_until' in step:
            cond = step['wait_until']
            self.wait_until(observation, cond['condition'], *cond['args'])
        elif 'action' in step:
            self.update_action(step['action'])
        else:
            raise ValueError(f"Unknown step format: {step}")