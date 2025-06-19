import numpy as np

from processing.tools import is_object_near, is_object_far, is_on_wall, changed_direction, describe_observation

CONDITIONS = {
    'is_object_near': is_object_near,   
    'is_object_far': is_object_far,
    'is_on_wall': is_on_wall,
    'changed_direction': changed_direction,
}

ACTIONS = [
    'do_nothing',
    'jump',
    'release_jump'
]


class Agent:
    def __init__(self, plan: list[dict]):
        self.plan = plan
        self.observation_history = []

        self.action = 0 # 0: do nothing, 1: jump
        self.current_step_index = 0

        # variables used to generate the report
        self.traceback_cutoff = 0


    def act(self, observation: list[float], reward: float, done: bool) -> int:
        if done:
            return False
        
        self.check_plan(observation)
        self.observation_history.append(observation)

        return np.array([self.action])
        
    
    def wait_until(self, observation: list[float], condition_name: str, *args):
        condition = CONDITIONS[condition_name]
        previous_observation = None if not self.observation_history else self.observation_history[-1]
        if condition(observation, previous_observation, *args):
            self.current_step_index += 1


    def update_action(self, action_name: str):
        if action_name not in ACTIONS:
            raise ValueError(f"Unknown action: {action_name}")
        
        print(action_name)
        
        match action_name:
            case 'jump':
                self.action = 1
            case _:
                self.action = 0
        
        self.current_step_index += 1

    
    def check_plan(self, observation: list[float]):
        if self.current_step_index >= len(self.plan):
            return
        
        step = self.plan[self.current_step_index]

        if 'wait_until' in step:
            cond = step['wait_until']
            condition_name = cond['condition']
            condition_arguments = cond.get('args', [])
            self.wait_until(observation, condition_name, *condition_arguments)
        elif 'action' in step:
            self.update_action(step['action'])
            self.traceback_cutoff = len(self.observation_history)
        else:
            raise ValueError(f"Unknown step format: {step}")
        
    
    def generate_report(self):
        valid_step_index = max(0, self.current_step_index - 1)
        step = self.plan[valid_step_index]
        formatted_observation_history = [
            describe_observation(observation) 
            for observation in self.observation_history[self.traceback_cutoff:self.traceback_cutoff+5]
        ]
        report = {
            'traceback': formatted_observation_history,
            'last_step': step,
            'action': ACTIONS[self.action]
        }
        return report