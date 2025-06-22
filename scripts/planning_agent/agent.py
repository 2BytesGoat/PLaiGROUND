import yaml
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
    def __init__(self, plan_path: str = None):
        self.observation_history = []
        self.action = 0 # 0: do nothing, 1: jump
        self.plan_observation_indexes = []
        self.current_step_index = 0

        self.plan = self.load_plan_from_file(plan_path) if plan_path else []


    def act(self, observation: list[float], reward: float, done: bool) -> int:
        if done:
            return False
        
        self.observation_history.append(observation)
        
        self.check_plan(observation)
        return np.array([self.action])
        
    
    def reset(self):
        self.observation_history = []
        self.action = 0
        self.plan_observation_indexes = []
        self.current_step_index = 0


    def evaluate_condition(self, observation: list[float], previous_observation: list[float], condition_spec):
        """
        Evaluate a single condition or a compound condition with logical operators.
        """
        if "operator" in condition_spec:
            # Handle compound conditions with logical operators
            operator = condition_spec["operator"].upper()
            conditions = condition_spec["conditions"]
            
            if operator == "AND":
                return all(self.evaluate_condition(observation, previous_observation, cond) for cond in conditions)
            elif operator == "OR":
                return any(self.evaluate_condition(observation, previous_observation, cond) for cond in conditions)
            else:
                raise ValueError(f"Unknown logical operator: {operator}")
        else:
            # Handle simple condition
            condition_name = condition_spec["condition"]
            condition_arguments = condition_spec.get("args", [])
            condition_func = CONDITIONS[condition_name]
            return condition_func(observation, previous_observation, *condition_arguments)

    
    def wait_until(self, observation: list[float], condition_spec):
        """
        Wait until a condition (simple or compound) is met.
        """
        previous_observation = None if not self.observation_history else self.observation_history[-1]
        
        # Handle legacy format for backward compatibility
        if isinstance(condition_spec, str):
            condition_spec = {"condition": condition_spec}
        
        if self.evaluate_condition(observation, previous_observation, condition_spec):
            self.current_step_index += 1
            self.plan_observation_indexes.append(len(self.observation_history) - 1)


    def update_action(self, action_name: str):
        """
        Update the action that must be taken
        """
        if action_name not in ACTIONS:
            raise ValueError(f"Unknown action: {action_name}")
        
        match action_name:
            case 'jump':
                self.action = 1
            case _:
                self.action = 0
        
        self.current_step_index += 1
        self.plan_observation_indexes.append(len(self.observation_history) - 1)

    
    def check_plan(self, observation: list[float]):
        if self.current_step_index >= len(self.plan):
            return
        
        step = self.plan[self.current_step_index]

        if 'wait_until' in step:
            condition_spec = step['wait_until']
            self.wait_until(observation, condition_spec)
        elif 'action' in step:
            self.update_action(step['action'])
        else:
            raise ValueError(f"Unknown step format: {step}")
    

    def load_plan_from_file(self, plan_path: str):
        with open(plan_path, 'r') as f:
            self.plan = yaml.safe_load(f)["plan"]
        return self.plan
    
    
    def load_plan_from_yaml(self, plan_yaml: str):
        self.plan = yaml.safe_load(plan_yaml)["plan"]
        return self.plan


    def generate_report(self, cutoff: int = 3):
        traceback = []
        for i in self.plan_observation_indexes:
            print("observation index - ", i)
            traceback.append({
                'step': self.plan[i],
                'observations': [describe_observation(obs) for obs in self.observation_history[i:i+cutoff]]
            })

        return traceback