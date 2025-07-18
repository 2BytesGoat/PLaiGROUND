import yaml
import numpy as np

from processing.tools import is_object_near, is_object_far, is_sliding_on_wall, changed_direction, describe_observation

CONDITIONS = {
    'is_object_near': is_object_near,   
    'is_object_far': is_object_far,
    'is_sliding_on_wall': is_sliding_on_wall,
    'changed_direction': changed_direction,
}

ACTIONS = [
    'release_jump',
    'jump',
    'do_nothing',
]


class Agent:
    def __init__(self, plan_path: str = None):
        self.action = 0 # 0: do nothing, 1: jump
        self.current_step_index = 0
        self.observation_history = []
        self.step_traceback = {}

        self.plan = [] if not plan_path else self.load_plan_from_file(plan_path)


    def act(self, observation: list[float], reward: float, done: bool) -> int:
        if done:
            return False
        
        self.check_plan(observation)
        self.update_step_traceback(observation)

        return np.array([self.action])

    
    def reset(self):
        self.action = 0
        self.current_step_index = 0
        self.step_traceback = {}

    
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
        plan:
        - action: do_nothing
        - wait_until:
            operator: OR
            conditions:
                - operator: OR
                conditions:
                    - condition: is_object_near
                    args: [WALL, RIGHT, 0.5]
                    - condition: is_object_far
                    args: [WALL, UP, 0.2]
                - condition: is_sliding_on_wall
        - action: jump
        """
        previous_observation = None if len(self.observation_history) == 0 else self.observation_history[-1]
        
        # Handle legacy format for backward compatibility
        if isinstance(condition_spec, str):
            condition_spec = {"condition": condition_spec}
        
        if self.evaluate_condition(observation, previous_observation, condition_spec):
            self.current_step_index += 1


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
    

    def get_plan_as_yaml(self):
        return yaml.dump({"plan": self.plan})
    

    def load_plan_from_file(self, plan_path: str):
        with open(plan_path, 'r') as f:
            self.load_plan_from_yaml(f.read())
        return self.plan
    
    
    def load_plan_from_yaml(self, plan_yaml: str):
        self.plan = yaml.safe_load(plan_yaml)["plan"]
        return self.plan
    

    def add_steps_to_plan(self, plan_yaml: str):
        steps = yaml.safe_load(plan_yaml)["plan"]
        self.plan.extend(steps)
        return self.plan

    def update_step_traceback(self, observation: list[float]):
        step_index = min(self.current_step_index, len(self.plan) - 1)

        if step_index not in self.step_traceback:
            self.step_traceback[step_index] = {
            'step': self.plan[step_index],
            'action': ACTIONS[self.action],
            'observation_start': len(self.observation_history)
        }

        self.observation_history.append(observation)

    def generate_report(self, samples: int = 3):
        full_report = {}
        # save the last samples observations for each step
        for trace_i, trace in self.step_traceback.items():
            traceback_description = []
            prev_observation = None
            start_index = trace["observation_start"] + 1
            end_index = start_index + samples
            for i, observation in enumerate(self.observation_history[start_index:end_index]):
                description = describe_observation(observation, prev_observation)
                traceback_description.append({
                    f"timestamp_{i}": description
                })
                prev_observation = observation
            full_report[f"step_{trace_i}"] = traceback_description
            full_report[f"plan_{trace_i}"] = self.plan[trace_i]

        return full_report