import dotenv
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from planning_agent.agent import Agent as PlanningAgent
from llm_agent.prompts import PROMPTS


class Agent:
    def __init__(self, environment: str, base_plan: str = 'scripts/llm_agent/base_agent_plan.yml'):
        dotenv.load_dotenv()
        
        self.planning_agent = PlanningAgent(base_plan)
        self.environment = environment

        self.llm_observer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.llm_critic = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.llm_planner = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.update_history = []
        

    def learn(self, steps_per_episode: int):
        done = False
        reward = 0
        obs = self.environment.reset()

        # GET NUMBER OF CONCURENT AGENTS IN ONE ENVIRONMENT
        nb_agents = len(obs["obs"])

        # ACT BASED ON THE CURRENT PLAN
        for i in range(steps_per_episode):
            # TAKE AN ACTION FOR EACH AGENT
            # TODO: make reward a list of rewards for each agent
            # TODO: make done a list of dones for each agent
            actions = [self.planning_agent.act(obs["obs"][i], reward, done) for i in range(nb_agents)]

            # FORMAT THE ACTIONS AS A NUMPY ARRAY
            actions = np.array(actions, dtype=np.int64)

            # EXECUTE THE ACTIONS INSIDE THE ENVIRONMENT
            obs, reward, done, info = self.environment.step(actions)

            # IF ANY OF THE AGENTS FINISHES OR TIME EXPIRES END THE LOOP
            if done:
                return

        report = self.planning_agent.generate_report(samples=3)
        last_step_report = self.get_last_step_report(report)
        current_plan = self.planning_agent.get_plan_as_yaml()

        critic_response = self.critique_plan(last_step_report, current_plan)
        self.update_history.append(critic_response)
        print(critic_response)

        updated_plan = self.update_plan(critic_response)
        print(updated_plan)

        self.planning_agent.add_steps_to_plan(updated_plan)
        self.planning_agent.reset()
        self.learn(steps_per_episode)


    def observe_environment(self, report: dict):
        prompt = ChatPromptTemplate.from_template(PROMPTS["game_observer"])
        observer_chain = prompt | self.llm_observer
        observer_response = observer_chain.invoke({
            "traceback": report
        })
        with open("observer_input.txt", "w") as f:
            f.write(prompt.format_prompt(
                traceback=report).to_string())
        return observer_response.content

    
    def critique_plan(self, report_description: str, current_plan: str):
        prompt = ChatPromptTemplate.from_template(PROMPTS["game_critic"])
        critic_chain = prompt | self.llm_critic
        critic_response = critic_chain.invoke({
            "current_plan": current_plan
        }, {"input": report_description})
        with open("critic_input.txt", "w") as f:
            f.write(prompt.format_prompt(
                current_plan=current_plan, 
                report_description=report_description).to_string())
        return critic_response.content


    def get_last_step_report(self, report: dict):
        keys = [key for key in report.keys() if "step" in key]
        return report[keys[-1]]


    def update_plan(self, critique: str):
        prompt = ChatPromptTemplate.from_template(PROMPTS["game_planner"])
        planner_chain = prompt | self.llm_planner
        planner_response = planner_chain.invoke({
            "critic_feedback": critique,
        })
        with open("planner_input.txt", "w") as f:
            f.write(prompt.format_prompt(critic_feedback=critique).to_string())
        return planner_response.content.replace("```", "").replace("```yaml", "").replace("```", "")
