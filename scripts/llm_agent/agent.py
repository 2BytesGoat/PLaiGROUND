import dotenv
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from planning_agent.agent import Agent as PlanningAgent
from llm_agent.prompts import PROMPTS

class Agent:
    def __init__(self, environment: str, base_plan: str = 'scripts/llm_agent/base_agent_plan.yml'):
        dotenv.load_dotenv()
        
        self.planning_agent = PlanningAgent(base_plan)
        self.environment = environment

        self.llm_critic = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.llm_planner = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        

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

        report = self.planning_agent.generate_report(cutoff=5)

        critic_response = self.critique_plan(report)
        print(critic_response)

        updated_plan = self.update_plan(critic_response, self.planning_agent.plan)
        print(updated_plan)

        self.planning_agent.load_plan_from_yaml(updated_plan)
        self.planning_agent.reset()
        self.learn(steps_per_episode)

    
    def critique_plan(self, report: dict):
        prompt = ChatPromptTemplate.from_template(PROMPTS["game_critic"])
        critic_chain = prompt | self.llm_critic
        critic_response = critic_chain.invoke({
            "report": report,
            "plan": self.planning_agent.plan,
        })
        return critic_response.content

    def update_plan(self, critique: str, plan: dict):
        prompt = ChatPromptTemplate.from_template(PROMPTS["game_planner"])
        planner_chain = prompt | self.llm_planner
        planner_response = planner_chain.invoke({
            "critique": critique,
            "plan": plan,
        })
        return planner_response.content
