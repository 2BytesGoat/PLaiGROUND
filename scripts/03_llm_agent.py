from utils import setup_environment
from llm_agent.agent import Agent


def main():
    # SETUP THE ENVIRONMENT
    env = setup_environment()
    
    # INITIALIZE THE LLM AGENT
    llm_agent = Agent(env)
    
    llm_agent.learn(steps_per_episode=100)

    env.close()


if __name__ == "__main__":
    main()
