"""
NEAT example using the Dragon Jump environment.
"""

import os
import sys

import neat
import visualize
import numpy as np

from scripts.utils import setup_environment


def eval_genomes(genomes, config):
    nb_agents = len(genomes)
    env = setup_environment(nb_agents=nb_agents)
    obs = env.reset()

    # Build networks for each genome
    nets = []
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

    # One genome per agent
    nb_agents = len(nets)
    fitnesses = [0.0 for _ in range(nb_agents)]

    done = [False] * nb_agents
    closeness_to_goal = [0.0 for _ in range(nb_agents)]
    
    for _ in range(255):
        actions = []
        for i in range(nb_agents):
            if not done[i]:
                input_data = obs["obs"][i]
                output = nets[i].activate(input_data)
                action = 1 if output[0] > 0.5 else 0
                actions.append(action)
            else:
                actions.append(0)  # dummy action

        actions = np.array(actions, dtype=np.int64).reshape(-1, 1)
        obs, rewards, done, _ = env.step(actions)

        for i in range(nb_agents):
            closeness_to_goal[i] = max(closeness_to_goal[i], obs["obs"][i][0])
            fitnesses[i] += rewards[i]

        if any(done):
            break

    for i in range(nb_agents):
        fitnesses[i] += closeness_to_goal[i] * 10

    env.close()

    # Assign fitness back to genomes
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = fitnesses[i]


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations (reduced from 300 since environment episodes take longer)
    winner = p.run(eval_genomes, n=50)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Test the winning genome
    print('\nTesting winner:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Run a test episode
    env = setup_environment()
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    max_steps = 1000
    
    while steps < max_steps:
        observation = obs["obs"][0]
        output = winner_net.activate(observation)
        action = [1 if output[0] > 0.5 else 0]
        
        obs, reward, done, info = env.step(np.array([action], dtype=np.int64))
        total_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
        steps += 1
        
        if done[0] if isinstance(done, (list, np.ndarray)) else done:
            break
    
    env.close()
    print(f"Test episode: {steps} steps, total reward: {total_reward}")

    # Visualize the network
    node_names = {i: f'input_{i}' for i in range(-69, 0)}  # 69 inputs
    node_names[0] = 'jump_action'  # 1 output
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)