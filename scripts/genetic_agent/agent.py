import neat
import numpy as np

from scripts.utils import setup_environment


def learn(config_file):
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