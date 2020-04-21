"""
Author: Furkan Toprak
Date (Last Updated): 1/7/2020
Crab simulation using forward-feeding neural network.
"""
# Imports
import crabneat
import os
import neat
import pickle
import random
import sys
import statistics
import crabvisualize
import multiprocessing
# Correct usage 
if len(sys.argv) != 3:
    print("Error! Usage: python3 crabevolve.py <num_nodes> <num_gens>")
    exit(1)
# Presets
random.seed(31415)
runs_per_net = 10
num_gens = int(sys.argv[2])
num_nodes = sys.argv[1]
config_name = 'configs/config' + num_nodes 
# Use the NN network phenotype.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # Logs of fitness
    fitnesses = []
    # Runs each simulation 'runs_per_net' times.
    for runs in range(runs_per_net):
        # A simulation is a crab.
        sim = crabneat.Crab()
        # Run the given simulation for up to num_steps time steps.
        for i in range(sim.get_max_steps()):
            inputs = [sim.get_radius(), sim.get_theta()]
            prev_radius = sim.get_radius()
            prev_theta = sim.get_theta()
            delta_theta = net.activate(inputs)[0]  # Returns a list of outputs when inputted radius and theta.
            sim.step(delta_theta)
        # Gets fitness
        fitness = sim.get_fitness()
        fitnesses.append(fitness)
    return statistics.mean(fitnesses)  # Average fitness over runs is the genome's fitness.


# Runs program.
def run():
    # Load the config-crab file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    check_freq = num_gens // 10
    pop.add_reporter(neat.Checkpointer(check_freq, 15 * check_freq, filename_prefix='neat-checkpoint-' + num_nodes + '-'))
    # Number of processors.
    num_processors = multiprocessing.cpu_count()
    pe = neat.ParallelEvaluator(num_processors, eval_genome)
    winner = pop.run(pe.evaluate, num_gens)
    # Save the winner.
    with open('ff-absolute-winner-' + num_nodes + "-" + str(num_gens), 'wb') as f:
        pickle.dump(winner[0], f)

    with open('ff-absolute-all-' + num_nodes + "-" + str(num_gens), 'wb') as f:
        pickle.dump(winner[1], f)
    crabvisualize.plot_stats(stats, ylog=True, view=False, filename="ff-absolute-fitness-" + num_nodes + "-" + str(num_gens) + ".svg")
    crabvisualize.plot_species(stats, view=False, filename="ff-absolute-speciation-" + num_nodes + "-" + str(num_gens) + ".svg")


# Checks if run as main.
if __name__ == '__main__':
    run()

