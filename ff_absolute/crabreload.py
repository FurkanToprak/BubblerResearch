"""
Author: Furkan Toprak
Date (Last Updated): 08/05/2019
Reloads crab simulation state.
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
# Correct usage FIXED
if len(sys.argv) != 3:
    print("Error! Usage: python3 crabevolve.py <num-nodes> <neat-checkpoint-#>")
    exit(1)
# Presets FIXED

random.seed(31415)
num_gens = int(sys.argv[2])
num_nodes = sys.argv[1]

# Use the NN network phenotype. FIXED
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
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
    return sim.get_fitness()


# Runs program. FIXED
def run():
    # Load the neat-checkpoint-# file.
    checkpoint_file = "neat-checkpoint-" + num_nodes + "-" + str(num_gens)
    check_freq = num_gens // 10
    checker = neat.Checkpointer(check_freq, 15 * check_freq)
    pop = checker.restore_checkpoint(checkpoint_file)
    pop.add_reporter(checker)
    # Number of processors.
    num_processors = multiprocessing.cpu_count()
    pe = neat.ParallelEvaluator(num_processors, eval_genome)
    winner = pop.run(pe.evaluate, 1)
    with open('ff-absolute-winner-' + num_nodes + "-" + str(num_gens), 'wb') as f:
        pickle.dump(winner[0], f)
    with open('ff-absolute-all-' + num_nodes + "-" + str(num_gens), 'wb') as f:
        pickle.dump(winner[1], f)


# Checks if run as main. FIXED
if __name__ == '__main__':
    run()
