'''
Author: Furkan Toprak
Date (Last Updated): 1/7/2020
Unpacks winner and plots it.
'''
# Imports
import crabneat
import os
import neat
import pickle
import random
import sys
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("Agg")

def test_genome(c, num_nodes, rank="winner"):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    net = neat.nn.FeedForwardNetwork.create(c, config)
    # A simulation is a crab.
    sim = crabneat.Crab()
    # Run the given simulation for up to num_steps time steps.
    for i in range(sim.get_max_steps()):
        inputs = [sim.get_radius(), sim.get_theta()]
        prev_radius = sim.get_radius()
        prev_theta = sim.get_theta()
        delta_theta = net.activate(inputs)[0]  # Returns a list of outputs when inputted radius and theta.
        sim.step(delta_theta)
    # Fitness and trajectory status FIXED
    fitness = sim.get_fitness()
    efficiency = sim.get_efficiency()
    trajectory = sim.get_trajectory()

    # Makes a dynamic matplotlib window. FIXME
    def update_line(num, data, line):
        line.set_data(data[..., :num])
        return line,


    fig1 = plt.figure()
    l, = plt.plot([], [], '-')
    line_ani = animation.FuncAnimation(fig1, update_line, fargs=(np.array(trajectory), l), frames=len(trajectory[0]),
                                    repeat=False, interval=3, blit=True)

    max_x = max(trajectory[0])
    min_x = min(trajectory[0])
    max_y = max(trajectory[1])
    min_y = min(trajectory[1])
    scale = max(list(map(abs, [min_y, max_y, min_x, max_x]))) * 1.1
    plt.xlim(-scale, scale)
    plt.ylim(-scale, scale)
    plt.title("Fitness:{0:.3f}          Efficiency:{1:.3f}".
            format(fitness, efficiency))
    line_ani.save("ff-absolute-movie-" + num_nodes + "-" + str(gen_num) + "-" + str(rank) + '.mp4')

# Correct usage FIXED
if len(sys.argv) != 3:
    print("Error! Usage: python3 crabtester.py <num_nodes> <gen_num>")
    exit(1)
# Presets FIXED
random.seed(31415)
gen_num = int(sys.argv[2])
num_nodes = sys.argv[1]
config_name = 'configs/config' + num_nodes
# Load winner and configurations FIXED
with open('ff-absolute-winner-' + num_nodes + "-" + str(gen_num), 'rb') as f:
    c = pickle.load(f)
    test_genome(c, num_nodes)
with open('ff-absolute-all-' + num_nodes + "-" + str(gen_num), 'rb') as f:
    cs = pickle.load(f)
    for rank in range(min(10, len(cs))):
        test_genome(cs[rank], num_nodes, rank)