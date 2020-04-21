"""
Author: Furkan Toprak
Date (Last Updated): 1/9/2020
Searches Parameter Space of NEAT Simulations.
"""
# Imports
import crabneat
import random
import numpy as np
import math
from multiprocessing import Process, Manager
import time

# Presets
random.seed(31415)
np.random.seed(31415)

# Limits the parameter search space.
search_space = 10

# The number of localized searches to initiate.
cream_of_crop = 10

# Increase for more reliable understanding of fitness for each configuration during general scan. (More replicable.)
general_scan_certainty = 10

# Increase for more reliable understanding of fitness for each configuration during localized search. (More replicable.)
local_search_certainty = 10

# The number of times to do the general parameter scan.
general_scan_search_times = 10

# The number of times to do a gradient descent.
local_search_times = 10

# Node structure; specifies number of hidden nodes.
node_num = 3

# The magnitude of noise added to each parameter during gradient descent. 
# Increase for a lesser probability of getting stuck in local optima.
# Decrease noise_range or increase local_search_times for lesser probability of skipping a peak.
noise_range = 0.25

# Models a sigmoid function FIXED
def sigmoid_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 2 * np.pi / (1.0 + math.exp(-z))


# Displays results. FIXED
def display(crab_item):
    print("Fitness: ", crab_item[0])
    print("Parameterization:")
    print("[",end='')
    for i in range(1, len(crab_item) - 1):
        print(np.array2string(crab_item[i], separator=', '), end='')
        if i < len(crab_item) - 2:
            print(", ")
    print("]")
    print("___________________")


# A simulation is a crab. FIXED
def simulate(w_0, w, b_0, b, trace=False):
    sim = crabneat.Crab()
    # Run the given simulation for up to num_steps time steps.
    pred_step = random.randint(0, crabneat.step_num)
    # While the crab can take steps and isn't dead.
    while sim.get_steps() < crabneat.step_num and not sim.is_dead():
        if sim.get_steps() == pred_step and random.random() <= sim.death_probability():
            sim.rip() # Dies upon predatory event.
        elif sim.get_steps() == pred_step and round(sim.get_radius()) != 0:  # Survives predatory event and burrows home.
            sim.burrow()
        else:
            inputs = np.array([[sim.get_radius(), sim.get_theta()]])
        # Makes weights in [-search_space, search_space] space randomly.
        v_sigmoid_activation = np.vectorize(sigmoid_activation)
        hidden_layer = np.dot(inputs, w_0) + b_0
        hidden_layer = v_sigmoid_activation(hidden_layer)
        delta_theta = np.dot(hidden_layer, w) + b  # Returns a list of outputs when inputted radius and theta.
        delta_theta = v_sigmoid_activation(delta_theta)
        sim.step(delta_theta)
    # Gets fitness
    fitness = sim.get_fitness()
    if trace:
        return sim
    else:
        return fitness


# A general parameter scan across the parameter space. Parallelized for efficiency.
def scan_worker(fitness_list):
    w_0 = search_space * 2 * np.random.rand(2, node_num) - search_space  # Weights between inputs & hidden layer
    w = search_space * 2 * np.random.rand(node_num, 1) - search_space  # Weights between hidden & output
    b_0 = search_space * 2 * np.random.rand(1, node_num) - search_space  # Bias of hidden layer nodes
    b = search_space * 2 * np.random.rand(1, 1) - search_space  # Bias of output node
    # Run `general_scan_certainty` number of experiments in order to have a reliable understanding of fitness with current configuration.
    avg_fitness = 0
    for _ in range(general_scan_certainty):
        avg_fitness += simulate(w_0, w, b_0, b)
    avg_fitness /= general_scan_certainty
    # Record configuration.
    fitness_list.append([avg_fitness, w_0, w, b_0, b, False])  # Appends average performance fitness, parameters, and a flag for localized search.


# Parallelizes local search with the next
def local_search_worker(fitness_list):
    # Find the next fittest crab.
    i = 1
    while i < len(fitness_list) and fitness_list[-i][-1]:
        i += 1
    if i == len(fitness_list):
        print("Warning: May be a failure in the program. If this message shows twice, there is a bug.")
    # Flag the crab as searched.
    fitness_list[-i][5] = True
    # Unpack the parameters and fitness of the crab.
    crop_crab = fitness_list[-i][1:5]
    fit_w0, fit_w, fit_b0, fit_b = crop_crab
    best_fitness = fitness_list[-i][0]
    # Start gradient descent.
    for _ in range(local_search_times):
        # Add noise to parameters.
        w_0_noise = fit_w0 + noise_range * 2 * np.random.rand(2, node_num) - noise_range
        w_noise = fit_w + noise_range * 2 * np.random.rand(node_num, 1) - noise_range
        b_0_noise = fit_b0 + noise_range * 2 * np.random.rand(1, node_num) - noise_range
        b_noise = fit_b + noise_range * 2 * np.random.rand(1, 1) - noise_range
        # Average for reproducible results.
        this_fitness = 0
        for i in range(local_search_certainty):
            this_fitness += simulate(w_0_noise, w_noise, b_0_noise, b_noise)
        this_fitness /= local_search_certainty
        # If the gradient is positive, follow it.
        if this_fitness - best_fitness > 0:
            fit_w0 = w_0_noise
            fit_w = w_noise
            fit_b0 = b_0_noise
            fit_b = b_noise
            best_fitness = this_fitness
    # Append the fittest crab produced through gradient descent.
    fitness_list.append([best_fitness, fit_w0, fit_w, fit_b0, fit_b, True])


# Runs search.
def run():
    # Traverses through nodes.
    with Manager() as manager:
        fitness_list = manager.list()  # Records fitness_list.
        processes = []
        for _ in range(general_scan_search_times):
            p = Process(target=scan_worker, args=(fitness_list,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        fitness_list = sorted(fitness_list, key=lambda x: x[0])  # Ascending order of fitness.
        for i in fitness_list:
            print(i)
        processes = []
        for _ in range(cream_of_crop):  # Localized search around cream_of_crop peaks.
            p = Process(target=local_search_worker, args=(fitness_list,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        fitness_list = sorted(fitness_list, key=lambda x: x[0])  # Ascending order of fitness.
        for i in range(1, cream_of_crop + 1):
            display(fitness_list[-i])


# Checks if run as main. FIXED
if __name__ == '__main__':
    if cream_of_crop > general_scan_search_times:
        print("Error: cream_of_crop variable cannot be greater than general_scan_search_times variable.")
    else:
        run()
