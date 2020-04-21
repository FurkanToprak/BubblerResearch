"""
Author: Furkan Toprak
Date (Last Updated): 1/9/2020
Takes a parameter set from paramsearch.py results and produces a movie.
"""
# Imports
import crabneat
import random
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Presets
random.seed(31415)
np.random.seed(31415)


# Models a sigmoid function
def sigmoid_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 2 * np.pi / (1.0 + math.exp(-z))


# Updates animation
def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,


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


def run():
    # Traverses through nodes.
    # If cream of crop crab, save trajectory.
    crop_crab = [[[2.91637212, 3.5034834 ],
 [9.41407972, 7.69608729]], 
[[8.18131685],
 [6.47185933]], 
[[-0.26580558,  2.41987159]], 
[[2.87024926]]]

    most_fit_crab = simulate(*crop_crab, trace=True)
    trajectory = most_fit_crab.get_trajectory()
    is_dead = most_fit_crab.is_dead()
    efficiency = most_fit_crab.get_efficiency()
    best_fitness = most_fit_crab.get_fitness()
    fig1 = plt.figure()
    l, = plt.plot([], [], '-')
    line_ani = animation.FuncAnimation(fig1, update_line,
                                       fargs=(np.array(trajectory), l),
                                       frames=len(trajectory[0]), repeat=False,
                                       interval=3, blit=True)
    max_x = max(trajectory[0])
    min_x = min(trajectory[0])
    max_y = max(trajectory[1])
    min_y = min(trajectory[1])
    scale = max(list(map(abs, [min_y, max_y, min_x, max_x]))) * 1.1
    plt.xlim(-scale, scale)
    plt.ylim(-scale, scale)
    plt.title("Fitness:{0:.3f}|Dies:{1:s}|Efficiency:{2:.3f}".
              format(best_fitness, str(is_dead), efficiency))
    line_ani.save("ff-abs-movie-node-" + str(len(crop_crab[0][0])) + "fit-" + str(best_fitness) + '.mp4')
    plt.close()
    plt.clf()


# Checks if run as main. FIXED

if __name__ == '__main__':
    run()
