'''
Author: Furkan Toprak
Date (Last Updated): 08/05/2019
Unpacks winner and plots its delta chart.
Produces 2D and 3D plot of theta vs. r vs. delta theta.
'''
# Imports
import crabneat
import os
import neat
import pickle
import random
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Correct usage FIXED
if len(sys.argv) != 2:
    print("Error! Usage: python3 deltachart.py <winner>")
    exit(1)
# Presets FIXED
config_name = 'config-crab'
random.seed(31415)
winner = int(sys.argv[1])
three_d = True
# Load winner and configurations FIXED
with open('ff-abs-winner-crab-' + str(winner), 'rb') as f:
    c = pickle.load(f)
print('Loaded genome:')
print(c)
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, config_name)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)


net = neat.nn.FeedForwardNetwork.create(c, config)
sim = crabneat.Crab()
fig = plt.figure()
if three_d:
    ax = fig.add_subplot(111, projection='3d')
else:
    ax = fig.add_subplot(111)
thetas = []
radii = []
delta_thetas = []
angle_start = 0
angle_end = 2 * np.pi
distance_start = 0
distance_end = 100
angle_density = 1000
distance_density = 1000
plt.xlim(angle_start, angle_end)
plt.ylim(distance_start, distance_end)
for curr_theta in np.linspace(angle_start, angle_end, angle_density):
    for curr_radius in np.linspace(distance_start, distance_end, distance_density):
        inputs = [sim.get_radius(), sim.get_theta()]
        delta_theta = net.activate(inputs)[0]
        if three_d:
            thetas.append(curr_theta)
            radii.append(curr_radius)
            delta_thetas.append(delta_theta)
        else:
            ax.plot(curr_theta, curr_radius, '.', color=(delta_theta / (2 * np.pi), 0, 0))
if three_d:
    ax.scatter(thetas, radii, delta_thetas)
    ax.set_zlabel('Delta theta')
ax.set_xlabel('Theta')
ax.set_ylabel('Radius')
x_tick = np.arange(0, 2 * np.pi + 0.001, np.pi / 4)
x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$", r"$\frac{5\pi}{4}$",
           r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$", r"2$\pi$"]
ax.set_xticks(x_tick)
ax.set_xticklabels(x_label)
plt.show()