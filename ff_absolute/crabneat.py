'''
Neat implementation of Bubbler simulation.
Date (Last Updated): 4/20/2020
Dev notes: Theta won't overflow, but may if step_num approaches a huge number.
'''
# Imports
import math

class Crab(object):
    # Constructor
    def __init__(self):
        self.max_steps = 1000  # Trajectory length
        self.x = 0  # x position of crab
        self.y = 0  # y position of crab
        self.r = 0  # Initial radius from the burrow (input).
        self.theta = 0  # orientation of crab (input).
        self.visited = [[], []] # x,y pairs of coords visited
        self.unique = set() # Keep track of unique coordinates (rounded).
        self.fitness = 0 # Fitness.
        self.efficency = 0 # Efficiency
        # Constants for the fitness function.
        self.a_coeff = 1
        self.b_ceoff = 1
        self.c_coeff = self.max_steps / math.log(2)

    # One step of trajectory.
    def step(self, delta_theta):
        # Update current theta, x, y, and r
        self.theta += delta_theta
        self.x += math.cos(self.theta)
        self.y += math.sin(self.theta)
        self.r = (self.x ** 2 + self.y ** 2) ** 0.5
        # Record x and y
        self.visited[0].append(self.x)
        self.visited[1].append(self.y)
        # Record if unique.
        self.unique.add((int(self.x), int(self.y)))
        # Punish fitness based on position.
        self.fitness -= math.exp(self.r / self.c_coeff) - 1

    # Return radius.
    def get_radius(self):
        return self.r

    # Returns theta.
    def get_theta(self):
        return self.theta

    # Returns fitness.
    def get_fitness(self):
        self.fitness /= self.max_steps
        self.fitness *= self.b_ceoff
        self.fitness += self.a_coeff * self.get_efficiency()
        return self.fitness

    # Returns efficiency
    def get_efficiency(self):
        self.efficency = len(self.unique) / self.max_steps
        return self.efficency

    # Returns trajectory.
    def get_trajectory(self):
        return self.visited;

    # Returns trajectory length
    def get_max_steps(self):
        return self.max_steps

