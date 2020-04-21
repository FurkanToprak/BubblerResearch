from __future__ import print_function

import warnings
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("Agg")
sparsity = 10


def plot_stats(statistics, ylog=False, view=False, filename='ff-abs-avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(0, len(statistics.most_fit_genomes), max(1, len(statistics.most_fit_genomes)) // sparsity)
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness[0::len(avg_fitness) // sparsity], 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.fill_between(generation, avg_fitness[0::len(avg_fitness) // sparsity]
                     - stdev_fitness[0::len(stdev_fitness) // sparsity],
                     avg_fitness[0::len(avg_fitness) // sparsity] + stdev_fitness[0::len(stdev_fitness) // sparsity])
    plt.plot(generation, best_fitness[0::len(best_fitness) // sparsity], 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename, bbox_inches='tight')
    if view:
        plt.show()

    plt.close()


def plot_species(statistics, view=False, filename='ff-abs-speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T
    e_curves = []
    for i in curves:
        e_curves.append(i[0::len(i) // sparsity])
    fig, ax = plt.subplots()

    ax.stackplot(range(0, num_generations, max(1, num_generations // sparsity)), *e_curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename, bbox_inches='tight')

    if view:
        plt.show()

    plt.close()
