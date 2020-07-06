# An example of a Genetic Algorithm that maximizes
# the sum of it's genes

import sys
sys.path.append('./../')
import numpy as np

from genetic_algorithm.ga import GeneticAlgorithm

def fitness_function(chromosome):
	return np.sum(chromosome)
	
ga = GeneticAlgorithm()
ga.population_size = 1000
ga.number_of_generations = 250
ga.fitness_function = fitness_function

print(ga.run())

ga.plot_fitness()
