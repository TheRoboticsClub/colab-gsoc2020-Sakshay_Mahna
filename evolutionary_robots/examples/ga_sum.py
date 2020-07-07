# An example of a Genetic Algorithm that maximizes
# the sum of it's genes

import sys
sys.path.append('./../')
import numpy as np

from genetic_algorithm.ga import GeneticAlgorithm


# Define the fitness function
# which is a sum of the alleles of the chromosome
def fitness_function(chromosome):
	return np.sum(chromosome)
	
# Intialize the Genetic Algorithm object
ga = GeneticAlgorithm()

# Set the population size of the algorithm
ga.population_size = 1000

# Set the number of generations of the algorithm
ga.number_of_generations = 250

# Set the fitness function we defined above
ga.fitness_function = fitness_function

# Run the Genetic Algorithm and print it's result
# Which is the best chromosome in our case
print(ga.run())

# Plot the fitness value against the generation number
ga.plot_fitness()
