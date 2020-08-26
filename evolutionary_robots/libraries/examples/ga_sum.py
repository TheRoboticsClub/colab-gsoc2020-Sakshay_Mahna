# An example of a Genetic Algorithm that maximizes
# the sum of it's genes

import sys
sys.path.append('./../')
import numpy as np

from genetic_algorithm.ga import GeneticAlgorithm


# Define the fitness function
def fitness_function(chromosome):
	# Sum of the genes of the chromosome
	# Change the fitness function here to
	# experiment
	return np.sum(chromosome)

	
# Intialize the Genetic Algorithm object
ga = GeneticAlgorithm()

# Set the population size of the algorithm
ga.population_size = 50

# Set the chromosome length
ga.chromosome_length = 5

# Set the mutation probability
ga.mutation_probability = 0.01

# Set the number of elites
ga.number_of_elites = 4

# Set the number of generations of the algorithm
ga.number_of_generations = 1000

# Pass the fitness function as an attribute
ga.fitness_function = fitness_function

# Run the Genetic Algorithm and print it's result
# Which is the best chromosome in our case
print(ga.run())

# Plot the fitness value against the generation number
ga.plot_fitness('fitness_plot', True)

# If we want to run the generation from 25%
# completion again
print(ga.run(25))
