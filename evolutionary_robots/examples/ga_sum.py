# An example of a Genetic Algorithm that maximizes
# the sum of it's genes

import sys
sys.path.append('./../')
import numpy as np

from genetic_algorithm.ga import GeneticAlgorithm
	
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

# Set the fitness function we defined above
# Define the fitness function
def fitness_function(chromosome):
	# Sum of the alleles of the chromosome
	return np.sum(chromosome)

# Pass the fitness function as an attribute
ga.fitness_function = fitness_function

# Run the Genetic Algorithm and print it's result
# Which is the best chromosome in our case
print(ga.run())

# Plot the fitness value against the generation number
ga.plot_fitness()
