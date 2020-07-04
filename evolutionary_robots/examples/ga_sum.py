# An example of a Genetic Algorithm that maximizes
# the sum of it's genes

import numpy as np
import matplotlib.pyplot as plt

# This class contains the functions for initializing
# and working with the genetic algorithm
class GeneticAlgorithm(object):
	def __init__(self):
		# Initializations
		self.population_size = 1000
		self.number_of_generations = 1000
		self.mutation_rate = 0.01
		self.chromosome_length = 5
		self.number_of_elites = 0
		
		# The BEST individual
		self.best_chromosome = None
		self.best_fitness = -999
		
		# Plotting parameters
		self.max_fitness = []
		self.min_fitness = []
		self.avg_fitness = []
	
	# Generates a population of individuals
	def generate_population(self):
		# Range of the alleles is [-10, 10]
		self.population = np.random.randint(-10, 11, (self.population_size, self.chromosome_length))
	
	# Fitness Function
	def fitness_function(self, chromosome):
		# The sum is the fitness function in this case
		fitness = np.sum(chromosome)
		
		self.best_fitness = max(self.best_fitness, fitness)
		if(fitness == self.best_fitness):
			self.best_chromosome = chromosome
		
		return fitness
	
	# Generate the fitness value of the current population
	def determine_fitness(self):
		# Fitness vector stores the fitness of the current population
		self.fitness_vector = []
		
		# Iterate over the population
		for individual in self.population:
			self.fitness_vector.append(self.fitness_function(individual))
			
		# Numpy conversion, to maintain defaut settings
		self.fitness_vector = np.array(self.fitness_vector, np.float64)
		
		# Get some statistics and print
		min_fitness = self.fitness_vector.min()
		max_fitness = self.fitness_vector.max()
		sum_fitness = np.sum(self.fitness_vector)
		
		print("Max Fitness: " + str(max_fitness) + "\tMin Fitness: " + str(min_fitness) + "\tAverage Fitness: " + str(sum_fitness / self.population_size))
		
		# Append to plots
		self.min_fitness.append(min_fitness)
		self.max_fitness.append(max_fitness)
		self.avg_fitness.append(sum_fitness / self.population_size)
		
		# Remove the elites from the calculation
		if(self.number_of_elites != 0):
			elite_index = np.argpartition(self.fitness_vector, -self.number_of_elites)[-self.number_of_elites:]
			self.elites = self.population[elite_index]
			self.fitness_vector = np.delete(self.fitness_vector, elite_index)
		
		# Normalize the fitness values to lie between 0 and 1 and sum to 1
		min_fitness = self.fitness_vector.min()
		max_fitness = self.fitness_vector.max()
		
		try:
			self.fitness_vector = (self.fitness_vector - min_fitness) / (max_fitness - min_fitness)
		except:
			pass
		
		# Average the fitness
		self.fitness_vector = self.fitness_vector / np.sum(self.fitness_vector)
		
	# Roullete based selection
	def selection(self):
		# Random choice, roullete selection
		try:
			effective_population = self.population_size - self.number_of_elites
			self.roullete_selection = np.random.choice(effective_population, effective_population, p = self.fitness_vector)
		except:
			pass
		
	# Crossover Logic
	def crossover(self):
		# New population
		new_population = []
		# Based on the roullete selection, we crossover mum and dad!
		for index in range(0, self.population_size - self.number_of_elites, 2):
			mum = self.population[self.roullete_selection[index]]
			dad = self.population[self.roullete_selection[index + 1]]
			
			cross_position = np.random.randint(0, self.population_size)
			
			# Cross over
			son = np.concatenate([mum[:cross_position], dad[cross_position:]])
			daughter = np.concatenate([dad[:cross_position], mum[cross_position:]])
			
			new_population.append(son); new_population.append(daughter)
			
		# The offsprings are the new population now
		# Along with the elites
		self.population = np.array(new_population)
		try:
			self.population = np.concatenate([self.population, self.elites])
		except:
			pass
		
	# Mutation Logic
	def mutation(self):
		# Iterate over all the elements
		for row, column in np.ndindex(self.population.shape):
			# Mutate or not
			mutate = np.random.choice(2, 1, p = [1 - self.mutation_rate, self.mutation_rate])
			if(mutate[0] == 1):
				# Mutate
				self.population[row, column] = np.random.randint(-10, 11)
				
	# Plotting Function
	def plot_fitness(self):
		# Generate the range of Generations
		generations = range(1, self.number_of_generations)
		
		# Plot Max Fitness
		plt.plot(generations, self.max_fitness, label="MAX")
		
		# Plot Min Fitness
		plt.plot(generations, self.min_fitness, label="MIN")
		
		# Plot Average Fitness
		plt.plot(generations, self.avg_fitness, label="AVERAGE")
		
		# Name the axes
		plt.xlabel("Generations")
		plt.ylabel("Fitness Value")
		
		# Show
		plt.title("Fitness Plot")
		plt.legend()
		plt.show()
		
	# Run the complete Genetic Algorithm
	def run(self):
		# Generate a Random Population
		self.generate_population()
		
		# Keep going through generations with selection,
		# crossover and mutation
		for generation in range(1, self.number_of_generations):
			# For statistics
			print("Generation: " + str(generation - 1))
			
			# Determine the fitness of all the individuals
			self.determine_fitness()
			
			# Select the individuals for crossover
			self.selection()
			
			# Cross over generates the next generation
			self.crossover()
			
			# Apply mutation
			self.mutation()
			
		# Print the best fitness and chromosome
		print(self.best_fitness)
		print(self.best_chromosome)
		
		# Pyplot
		self.plot_fitness()
		
if __name__ == "__main__":
	ga = GeneticAlgorithm()
	ga.run()
			
		
			
	
