# An example of a Genetic Algorithm that maximizes
# the sum of it's genes

import numpy as np

# This class contains the functions for initializing
# and working with the genetic algorithm
class GeneticAlgorithm(object):
	def __init__(self):
		# Initializations
		self.population_size = 100
		self.number_of_generations = 200
		self.mutation_rate = 0.01
		self.chromosome_length = 5
		self.number_of_elites = 4
		
		# The BEST individual
		self.best_chromosome = None
		self.best_fitness = -999
	
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
		
		# Normalize the fitness values to lie between 0 and 1 and sum to 1
		min_fitness = self.fitness_vector.min()
		max_fitness = self.fitness_vector.max()
		sum_fitness = np.sum(self.fitness_vector)
		try:
			self.fitness_vector = (self.fitness_vector - min_fitness) / (max_fitness - min_fitness)
		except:
			pass
		
		# Average the fitness
		self.fitness_vector = self.fitness_vector / np.sum(self.fitness_vector)
		
		# Print some statistics
		print("Max Fitness: " + str(max_fitness) + "\tMin Fitness: " + str(min_fitness) + "\tAverage Fitness: " + str(sum_fitness / self.population_size))
		
	# Roullete based selection
	def selection(self):
		# Random choice, roullete selection
		try:
			self.roullete_selection = np.random.choice(self.population_size, self.population_size, p = self.fitness_vector)
		except:
			pass
		
		# Cross over generates the next generation
		self.crossover()
		
	# Crossover Logic
	def crossover(self):
		# New population
		new_population = []
		# Based on the roullete selection, we crossover mum and dad!
		for index in range(0, self.population_size, 2):
			mum = self.population[self.roullete_selection[index]]
			dad = self.population[self.roullete_selection[index + 1]]
			
			cross_position = np.random.randint(0, self.population_size)
			
			# Cross over
			son = np.concatenate([mum[:cross_position], dad[cross_position:]])
			daughter = np.concatenate([dad[:cross_position], mum[cross_position:]])
			
			new_population.append(son); new_population.append(daughter)
			
		# The offsprings are the new population now
		self.population = np.array(new_population)
		
	# Mutation Logic
	def mutation(self):
		# Iterate over all the elements
		for row, column in np.ndindex(self.population.shape):
			# Mutate or not
			mutate = np.random.choice(2, 1, p = [1 - self.mutation_rate, self.mutation_rate])
			if(mutate[0] == 1):
				# Mutate
				self.population[row, column] = np.random.randint(-10, 11)
		
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
			
			# Select the individuals, the next generation
			self.selection()
			
			# Apply mutation
			self.mutation()
			
		# Print the best fitness and chromosome
		print(self.best_fitness)
		print(self.best_chromosome)
		
if __name__ == "__main__":
	ga = GeneticAlgorithm()
	ga.run()
			
		
			
	
