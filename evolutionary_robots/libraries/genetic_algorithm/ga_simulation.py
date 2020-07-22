"""Docstring for ga_simulation.py

This module implements the Genetic
Algorithm class adapted to the Robotics
Academy Template simulation
"""

import numpy as np
from genetic_algorithm.ga_nn import GeneticAlgorithmNN

class GeneticAlgorithmGazebo(GeneticAlgorithmNN):
	"""
	An inherited class from GeneticAlgorithmNN
	This class provides an interface to use
	GeneticAlgorithm and ArtificialNeuralNetwork 
	class with Robotics Academy Template Gazebo
	Simulation
	
	...
	Parameters
	----------
	neural_network: ArtificialNeuralNetwork object
		An instance of the ArtificialNeuralNetwork object
		
	evaluation_steps: integer
		The number of time steps for which
		to evaluate the simulation
		
	Rest of the parameters are the same
	
	Attributes
	----------
	evaluation_steps: integer
		The number of time steps for which
		to evaluate the simulation
		
	Rest of the attributes are the same
	
	Methods
	-------
	calculate_fitness(chromosome)
		Takes the chromosome and generates it's
		fitness for one time step
		
	determine_fitness(individual_fitness)
		Takes the fitness values for evaluation time steps
		Averages the values and returns them
		
		
	Rest of the methods are the same
	"""
	def __init__(self, neural_network, evaluation_steps=100, population_size=100,
				 number_of_generations=10, mutation_probability=0.01,
				 number_of_elites=0):
				 
		"""
		Initialization function of the class
		...
		
		Parameters
		----------
		Specified in the class docstring
		
		Returns
		-------
		None
		
		Raises
		------
		None
		"""
		self.evaluation_steps = evaluation_steps
		
		# Initialize the Parent Class
		GeneticAlgorithmNN.__init__(self, neural_network, population_size,
									number_of_generations, mutation_probability,
									number_of_elites)
									
	def calculate_fitness(self, chromosome):
		"""
		Takes the chromosome and generates it's
		fitness for one time step
		"""
		# Fitness calculated according to
		# fitness function
		fitness = self.fitness_function(chromosome)
			
		# Return the value
		return fitness
		
	def determine_fitness(self, individual_fitness, chromosome):
		"""
		Takes the fitness values for evaluation time steps
		Averages the values and returns them
		"""
		# Average the fitness
		fitness = np.sum(individual_fitness) / self.evaluation_steps
		
		# Determine the best fitness
		# And when it occured the first time
		if(fitness != self.best_fitness):
			self.best_fitness = max(self.best_fitness, fitness)
			if(fitness == self.best_fitness):
				self.best_chromosome = chromosome
				self.best_generation = self.current_generation
		
		return fitness
		
	def fraction_save(self, generation):
		"""
		Determines if the generation is a multiple of
		the replay fraction and saves it, if it is!
		"""
		try:
			# Check the current fraction and save if required
			if(generation % int(self.replay_fraction * (self.number_of_generations)) == 0):
				fraction = float(generation) / float(self.number_of_generations)
				self.save_chromosome(self.population, './log/generation' + str(int(100 * fraction)) + "%", 
									 "Generation #" + str(self.current_generation))
		except ZeroDivisionError:
			pass
		
	@property
	def evaluation_steps(self):
		""" 
		The number of time steps for which
		the fitness function should be evaluated
		"""
		return self._evaluation_steps
		
	@evaluation_steps.setter
	def evaluation_steps(self, steps):
		if(steps <= 0):
			steps = 100
			
		self._evaluation_steps = steps
	
