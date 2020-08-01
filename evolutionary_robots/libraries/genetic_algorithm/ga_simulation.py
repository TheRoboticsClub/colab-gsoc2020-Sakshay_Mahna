"""Docstring for ga_simulation.py

This module implements the Genetic
Algorithm class adapted to the Robotics
Academy Template simulation
"""

import numpy as np
from copy import deepcopy
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
		
	test_network: array_like
		Sets a network whose parameters we don't
		require to change too much.
		
		Takes in a list of paramters and converts
		it to the required neural network
		
	Rest of the attributes are the same
	
	Methods
	-------
	calculate_fitness(chromosome)
		Takes the chromosome and generates it's
		fitness for one time step
		
	determine_fitness(individual_fitness)
		Takes the fitness values for evaluation time steps
		Averages the values and returns them
		
	test_output(input_dict)
		Calculates the output of the test network
		
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
									
		self._test_network = [None] * 5
									
	def calculate_fitness(self, index):
		"""
		Takes the chromosome and generates it's
		fitness for one time step
		"""
		# Fitness calculated according to
		# fitness function
		fitness = self.fitness_function(index)
			
		# Return the value
		return fitness
		
	def determine_fitness(self, individual_fitness, chromosome):
		"""
		Takes the fitness values for evaluation time steps
		Averages the values and returns them
		"""
		# Average the fitness
		fitness = 10 * np.sum(individual_fitness) / self.evaluation_steps
		
		# Determine the best fitness
		# And when it occured the first time
		if(fitness != self.best_fitness):
			self.best_fitness = max(self.best_fitness, fitness)
			if(fitness == self.best_fitness):
				self.best_chromosome = chromosome
				self.best_generation = self.current_generation
		
		return fitness
			
	def test_output(self, input_dict, index):
		"""
		Function used to work with test network
		It calculates the output for a given input_dict
		"""
		output = self.test_network[index].forward_propagate(input_dict)
		
		return output
			
	@property
	def test_network(self):
		"""
		Used to save the network with parameters,
		so we don't have to change again and again
		"""
		return self._test_network
		
	@test_network.setter
	def test_network(self, individual):
		try:
			index, individual = individual
		except ValueError:
			raise ValueError("Pass individual and index")
			
		ready_individual = self.convert_chromosome(individual)
		self.neural_network.load_parameters_from_vector(ready_individual)
		self._test_network[index] = deepcopy(self.neural_network)
		
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
	
