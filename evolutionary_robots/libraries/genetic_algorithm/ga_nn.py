"""Docstring for ga_nn.py

This module implements the Genetic
Algorithm class adapted to Neural Networks.
The Neural Networks used are taken from
the ann library
"""

import numpy as np
from ga import GeneticAlgorithm

# The Genetic Algorithm class
class GeneticAlgorithmNN(GeneticAlgorithm):
	"""
	An inherited class from GeneticAlgorithm
	This class provides an interface to use
	GeneticAlgorithm class with ArtificialNeuralNetwork
	class
	
	...
	Parameters
	----------
	neural_network: ArtificialNeuralNetwork object
		An instance of the ArtificialNeuralNetwork object
		
	Rest of the parameters are the same
	
	Attributes
	----------
	neural_network: ArtificialNeuralNetwork object
		An instance of the ArtificialNeuralNetwork object
		
	output_range: array_like
		A 2 element list defining the range to which 
		chromosome vector should be interpolated to
		
	Rest of the attributes are the same
	
	Methods
	-------
	calculate_output(input_dictionary, chromosome)
		To calculate the output of the Neural Network
		according to the chromosome and the input
		
	Rest of the methods are the same
	"""
	def __init__(self, neural_network, population_size=100,
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
		# Set the Neural Network
		self.neural_network = neural_network
		
		# Set the default range
		self.output_range = [-5, 5]
		
		# Chromosome length is derived from Neural Network
		chromosome_length = self.neural_network.number_of_parameters
		
		GeneticAlgorithm.__init__(self, population_size, number_of_generations,
								  mutation_probability, chromosome_length,
								  number_of_elites)	
		
		# Generate some class variables
		self._chromosome_setting()
		
	def _chromosome_setting(self):
		"""
		Private function to adjust the chromosome
		settings to work with the neural network
		"""
		# Get the parameter dictionary
		parameter_dictionary = self.neural_network.return_parameters_as_vector()
		
		# Purpose is to generate the dimension vector
		# This is useful for altering the chromosome
		# to work with the neural network
		self.dimension_vector = []
		
		# Use the order of initialization of the
		# neural network to generate the settings
		for index in range(len(self.neural_network._order_of_initialization)):
			layer_name = self.neural_network._order_of_initialization[index][0]
			try:
				length = parameter_dictionary[layer_name]
				self.dimension_vector.append(len(length))
			except KeyError:
				length = 0
				self.dimension_vector.append(0)
		
	def _interpolate_range(self, chromosome):
		"""
		Private function to interpolate the value of
		chromosome from [0, 1] to output_range
		"""
		# Get the ranges
		min_range = self.output_range[0]
		max_range = self.output_range[1]
		
		# Apply the interpolation formula
		interpolated_chromosome = (max_range - min_range) * chromosome + min_range
		
		return interpolated_chromosome
	
	def calculate_output(self, input_dictionary, chromosome):
		"""
		Function to calculate the output of the
		Neural Network
		
		Parameters
		----------
		input_dictionary: dictionary
			Dictionary specfying the input to various
			sensors
			
		chromosome: array-like
			An array specfying the chromosomes for which
			the output should be calculated
			
		Returns
		-------
		output: dictionary
			Dictionary specifying the outputs
			
		Raises
		------
		None
		"""
		# Load the parameters and calculate the output
		chromosome = self._interpolate_range(chromosome)
		
		# Adjust the chromosome to work with neural network
		chromosome_list = []
		dimension_index = 0
		for dimension in self.dimension_vector:
			chromosome_list.append(chromosome[dimension_index : dimension_index + dimension])
			dimension_index = dimension_index + dimension
		
		self.neural_network.load_parameters_from_vector(chromosome_list)
		output = self.neural_network.forward_propagate(input_dictionary)
		
		return output
	
	# Getters and Setters
	@property
	def neural_network(self):
		"""The Neural Network class of Algorithm
		"""
		return self._neural_network
		
	@neural_network.setter
	def neural_network(self, neural_network):
		self._neural_network = neural_network
		
	@property
	def output_range(self):
		"""
		The range of output to which the
		chromosome values should be interpolated
		"""
		return self._output_range
		
	@output_range.setter
	def output_range(self, output_range):
		output_range.sort()
		self._output_range = output_range
		
	
