"""Docstring for interface.py

This module provides a Layer class for easier
user interface to define various attributes of a layer

"""

from activation_functions import LinearActivation
import numpy as np

class Layer(object):
	"""
	
	"""
	def __init__(self, number_of_neurons = 1, type_of_layer = 1, activation_function = LinearActivation(), input_connections = [], output_connections = []):
		self.number_of_neurons = number_of_neurons
		self.type_of_layer = type_of_layer
		self.activation_function = activation_function
		self.input_connections = input_connections
		self.output_connections = output_connections
		
		# Some defaults
		self.gains = np.ones((number_of_neurons, ))
		self.time_constants = np.ones((number_of_neurons, ))
		
	# Getters and Setters
	@property
	def number_of_neurons(self):
		""" Getter for Number of Neurons """
		return self._number_of_neurons
		
	@number_of_neurons.setter
	def number_of_neurons(self, neuron):
		self._number_of_neurons = neuron
		self._gains = np.ones((neuron, ))
		self._time_constants = np.ones((neuron, ))
		
	@property
	def type_of_layer(self):
		""" Getter for Type of Layer """
		if(self._type_of_layer == 0):
			return "Input"
		elif(self._type_of_layer == 1):
			return "Simple"
		elif(self._type_of_layer == 2):
			return "Continuous Time Recurrent"
		else:
			return "Undefined"
		
	@type_of_layer.setter
	def type_of_layer(self, layer):
		self._type_of_layer = layer
		
	@property
	def activation_function(self):
		""" Getter for Activation Function """
		return type(self._activation_function)
		
	@activation_function.setter
	def activation_function(self, function):
		self._activation_function = function
		
	@property
	def input_connections(self):
		""" Getter for a list of input_connections """
		# Converts to integer behind the scenes
		connections = [index[0] for index in self._input_connections]
		return connections
		
	@input_connections.setter
	def input_connections(self, connections):
		# Try Except Block for any exception
		try:
			# Converts to tuple behind the scenes
			if(type(connections[0]) is int):
				self._input_connections = [[i, False] for i in connections]
				
			else:
				self._input_connections = connections
		except:
			self._input_connections = connections
			
	@property
	def delayed_connections(self):
		""" Getter for a list of connections that are delayed by one step """
		# Input connections that have a tuple value of True are delayed
		connections = [index[0] for index in self._input_connections if index[1] is True]
		return connections
		
	@delayed_connections.setter
	def delayed_connections(self, connections):
		# The input connections should have been called before calling this!
		# First convert the list of tuples to a dictionary(Lazy Execution)
		input_connections = dict(self._input_connections)
		for index in connections:
			input_connections[index] = True
			
		# Reflect the changes in input_connections
		for index in range(len(self._input_connections)):
			try:
				self._input_connections[index][1] = input_connections[self._input_connections[index][0]]
			except:
				self._input_connections[index][1] = False
			
	@property
	def output_connections(self):
		""" Getter for a list of output connections """
		return self._output_connections
		
	@output_connections.setter
	def output_connections(self, connections):
		self._output_connections = connections
		
	@property
	def gains(self):
		""" Getter for a list of gains """
		return self._gains
		
	@gains.setter
	def gains(self, gains):
		self._gains = np.array(gains)
		
	@property
	def time_constants(self):
		""" Getter for a list of time constants """
		return self._time_constants
		
	@time_constants.setter
	def time_constants(self, time_constants):
		self._time_constants = np.array(time_constants)
		
	# Get Item to make the Layer behave as a list
	def __getitem__(self, index):
		"""
		getitem function for the class to behave like a list
		The Layers are declared as a variable and passed to the ANN class
		
		This function is more of an internal API
		"""
		if(index == 0):
			return self._number_of_neurons
			
		elif(index == 1):
			return self._type_of_layer
			
		elif(index == 2):
			return self._activation_function
			
		elif(index == 3):
			return self._input_connections
			
		elif(index == 4):
			return self._output_connections
		
		elif(index == 5):
			return self._gains
			
		elif(index == 6):
			return self._time_constants
			
		else:
			raise IndexError("List Index out of Range")
		
	
