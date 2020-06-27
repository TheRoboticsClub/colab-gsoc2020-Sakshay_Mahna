"""Docstring for interface.py

This module provides a Layer class for easier
user interface to define various attributes of a layer

"""

from activation_functions import LinearActivation
import numpy as np

class Layer(object):
	"""
	Layer Class provides an easier abstraction for developing an 
	Artificial Neural Network.
	Layer Objects can be initialized and updated according to the
	user and passed as initialization parameters to the network.
	
	Parameters
	----------
	layer_name: string
		Specifies the name of the layer
		
	number_of_neurons: integer
		Specifies the number of neurons in the layer
		
	activation_function: ActvationFunction object
		Specifies the activation function of the Layer
		Use IdentityActivation, if layer is an input layer
		
	sensor_input: string
		Specifies the sensors that input the current layer
		The name can be set whatever we want, but needs to be 
		coherent everywhere
		
		Example: "CAMERA" or "INFRARED"
		
		*Use captials
		
	output_connections: array_like
		Specifies the name of the layers and hardware the layer 
		outputs to. The array is to be a list of strings
		
		Names must be taken from the layer_name attributes 
		set for other layers. The hardware components must be 
		in capitals
		
		Example: ["layer1", "MOTORS"]
		
	
	Attributes
	----------
	The attributes are same as the parameters
	"""
	def __init__(self, layer_name, number_of_neurons = 1, activation_function = LinearActivation(), sensor_input = "", output_connections = []):
		""" 
		Initialization function of Layer
		
		...
		
		Parameters
		----------
		Parameters specified in the class docstring
			
		Returns
		-------
		None
		
		Raises
		------
		None
		
		"""
		
		# Attributes
		self.__layer_name = layer_name
		self.number_of_neurons = number_of_neurons
		self.activation_function = activation_function
		self.sensor_input = sensor_input
		self.output_connections = output_connections
		
	# Getters and Setters
	@property
	def number_of_neurons(self):
		""" The number of neurons in the layer """
		return self._number_of_neurons
		
	@number_of_neurons.setter
	def number_of_neurons(self, neuron):
		self._number_of_neurons = neuron
		
	@property
	def activation_function(self):
		""" The Activation Function of the Layer """
		# Better readability
		return type(self._activation_function)
		
	@activation_function.setter
	def activation_function(self, function):
		self._activation_function = function
		
	@property
	def sensor_input(self):
		""" The Sensor Input """
		# Parameter made for entering the sensor input
		return self._sensor_input
		
	@sensor_input.setter
	def sensor_input(self, sensor):
		self._sensor_input = sensor
			
	@property
	def output_connections(self):
		""" A list of output connections """
		return self._output_connections
		
	@output_connections.setter
	def output_connections(self, connections):
		self._output_connections = connections
		
	# Get Item to make the Layer behave as a list
	def __getitem__(self, index):
		"""
		getitem function for the class to behave like a list
		The Layers are declared as a variable and passed to the ANN class
		
		This function is an internal API
		Quite useful for internal workings and construction of the network
		"""
		if(index == 0):
			return self.__layer_name
			
		elif(index == 1):
			return self._number_of_neurons
			
		elif(index == 2):
			return self._activation_function
			
		elif(index == 3):
			return self._sensor_input
		
		elif(index == 4):
			return self._output_connections
			
		else:
			raise IndexError("List Index out of Range")
		
	
