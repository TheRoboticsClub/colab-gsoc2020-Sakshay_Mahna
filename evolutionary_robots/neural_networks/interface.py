"""Docstring for interface.py

This module provides a Layer class for easier
user interface to define various attributes of a layer

"""

from activation_functions import LinearActivation
import numpy as np

class Layer(object):
	"""
	Layer Class provides an easier abstraction for developing an Artificial Neural Network
	Layer Objects can be initialized and updated according to the user and passed as initialization parameters to the network
	
	Parameters
	----------
	layer_name: string
		Specifies the name of the layer
		
	number_of_neurons: integer
		Specifies the number of neurons in the layer
		
	type_of_layer: string
		Specifies the type of layer
		There are two options: "STATIC" or "DYNAMIC"
		
		*Use capitals
		
	activation_function: ActvationFunction object
		Specifies the activation function of the Layer
		Use IdentityActivation, if layer is an input layer
		
	sensor_input: string
		Specifies the sensors that input the current layer
		The name can be set whatever we want, but needs to be coherent everywhere
		
		Example: "CAMERA" or "INFRARED"
		
		*Use captials
		
	output_connections: array_like
		Specifies the name of the layers and hardware the layer outputs to
		The array is to be a list of strings
		
		Names must be taken from the layer_name attributes set for other layers
		The hardware components must be in capitals
		
		Example: ["layer1", "MOTORS"]
		
	
	Attributes
	----------
	The attributes are same as the parameters
		
		
	Additional Attributes
	---------------------
		
	delayed_connections: array_like
		Specifies the list of output connections whose input should be delayed before sending to other layer
		The array is to be a list of strings
		
		* The names should be chosen from the output connections
		
		Example: ["layer1", "layer2"]
	"""
	def __init__(self, layer_name, number_of_neurons = 1, type_of_layer = "STATIC", activation_function = LinearActivation(), sensor_input = "", output_connections = []):
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
		self.type_of_layer = type_of_layer
		self.activation_function = activation_function
		self.sensor_input = sensor_input
		self.output_connections = output_connections
		
		# Default attribute
		self.delayed_connections = []
		
	# Getters and Setters
	@property
	def number_of_neurons(self):
		""" The number of neurons in the layer """
		return self._number_of_neurons
		
	@number_of_neurons.setter
	def number_of_neurons(self, neuron):
		self._number_of_neurons = neuron
		
	@property
	def type_of_layer(self):
		""" The type of Layer """
		return self._type_of_layer
		
	@type_of_layer.setter
	def type_of_layer(self, layer):
		# Just using the first letter of the layer word
		# So spelling mistakes are taken care of!
		if(layer[0].upper() == "D"):
			self._type_of_layer = "DYNAMIC"
		elif(layer[0].upper() == "S"):
			self._type_of_layer = "STATIC"
		else:
			# Backup Plan, if user enters something entirely different
			self._type_of_layer = "STATIC"
		
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
		
	@property
	def delayed_connections(self):
		""" A list of connections that are delayed by one step """
		return self._delayed_connections
		
		
	@delayed_connections.setter
	def delayed_connections(self, connections):
		# The connections that are delayed should be present as output connections
		self._delayed_connections = [layer for layer in connections if layer in self._output_connections]
		
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
			return self._type_of_layer
			
		elif(index == 3):
			return self._activation_function
			
		elif(index == 4):
			return self._sensor_input
		
		elif(index == 5):
			return self._output_connections
			
		elif(index == 6):
			return self._delayed_connections
			
		else:
			raise IndexError("List Index out of Range")
		
	
