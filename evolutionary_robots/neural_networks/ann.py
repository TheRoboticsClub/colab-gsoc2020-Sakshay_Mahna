"""Docstring for the ann.py module

This module implements Artificial Neural Networks
The Network can have a variable number of layers,
variable number of neurons in each layer, variable
activation function and variable type of layer(Static or Dynamic)

"""

import numpy as np
import pickle
from graphviz import Digraph
from layers import StaticLayer, DynamicLayer

# Library used to genrate warnings
import warnings

class ArtificialNeuralNetwork(object):
	"""
	The Artificial Neural Network Class
	The network is created using a list of Layer interface
	
	The ANN class can:
		Generate a Neural Network
		Generate the output using the input provided
		Change the weight parameters
		Save and Load the parameters in a file
	
	...
	
	Parameters
	----------
	layer_vector: array_like
		A list of interface.Layer objects
		
		The configuration of the array should be according to the order of execution.
		It is upto the user to decide the order of execution of the Neural Network!
		The layers are indexed according to the order of execution.
		
	time_interval(optional): float
		A float specifying the time interval
		
		*Useful especially for networks with Dynamic Layers
		
	Attributes
	----------
	number_of_layers: integer
		Specifies the number of layers in the network
		
	order_of_execution: array_like
		Specifies the order in which layer outputs should be calculated to generate the overall output
		
		*It is a list of layer variables defined as the Layer interface object
		
	time_interval: float
		Float specifying the time interval
		
		*Useful for networks with Dynamic Layers
		
	output_matrix: dictionary
		A dictionary containing the current and previous outputs of all the layers in the current iteration
		
	Methods
	-------
	forward_propagate(input_dict)
		Calculate the output of the Neural Network for one iteration
		
	save_parameters_to_file(file_name)
		Save the parameters of the Neural Network to a file
		
	load_weights_from_file(file_name)
		Load the parameters of the Neural Network from a file
		
	return_parameters_as_vector()
		Return the parameters of the Neural Network as a vector
		
	load_parameters_from_vector(parmeter_vector)
		Load the parameters of the Neural Network from a vector
	"""
	
	def __init__(self, layer_vector, time_interval=0.01):
		"""
		Initialization function of ArtificialNeuralNetwork class		
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
		# Class declarations
		self.__number_of_layers = len(layer_vector)
		self._order_of_execution = [layer for layer in layer_vector]		# Ordered collection of layer variables
		self.__time_interval = time_interval
		
		# Internal Attributes
		self.__input_connections = {}		# To store the input connections of various layers
		self.__output_connections = {}		# To store the output connections of various layers
		
		self.__output_layers = []			# To store the layers that are used as output(majorly hardware layers)
		self.__input_layers = []			# To store the layers that are used as input
		
		# Construct the layers and the execution graph
		self._construct_layers()
		
		# Output matrix dictionary for Dynamic Programming Solution
		self.__output_matrix = {}
			
	
	# Function to construct the layers given the inputs
	# in essence, a Neural Network
	def _construct_layers(self):
		"""
		Private function for the construction of the layers for the Neural Network
		
		...
		
		Parameters
		----------
		None
			
		Returns
		-------
		None
		
		Raises
		------
		Exception
			If there is something wrong with the layer_vector
			
		"""
		# A dictionary for storing each layer
		self.__layer_map = {}
		
		# Helper Dictionary used to store the number of layers for each layer
		self.__neuron_map = {}
		
		# Construct the input-output dictionaries
		for layer in self._order_of_execution:
			# Store the number of neurons for each layer
			self.__neuron_map[layer[0]] = layer[1]
		
			# An instance of output connections for the current layer
			self.__output_connections[layer[0]] = layer[5]
			
			# Iterate over the output connections and fill the input connections
			for output_layer in layer[5]:
				# Layer tuple to insert as input connections
				# If output_layer is present in the 6th index, then a True to delayed
				layer_tuple = (layer[0], output_layer in layer[6])
				try:
					self.__input_connections[output_layer].append(layer_tuple)
				except:
					self.__input_connections[output_layer] = [layer_tuple] 
			
		# Iterate the layer names according to the order of execution
		for layer in self._order_of_execution:
			# Static Layer
			if(layer[2] == "STATIC"):
				# Input dimensions
				input_dimension = 0
			
				# Is it an input layer?
				try:
					for connection in self.__input_connections[layer[0]]:
						input_dimension = input_dimension + self.__neuron_map[connection[0]]
				except:
					self.__input_layers.append(layer[0])
					input_dimension = layer[1]
				
				# Output dimensions
				output_dimension = layer[1]
				
				# Activation Function
				activation_function = layer[3]
				
				# Generate the layer
				self.__layer_map[layer[0]] = StaticLayer(input_dimension, output_dimension, activation_function, layer[0])
				
			# Dynamic Layer
			elif(layer[2] == "DYNAMIC"):
				# Input dimensions
				input_dimension = 0
				
				# Is it an input layer?
				try:
					for connection in self.__input_connections[layer[0]]:
						input_dimension = input_dimension + self.__neuron_map[connection[0]]		
				except:
					self.__input_layers.append(layer[0])
					input_dimension = layer[1]
						
				# Output dimensions
				output_dimension = layer[1]
				
				# Collect the activation function
				activation_function = layer[3]
				
				# Generate the layer
				self.__layer_map[layer[0]] = DynamicLayer(input_dimension, output_dimension, activation_function, self.__time_interval, np.ones((output_dimension, )), layer[0])
				
		# Generate the output layers variable
		layer_keys = self.__output_connections.keys()
		for layers in self.__input_connections.keys():
			# Is it a hardware layer?
			if layers not in layer_keys:
				# If it is, then push to output layers
				self.__output_layers.append(layers)


	# The function to calculate the output
	def forward_propagate(self, input_dict):
		"""
		Generate the output of the Neural Network in a single iteration
		
		Parameters
		----------
		input_dict: dictionary
			Specify the input values of each layer according to the index
			Along with the associative sensor values
			
		Returns
		-------
		output_dict: dictionary
			Dictionary specifying the output of layers that do not have any output connections further(Output Layers, in essence)
			They are generally supposed to be hardware layers
			
		Raises
		------
		Exception
			Make sure the layers that provide output to the same hardware layer have the same dimensions
		
		Notes
		-----
		Dynamic Programming is used for the calculation of the output
		Based on the order of execution specified according to the user(taken as order of indices of layers),
		the output is calculated and stored in the output matrix.
		
		The layer outputs the activation for all the instances as specified by the delay
		
		"""
		# Iterate according to order of execution
		try:
			for layer in self._order_of_execution:
				# Used in various places throughout
				layer_name = layer[0]
				sensor_input = None
				
				# Get the sensor input
				try:
					sensor_input = input_dict[layer[4]]
				except:
					sensor_input = np.zeros((layer[1], ))
				
				# Concatenate the inputs required
				input_vector = np.array([])	
				
				try:
					for connection in self.__input_connections[layer_name]:
						# Additional check for delay
						try:
							if(connection[1] == True):
								# Delay required
								input_vector = np.concatenate([input_vector, self.__output_matrix[connection[0]][1]], axis = 0)
							else:
								# Delay not required
								input_vector = np.concatenate([input_vector, self.__output_matrix[connection[0]][0]], axis = 0)
								
						except:
							input_vector = np.concatenate([input_vector, np.zeros(self.__neuron_map[connection[0]])], axis = 0)
							
				except:
					input_vector = np.zeros((layer[1], ))
				
				# Calculate the output
				self.__output_matrix[layer_name] = self.__layer_map[layer_name].forward_propagate(input_vector, sensor_input)
				
		except:
			raise Exception("There is something wrong with the network configurations")
		
			
		# Return the output_dict
		output_dict = {}
		for layer in self.__output_layers:
			output_vector = None
			for connection in self.__input_connections[layer]:
				try:
					if output_vector == None:
						output_vector = np.array(self.__output_matrix[connection[0]][0])
					else:
						output_vector = np.sum(output_vector, self.__output_matrix[connection[0]][0])
				except:
					raise Exception("There is something wrong with the configuration of " + layer)
					
			output_dict[layer] = output_vector
			
		return output_dict
		
	# Function to save the layer parameters
	def save_parameters_to_file(self, file_name):
		"""
		Save the parameters of the Neural Network
		
		Using pickle, the list of layers is stored
		"""
		# Use pickle to save the layer_map
		with open(file_name, 'wb') as f:
			pickle.dump(self.__layer_map, f)
			
	# Function to load the layer parameters
	def load_weights_from_file(self, file_name):
		""" Load the parameters of the Neural Network """
		# Use pickle to load the layer_map
		with open(file_name, 'rb') as f:
			layer_map = pickle.load(f)
			
			# Load the dictionary
			self.__layer_map = layer_map
	
	# Function to return the parameters in the form of a vector		
	def return_parameters_as_vector(self):
		"""
		Return the parameters of the Neural Network
		
		Parameters
		----------
		None
		
		Returns
		-------
		output_dict: dictionary
			Dictionary specifying the parameters of each layer, keyed according to their name. The format of weights is specific to the layer.
			
		Raises
		------
		None
		
		Notes
		-----
		Input Layer does not have any weights
		"""
		# Initialize the output vector
		# Determine an individual layer's weight matrix in row major form and then it's bias
		# Then concatenate it with the previous output vector
		output_dict = {}
	
		for index in range(self.__number_of_layers):
			layer_key = "layer_" + str(index)
			if(layer_key not  in self.__input_layers):
				output_dict[layer_key] = self.__layer_map[layer_key].return_parameters()
				
		
		return output_dict
		
	# Function to load the parameters from a vector
	def load_parameters_from_vector(self, parameter_vector):
		"""
		Load the parameters of the network from a vector
		
		Parameters
		----------
		parameter_vector: array_like
			The parameter_vector is a list of list, each parameter list is indexed according to the order of computation specified by the user. The parameter follows the format specific to each layer
			
		Returns
		-------
		None
		
		Raises
		------			
		Warning
			If the user specifies parameters greater than required
		"""
		# Convert to numpy array
		parameter_vector = np.array(parameter_vector)
		
		# Load the parameters layer by layer
		for index in range(len(self._order_of_execution)):
			# For further use
			layer_name = self._order_of_execution[index][0]
			
			if(layer_name not in self.__input_layers):
				# Layer present as hidden
				self.__layer_map[layer_name].update_parameters(parameter_vector[index])
					
		# Raise a warning if the user specifies more parameters than required
		if(len(parameter_vector) > self.__number_of_layers):
			warnings.warn("The parameter vector consists of elements greater than required")
		
	# Getters and Setters
	@property
	def number_of_layers(self):
		""" Attribute for number of layers 
			Denotes the number of layers of the network
		"""
		return self.__number_of_layers
		
	@property
	def order_of_execution(self):
		""" Attribute for the order of execution 
			Specifies the order in which layer outputs should be calculated to generate the overall output
		"""
		order = [layer[0] for layer in self._order_of_execution]
		return order
		
	@property
	def time_interval(self):
		""" Attribute for time_interval
			Float specifying the time interval
		"""
		return self.__time_interval
		
	@property
	def output_matrix(self):
		""" Output matrix
			Shows the outputs of every layer
		"""
		
		return self.__output_matrix
		
	
			
	
				
		
		

