"""Docstring for the ann.py module

This module implements Artificial Neural Networks
The Network can have variable number of layers,
variable number of neurons in each layer, variable
activation function and variable type of layer(Simple or Continuous)

"""

import numpy as np
import pickle
from graphviz import Digraph
from layers import InputLayer, SimpleLayer, CTRLayer

# Library used to genrate warnings
import warnings

class ArtificialNeuralNetwork(object):
	"""
	The Artificial Neural Network Class
	
	...
	
	Attributes
	----------
	layer_vector: array_like
		Specifies the configuration of each layer in an array format
		[[number_of_neurons, type_of_layer, activation_function, input_connections, output_connections], ...]	*for each layer
		
		number_of_neurons: number of neurons(input dimensions)
		type_of_layer: 0 for input, 1 for simple and 2 for ctr
		activation_function: a class with attribute calculate_activation
		input_connections: tuple which specifies the index of the layer and an optional delay as a boolen, True for delay and False for not
		output_connections: List of layers connected to the output
		
		The layer configuration should be according to the order of execution. It is upto the user to decide the order of execution
		of the Neural Network! The layers are numbered according to the order of execution.
		
	time_constants(optional): array_like
		Dictionary specifying the time constants of a given layer index
		
	time_interval(optional): integer
		Integer specifying the time interval
		
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
	
	def __init__(self, layer_vector, time_constants={}, time_interval=0.01):
		# Class declarations
		self.__number_of_layers = len(layer_vector)
		self.__order_of_execution = range(self.__number_of_layers)
		self.__time_interval = time_interval
		
		# Private declarations
		self.__input_layers = []
		self.__output_layers = []
		
		# Construct the layers and the graph
		self._construct_layers(layer_vector, time_constants)
		
		# Output matrix dictionary for Dynamic Programming Solution
		self.__output_matrix = {}
			
	
	# Function to construct the layers given the inputs
	# in essence, a Neural Network
	def _construct_layers(self, layer_vector, time_constants):
		"""
		Private function for the construction of the layers for the Neural Network
		
		...
		
		Parameters
		----------
		layer_vector: array_like
			To specify the dimensions and options of the layers of the network
			
		time_constants: array_like
			Specify the time constants of the ctr layer. Should be empty if all the layers are simple
			
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
		
		# List for input output connections
		self.__input_connections = []
		self.__output_connections = []
		
		try:
			# Append the Layer classes
			for index in range(self.__number_of_layers):
				# 0 => input layer
				if(layer_vector[index][1] == 0):
					# Input Layer
					self.__input_layers.append("layer_" + str(index))
				
					# No need to collect the input dimensions as input layer doesn't take input from other layers
					input_dimension = layer_vector[index][0]
					self.__input_connections.append([])
					
					# Collect the output dimensions
					connect = []
					for connection in layer_vector[index][4]:
						connect.append("layer_" + str(connection))
						
					self.__output_connections.append(connect)
					
					# Generate the layer
					self.__layer_map["layer_" + str(index)] = InputLayer(input_dimension, "layer_"+str(index))
					
				# 1 => Simple Layer
				elif(layer_vector[index][1] == 1):
					# Collect the input dimensions
					input_dimension = 0
					connect = []
					for connection in layer_vector[index][3]:
						# Taking account the delay as well
						 connect.append(("layer_" + str(connection[0]), connection[1]))
						 input_dimension = input_dimension + layer_vector[connection[0]][0]
						 
					self.__input_connections.append(connect)
					
					# Collect the output dimensions
					output_dimension = layer_vector[index][0]
					connect = []
					for connection in layer_vector[index][4]:
						connect.append("layer_" + str(connection))
						
					# Determine the output layer
					if(len(connect) == 0):
						self.__output_layers.append("layer_" + str(index))
							
					self.__output_connections.append(connect)	
					
					# Collect the activation function
					activation_function = layer_vector[index][2]
					
					# Generate the layer
					self.__layer_map["layer_" + str(index)] = SimpleLayer(input_dimension, output_dimension, activation_function, "layer_"+str(index))
					
				# 2 => CTR Layer
				elif(layer_vector[index][1] == 2):
					# Collect the input dimensions
					input_dimension = 0
					connect = []
					for connection in layer_vector[index][3]:
						# Taking account the delay as well
						connect.append(("layer_" + str(connection[0]), connection[1]))
						input_dimension = input_dimension + layer_vector[connection[0]][0]
						
					self.__input_connections.append(connect)
					
					# Collect the output dimensions
					output_dimension = layer_vector[index][0]
					connect = []
					for connection in layer_vector[index][4]:
						connect.append("layer_" + str(connection))
						
					# Determine the output layer
					if(len(connect) == 0):
						self.__output_layers.append("layer_" + str(index))
						
					self.__output_connections.append(connect)
						
					# Collect the activaiton function
					activation_function = layer_vector[index][2]
					
					# Generate the layer
					try:
						self.__layer_map["layer_" + str(index)] = CTRLayer(input_dimension, output_dimension, activation_function, self.__time_interval, time_constants[index], "layer_"+str(index))
					except:
						time_dimension = (output_dimension, )
						self.__layer_map["layer_" + str(index)] = CTRLayer(input_dimension, output_dimension, activation_function, self.__time_interval, np.ones(time_dimension), "layer_" + str(index))
						
		except:
			raise Exception("There is something wrong with the configuration dictionary of the network!")


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
			
		Raises
		------
		None
		
		Notes
		-----
		Dynamic Programming is used for the calculation of the output
		Based on the order of execution specified according to the user(by default taken as order of indices of layers),
		the output is calculated and stored in the output matrix.
		
		The layer outputs the activation for both the current and previous
		instances. The delay specifies which output to select!
		
		** Perfect Example of EAFP
		"""
		# Iterate according to order of execution
		for index in self.order_of_execution:
			# Used in various places throughout
			layer_key = "layer_" + str(index)
			
			if layer_key in self.__input_layers:
				try:
					# If the layer is a part of the input
					self.__output_matrix[layer_key] = self.__layer_map[layer_key].forward_propagate(input_dict[index])
				except:
					# If the user does not specify an input for the layer
					input_dimension = self.__layer_map[layer_key].weight_dim[1]
					self.__output_matrix[layer_key] = self.__layer_map[layer_key].forward_propagate(np.zeros((input_dimension, )))
				
			else:
				# If the layer is not a part of the input
				# Does the user give a sensor input?
				if index not in input_dict:
					# If it is not, we define the input for the sensor
					input_dict[index] = np.zeros(self.__layer_map[layer_key].bias_dim)
					
				# Concatenate the inputs required
				input_vector = np.array([])
				for connection in self.__input_connections[index]:
					# Additional check for delay
					try:
						if(connection[1] == True):
							# If a delay is required
							input_vector = np.concatenate([input_vector, self.__output_matrix[connection[0]][1]], axis=0)
						else:
							# Delay not required
							input_vector = np.concatenate([input_vector, self.__output_matrix[connection[0]][0]], axis=0)
					except:
						input_vector = np.concatenate([input_vector, np.zeros(self.__layer_map[connection[0]].bias_dim)], axis=0)
						
				# Calculate the output
				self.__output_matrix[layer_key] = self.__layer_map[layer_key].forward_propagate(input_vector, input_dict[index])
		
		# Return the output_dict
		output_dict = {}
		for layer in self.__output_layers:
			index = int(layer[6:])
			output_dict[index] = self.__output_matrix[layer][0]
			
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
		for index in range(self.__number_of_layers):
			# For further use
			layer_key = "layer_" + str(index)
			
			if(layer_key not in self.__input_layers):
				# Layer present as hidden
				self.__layer_map[layer_key].update_parameters(parameter_vector[index])
					
		# Raise a warning if the user specifies more parameters than required
		if(len(parameter_vector) > self.__number_of_layers):
			warnings.warn("The parameter vector consists of elements greater than required")
		
	# Getters and Setters
	@property
	def number_of_layers(self):
		""" Getter for number of layers """
		return self.__number_of_layers
		
	@property
	def order_of_execution(self):
		""" Getter for the order of execution """
		return self.__order_of_execution
		
	@order_of_execution.setter
	def order_of_execution(self, order_list):
		""" Setter for order of exeuction """
		self.__order_of_execution = order_list
		
	@property
	def time_interval(self):
		""" Getter for time_interval """
		return self.__time_interval
		
	@property
	def output_matrix(self):
		""" Getter for the output matrix
			Shows the outputs of every layer """
		
		return self.__output_matrix
		
	def set_gain(self, index, gain):
		""" Setter function for setting the gain of a layer """
		gain = np.array(gain)
		self.__layer_map["layer_" + str(index)].gain = gain
		
	def get_gain(self, index):
		""" Getter function for the gain of a layer """
		return self.__layer_map["layer_" + str(index)].gain
		
	
			
	
				
		
		

