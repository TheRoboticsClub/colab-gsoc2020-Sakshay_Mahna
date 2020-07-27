"""Docstring for the ann.py module

This module implements Artificial Neural Networks
which can be Static or Dynamic. The Network can have
a variable number of layers, variable number of neurons
in each layer and a variable activation function

"""

import numpy as np
import pickle
from graphviz import Digraph
from layers import StaticLayer, DynamicLayer
from datetime import datetime

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
		
		The configuration of the array is recommended to be
		of the correct order of initialization.
		
	type_of_network: string
		Specifies the type of network
		
		The parameter can take values as "STATIC" or "DYNAMIC"
		
	time_interval(optional): float
		A float specifying the time interval
		
		*Useful especially for networks with Dynamic Layers
		
	Attributes
	----------
	number_of_layers: integer
		Specifies the number of layers in the network
		
	time_interval: float
		Float specifying the time interval
		
		*Useful for networks with Dynamic Layers
		
	output_matrix: dictionary
		Shows the output of each layer in the previous iteration of 
		the network
		
	number_of_parameters: integer
		The number of trainable parameters of the Neural
		Network
		
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
	
	def __init__(self, layer_vector, type_of_network, time_interval=0.01):
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
		self.type_of_network = type_of_network			# A setter function works behind the scenes
		self._order_of_initialization = [layer for layer in layer_vector]
		self.__time_interval = time_interval
		
		# Internal Attributes
		self.__input_connections = {}		# To store the input connections of various layers
		self.__output_connections = {}		# To store the output connections of various layers
		
		self.__output_layers = []			# To store the layers that are used as output(majorly hardware layers)
		self.__input_layers = []			# To store the layers that are used as input
		
		# Output and State matrix dictionary
		self.__output_matrix = {}
		self.__state_matrix = dict((layer[0], np.zeros((layer[1], ))) for layer in layer_vector)
		
		# Construct the layers and the execution graph
		self._order_of_execution = []
		self._construct_layers(layer_vector)
		self._construct_graph()
			
	
	# Function to construct the layers given the inputs
	# in essence, a Neural Network
	def _construct_layers(self, layer_vector):
		"""
		Private function for the construction of the layers for 
		the Neural Network
		
		...
		
		Parameters
		----------
		layer_vector: array_like
		A list of interface.Layer objects
		
		The network is constructed using the parameters
		of the objects
			
		Returns
		-------
		None
		
		Raises
		------
		None
			
		"""
		# A dictionary for storing each layer
		self.__layer_map = {}
		
		# Helper Dictionary used to store the number of neurons for each layer
		self.__neuron_map = {}
		
		# Construct the input-output dictionaries
		for layer in layer_vector:
			# Store the number of neurons for each layer
			self.__neuron_map[layer[0]] = layer[1]
		
			# An instance of output connections for the current layer
			self.__output_connections[layer[0]] = layer[4]
			
			# Iterate over the output connections and fill the input connections
			for output_layer in layer[4]:
				try:
					self.__input_connections[output_layer].append(layer[0])
				except KeyError:
					self.__input_connections[output_layer] = [layer[0]] 
		
		
		# Generate the layers		
		for layer in layer_vector:	
			# Static Layer
			if(self.__type_of_network == "STATIC"):
				# Input dimensions
				input_dimension = 0
			
				# Is it an input layer?
				try:
					for connection in self.__input_connections[layer[0]]:
						input_dimension = input_dimension + self.__neuron_map[connection]
				except KeyError:
					self.__input_layers.append(layer[0])
					input_dimension = layer[1]
				
				# Output dimensions
				output_dimension = layer[1]
				
				# Activation Function
				activation_function = layer[2]
				
				# Generate the layer
				self.__layer_map[layer[0]] = StaticLayer(input_dimension, output_dimension,
														 activation_function, layer[0])
				
			# Dynamic Layer
			elif(self.__type_of_network == "DYNAMIC"):
				# Input dimensions
				input_dimension = 0
				
				# Is it an input layer?
				try:
					for connection in self.__input_connections[layer[0]]:
						input_dimension = input_dimension + self.__neuron_map[connection]		
				except KeyError:
					self.__input_layers.append(layer[0])
					input_dimension = layer[1]
						
				# Output dimensions
				output_dimension = layer[1]
				
				# Collect the activation function
				activation_function = layer[2]
				
				# Generate the layer
				if(layer[0] in self.__input_layers):
					self.__layer_map[layer[0]] = StaticLayer(input_dimension, output_dimension, 
															 activation_function, layer[0])
				else:
					self.__layer_map[layer[0]] = DynamicLayer(input_dimension, output_dimension, 
															  activation_function, self.__time_interval, 
															  np.ones((output_dimension, )), layer[0])
				
		# Generate the output layers variable
		layer_keys = self.__output_connections.keys()
		for layers in self.__input_connections.keys():
			# Is it a hardware layer?
			if layers not in layer_keys:
				# If it is, then push to output layers
				self.__output_layers.append(layers)
				
	# Generate the computational graph
	def _construct_graph(self):
		"""
		Private function for the generation of the 
		computational graph of the Neural Network
		
		...
		
		Parameters
		----------
		None
			
		Returns
		-------
		None
		
		Raises
		------
		Exception:
			There should be no recurrent connections in Static
			Neural Networks
		
		Notes
		-----
		Algorithm:
		1. Iterate the layers through order of initialization
		2. For input layers the input is taken from sensor input only, 
		   input_vector is taken as zero
		3. For other layers the input is taken as a concatenation of 
		   vectors from output matrix(if static) or state matrix(if dynamic)
		4. If the output matrix gives a key error, we keep the current layer 
		   in an error queue
		5. The error queue is iterated again and again to reduce it's size to 0
		6. If the error queue is not reducing in size, then the network is not
		   correct(only in case of Static, Dynamic will not give any such problem)
		"""
		error_queue = []			# Saves the objects of which are yet to be generated
		error_index = 0				# To store the index of layer that is currently to be generated
		self.__output_matrix = {}
		
		# Initialization
		error_queue = [layer[0] for layer in self._order_of_initialization]
		prev_length = len(error_queue)
		
		while(len(error_queue) != 0):
			# The layer to work on
			layer = error_queue[error_index]
		
			# Generate the input vector
			input_vector = np.array([])
			
			# Get the input from other layers
			# If the layer is an input layer, then we have to pass a
			# constant vector of zero
			if(layer in self.__input_layers):
				input_vector = np.zeros((self.__neuron_map[layer],))
				
			# If the layer is not an input layer, then it needs to concatenate it's inputs
			else:
				# An error flag
				error_flag = False
				
				for connection in self.__input_connections[layer]:
					# An additional check for delay
					if(self.__type_of_network == "DYNAMIC"):
						# If the network is DYNAMIC then the input is taken from state matrix
						input_vector = np.concatenate([input_vector, self.__state_matrix[connection]], axis=0)
					else:
						# If the network is STATIC then the input is taken from output matrix
						# A try except block if we try accessing an element of output matrix that is
						# not yet declared
						try:
							input_vector = np.concatenate([input_vector, self.__output_matrix[connection]], axis=0)
						except KeyError:
							error_flag = True
							break
							
				# The proceeding steps can only be performed if
				# the flag has not been raised
				if(error_flag == True):
					error_index = (error_index + 1) % len(self.error_queue)
					if(error_index == 0):
						# The error_index has completed one round
						exception_string = "The Static Neural Network seems to contain some recurrent connections"
						assert prev_length != len(self.error_queue), exception_string
						prev_length = len(self.error_queue)
					
					continue
					
			# If all the above steps complete perfectly,
			# make an entry to output matrix
			self.__output_matrix[layer] = np.zeros((self.__neuron_map[layer],))
			self._order_of_execution.append(layer)
			error_queue.pop(0)
			error_index = 0
				
							
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
			Dictionary specifying the output of layers that do not have any 
			output connections further(Output Layers, in essence).
			
			They are generally supposed to be hardware layers
			
		Raises
		------
		Exception
			Make sure the layers that provide output to the same hardware layer 
			have the same dimensions
		
		Notes
		-----
		Get the sensor input from the input dictionary passed by the user
		And feed the dictionary to the tensorflow session
		
		"""
		# Iterate through the layers

		sensor_input = {}
		output = {}
		
		for layer in self._order_of_initialization:
			# Get the sensor input
			try:
				sensor_input[layer[0]] = input_dict[layer[3]]
			except KeyError:
				sensor_input[layer[0]] = np.zeros((self.__neuron_map[layer[0]], ))

		# Calculate the output according to order of execution
		if(self.__type_of_network == "STATIC"):
			for layer in self._order_of_execution:
				input_vector = np.array([])
				
				if(layer in self.__input_layers):
					input_vector = np.zeros((self.__neuron_map[layer]))
					self.__output_matrix[layer] = self.__layer_map[layer].forward_propagate(
												  input_vector, sensor_input[layer])
				else:
					# A small bottleneck
					for connection in self.__input_connections[layer]:
						input_vector = np.concatenate([input_vector, self.__output_matrix[connection]])
					self.__output_matrix[layer] = self.__layer_map[layer].forward_propagate(
												  input_vector, sensor_input[layer])
												  
			output = self.__output_matrix
												  
		else:
			for layer in self._order_of_execution:
				input_vector = np.array([])
				
				if(layer in self.__input_layers):
					input_vector = np.zeros((self.__neuron_map[layer]))
					self.__output_matrix[layer] = self.__layer_map[layer].forward_propagate(
												  input_vector, sensor_input[layer])
				else:
					# A small bottleneck
					for connection in self.__input_connections[layer]:
						input_vector = np.concatenate([input_vector, self.__state_matrix[connection]])
						
					print(self.__state_matrix)
					self.__output_matrix[layer] = self.__layer_map[layer].forward_propagate(
												  input_vector, sensor_input[layer])
												  
			output = self.__state_matrix
			self.__state_matrix = self.__output_matrix
		
		# Return the output_dict
		output_dict = {}
		for layer in self.__output_layers:
			output_vector = None
			# Collect all the hardware stuff
			for connection in self.__input_connections[layer]:
				try:
					if output_vector == None:
						output_vector = np.array(output[connection])
					else:
						output_vector = np.sum(output_vector, output[connection])
				except:
					raise RuntimeError("There is something wrong with the configuration of " + layer)
					
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
			Dictionary specifying the parameters of each layer, 
			keyed according to their name.
			The format of weights is specific to the layer.
			
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
	
		for index in range(len(self._order_of_initialization)):
			# For further use
			layer_name = self._order_of_initialization[index][0]
			
			if(layer_name not  in self.__input_layers):
				# Layer present as hidden
				output_dict[layer_name] = self.__layer_map[layer_name].return_parameters()
				
		
		return output_dict
		
	# Function to load the parameters from a vector
	def load_parameters_from_vector(self, parameter_vector):
		"""
		Load the parameters of the network from a vector
		
		Parameters
		----------
		parameter_vector: array_like
			The parameter_vector is a list of list, each parameter list 
			is indexed according to the order of initialization 
			specified by the user.
			 
			The parameter follows the format specific to each layer
			
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
		for index in range(len(self._order_of_initialization)):
			# For further use
			layer_name = self._order_of_initialization[index][0]
			
			if(layer_name not in self.__input_layers):
				# Layer present as hidden
				self.__layer_map[layer_name].update_parameters(parameter_vector[index])
					
		# Raise a warning if the user specifies more parameters than required
		if(len(parameter_vector) > self.__number_of_layers):
			warnings.warn("The parameter vector consists of elements greater than required")
			
	# Function to visualize the network
	def visualize(self, file_name, show=False):
		"""
		Visualize the computational graph of the network
		
		Parameters
		----------
		file_name: string
			This specifies the path, where we want to save the file
			
		show(optional): boolean
			This specifies whether we want to view the network or not
			
		Returns
		-------
		None
		
		Raises
		------			
		None
		
		Notes
		-----
		In terms of implementation, this is a simple python 
		automation of graphviz library for generating a 
		network using clusters
		
		We generate 3 clusters: Sensor, Layers and Hardware
		Then provide connections between them
		"""
		ann = Digraph('ANN', filename=file_name + '.gv')
		ann.node_attr['shape'] = 'box'
		ann.graph_attr['rankdir'] = 'LR'
		
		# We are going to use 3 clusters, hardware output, nodes and sensor input
			
		# Sensor Cluster
		with ann.subgraph(name="cluster_2") as cluster:
			cluster.attr(color="black", label="SENSORS")
			cluster.node_attr['style'] = 'filled'
			
			# There are no edges within the sensor cluster
			# Only nodes
			for layer in reversed(self._order_of_initialization):
				if(layer[3] != ""):
					cluster.node(layer[3])		
		
		# Node Cluster
		with ann.subgraph(name="cluster_1") as cluster:
			cluster.attr(color="black", label= self.__type_of_network + " LAYERS")
			cluster.node_attr['style'] = 'filled'
			
			# Add the edges within the node cluster
			for layer in self._order_of_initialization:
				cluster.node(layer[0], label=layer[0])
				for output in layer[4]:
					if(output not in self.__output_layers):
						cluster.edge(layer[0], output)
					
		# Hardware Cluster
		with ann.subgraph(name="cluster_0") as cluster:
			cluster.attr(color='black', label="HARDWARE")
			cluster.node_attr['style'] = 'filled'
			
			# There are no edges within the hardware cluster
			# Only nodes
			for hardware in self.__output_layers:
				cluster.node(hardware)
					
		# Combine everything now
		for layer in reversed(self._order_of_initialization):
			# Check for hardware output
			for output in layer[4]:
				if(output in self.__output_layers):
					ann.edge(layer[0], output)
					
			# Check for sensor input
			if(layer[3] != ""):
				ann.edge(layer[3], layer[0])
		
		# View the network, if show is True		
		if(show is True):
			ann.view()
				
		
	# Getters and Setters
	@property
	def number_of_layers(self):
		""" Attribute for number of layers 
			Denotes the number of layers of the network
		"""
		return self.__number_of_layers
		
	@property
	def type_of_network(self):
		"""Attribute for the type of network
		"""
		return self.__type_of_network
		
	@type_of_network.setter
	def type_of_network(self, network):
		# Some inclusions to ignore spelling mistakes by users
		if(network[0].upper() == "D"):
			self.__type_of_network = "DYNAMIC"
		else:
			self.__type_of_network = "STATIC"
		
	@property
	def order_of_initialization(self):
		""" Attribute for the order of initialization 
			Specifies the order in which layer outputs should be calculated to generate the overall output
		"""
		order = [layer[0] for layer in self._order_of_initialization]
		return order
		
	@property
	def number_of_parameters(self):
		"""The number of trainable parameters of the network
		"""
		
		length = 0
		# Loop through the various layers and add
		# their respective length of parameter vector
		for index in range(len(self._order_of_initialization)):
			layer_name = self._order_of_initialization[index][0]
			if(layer_name not  in self.__input_layers):
				length = length + len(self.__layer_map[layer_name].return_parameters())
				
		return length
		
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
		
		return self.__state
		
	
			
	
				
		
		

