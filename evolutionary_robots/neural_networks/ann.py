"""Docstring for the ann.py module

This module implements Artificial Neural Networks
The Network can have variable number of layers,
variable number of neurons in each layer, variable
activation function and variable type of layer(Simple or Continuous)

"""

import numpy as np
import tensorflow as tf
import pickle
from graphviz import Digraph
from layers import InputLayer, SimpleLayer, CTRLayer

# Library used to genrate warnings
import warnings

class ArtificialNeuralNetwork:
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
		
	Methods
	-------
	
	"""
	
	def __init__(self, layer_vector):
		# Class declarations
		self.number_of_layers = len(layer_vector)
		
		# Private declarations
		self.__input_layers = []
		self.__output_layers = []
		self.__graph = tf.Graph()
		
		# Construct the layers and the graph
		self._construct_layers(layer_vector)
		#self._construct_graph()
		
		tf.compat.v1.enable_eager_execution()
			
	
	# Function to construct the layers given the inputs
	# in essence, a Neural Network
	def _construct_layers(self, layer_vector):
		# A dictionary for storing each layer
		self.layer_map = {}
		
		# List for input output connections
		self.input_connections = []
		self.output_connections = []
		
		# Append the Layer classes
		for index in range(self.number_of_layers):
			# 0 => input layer
			if(layer_vector[index][1] == 0):
				# Input Layer
				self.__input_layers.append("layer_" + str(index))
			
				# No need to collect the input dimensions as input layer doesn't take input from other layers
				input_dimension = layer_vector[index][0]
				self.input_connections.append([])
				
				# Collect the output dimensions
				connect = []
				for connection in layer_vector[index][4]:
					connect.append("layer_" + str(connection))
					
				self.output_connections.append(connect)
				
				# Generate the layer
				self.layer_map["layer_" + str(index)] = InputLayer(input_dimension, "layer_"+str(index))
				
			# 1 => Simple Layer
			elif(layer_vector[index][1] == 1):
				# Collect the input dimensions
				input_dimension = 0
				connect = []
				for connection in layer_vector[index][3]:
					# Taking account the delay as well
					 connect.append(("layer_" + str(connection[0]), connection[1]))
					 input_dimension = input_dimension + layer_vector[connection[0]][0]
					 
				self.input_connections.append(connect)
				
				# Collect the output dimensions
				output_dimension = layer_vector[index][0]
				connect = []
				for connection in layer_vector[index][4]:
					connect.append("layer_" + str(connection))
					
				# Determine the output layer
				if(len(connect) == 0):
					self.__output_layers.append("layer_" + str(index))
						
				self.output_connections.append(connect)	
				
				# Collect the activation function
				activation_function = layer_vector[index][2]
				
				# Generate the layer
				self.layer_map["layer_" + str(index)] = SimpleLayer(input_dimension, output_dimension, activation_function, "layer_"+str(index))
				
			# 2 => CTR Layer
			elif(layer_vector[index][1] == 2):
				# Collect the input dimensions
				input_dimension = 0
				connect = []
				for connection in layer_vector[index][3]:
					# Taking account the delay as well
					connect.append(("layer_" + str(connection[0]), connection[1]))
					
				self.input_connections.append(connect)
				
				# Collect the output dimensions
				output_dimension = layer_vector[index][0]
				connect = []
				for connection in layer_vector[index][4]:
					connect.append("layer_" + str(connection))
					
				# Determine the output layer
				if(len(connect) == 0):
					self.__output_layers.append("layer_" + str(index))
					
				self.output_connections.append(connect)
					
				# Collect the activaiton function
				activation_function = layer_vector[index][2]
				
				# Generate the layer
				self.layer_map["layer_" + str(index)] = CTRLayer(input_dimension, output_dimension, activation_function, "layer_"+str(index))


	# The function to calculate the output
	def forward_propagate(self, input_dict):
		# Output matrix for computing through the computation graph
		self.input_matrix = [tf.Variable(np.array([0.0]))] * self.number_of_layers
		self.sensor_matrix = [tf.Variable(np.array([0.0]))] * self.number_of_layers
		self.output_matrix = [tf.Variable(np.array([0.0]))] * self.number_of_layers
		
		# Generate the computation graph
		for index in range(self.number_of_layers):
			# If the layer is part of input
			if "layer_" + str(index) in self.__input_layers:
				self.input_matrix[index].assign(input_dict[index], validate_shape=False)
				self.output_matrix[index].assign(tf.numpy_function(self.layer_map["layer_" + str(index)].forward_propagate, [self.input_matrix[index]], tf.float64))
				
			# If the layer is not part of the input
			else:
				# Sensor input for associative layer
				# In general it will be zero, but can take other value if specified
				self.sensor_matrix[index].assign(input_dict[index])
				
				input_vector = []
				# Collect the input vector
				for layer in self.input_connections:
					for connection in layer:
						# determine connection index
						connection_index = int(connection[0][6:])
						# No delay
						if(connection[1] == False):
							input_vector.append(self.output_matrix[connection_index][0])
						# Delay
						else:
							input_vector.append(self.output_matrix[connection_index][1])
						
				# Concatenate all the vectors
				self.input_matrix[index].assign(tf.concat(input_vector, axis = 0))
				
				# Calculate the activation corresponding to the concatenated input vector and sensor input
				self.output_matrix[index].assign(tf.numpy_function(self.layer_map["layer_" + str(index)].forward_propagate, [self.input_matrix[index], self.sensor_matrix[index]], tf.float64))

				


	# Function to save the layer parameters
	def save_parameters_to_file(self, file_name):
		"""
		Save the parameters of the Neural Network
		
		Using pickle, the list of layers is stored
		"""
		# First generate that list
		layer_vector = [self.input_layers, self.hidden_layers, self.output_layers]
		
		# Use pickle to save the layer_vector
		with open(file_name, 'wb') as f:
			pickle.dump(layer_vector, f)
			
	# Function to load the layer parameters
	def load_weights_from_file(self, file_name):
		""" Load the parameters of the Neural Network """
		# Use pickle to load the layer_vector
		with open(file_name, 'rb') as f:
			layer_vector = pickle.load(f)
			
			# Load the list parameters
			self.input_layers = layer_vector[0]
			self.hidden_layers = layer_vector[1]
			self.output_layers = layer_vector[2]
	
	# Function to return the parameters in the form of a vector		
	def return_parameters_as_vector(self):
		# Initialize the output vector
		# Determine an individual layer's weight matrix in row major form and then it's bias
		# Then concatenate it with the previous output vector
		output = []
	
		for index in range(self.number_of_layers):
			if(self.layer_category[index] == "Input"):
				# Layer present as input
				output.append(np.array([]))
				
			elif(self.layer_category[index] == "Hidden"):
				# Layer present as hidden
				output.append(np.array(self.hidden_layers[index].return_parameters()))
		
			elif(self.layer_category[index] == "Output"):
				# Layer present as output
				output.append(np.array(self.output_layers[index].return_parameters()))
		
		return output
		
	# Function to load the parameters from a vector
	def load_parameters_from_vector(self, parameter_vector):
		# Convert to numpy array
		parameter_vector = np.array(parameter_vector)
		
		# Load the parameters layer by layer
		for index in range(self.number_of_layers):
			if(self.layer_category[index] == "Hidden"):
				# Layer present as hidden
				self.hidden_layers[index].update_parameters(parameter_vector[index])
				
			elif(self.layer_category[index] == "Output"):
				# Layer present as output
				self.output_layers[index].update_parameters(parameter_vector[index])
					
					
	def generate_visual(self, filename, view=False):
		pass
		
		
	
			
	
				
		
		

