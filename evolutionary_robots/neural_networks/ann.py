"""Docstring for the ann.py module

This module implements Artificial Neural Networks
The Network can have variable number of layers,
variable number of neurons in each layer, variable
activation function and variable type of layer(Simple or Continuous)

"""

import numpy as np
import pickle
from graphviz import Digraph
from layers import InputLayer, StaticLayer, CTRNNLayer, RBFLayer

# Library used to genrate warnings
import warnings

class ArtificialNeuralNetwork:
	"""
	The Artificial Neural Network Class
	
	...
	
	Attributes
	----------
	number_of_layers: integer
		The number of layers in the Neural Network
		
	type_of_layer: array_like
		An array specifying the type of layer, Input, Static, CTR or RBF
		0 => Input, 1 => Static, 2 => CTR and 3 => RBF
		
	number_of_neurons: array_like
		An array of neurons in each layer
		
	activation_function: array_like
		An array of activation function objects
		For the input layers, it is ignored
		
	adjacency_matrix: array_like
		Adjacency Matrix to specify the connections between layers
		1 for [i, j] implies that ith layer gives output to jth layer
		
	Methods
	-------
	
	"""
	
	def __init__(self, number_of_layers, type_of_layer, number_of_neurons, activation_function, adjacency_matrix):
		# Class declarations
		self.number_of_layers = number_of_layers
		
		# Convert to Adjacency List
		self.matrix_to_list(adjacency_matrix)
		
		# Construct the layers
		self.construct_layers(type_of_layer, number_of_neurons, activation_function)
	
	# Function to generate an adjacency list from
	# an adjacency matrix
	def matrix_to_list(self, adjacency_matrix):
		"""
		Internal function to convert from adjacency matrix to input output lists
		"""
		adjacency_matrix = np.array(adjacency_matrix)
		
		# Output connections: For index i which layers take input from i
		# Input connections: For index i which layer give input to i
		self.input_connections = []
		self.output_connections = []
		
		# Append to output connections when a 1 occurs
		for node in range(len(adjacency_matrix)):
			# Generate a list wherever 1 occurs and then append to list for that layer
			list_of_indices = [i for i, x in enumerate(adjacency_matrix[node]) if x == 1]
			self.output_connections.append(list_of_indices)
				
		# Append to input connections when a 1 occurs in the transpose
		adjacency_matrix = np.transpose(adjacency_matrix)
		for node in range(len(adjacency_matrix)):
			# Generate a list wherever 1 occurs and then append to list for that layer
			list_of_indices = [i for i, x in enumerate(adjacency_matrix[node]) if x == 1]
			self.input_connections.append(list_of_indices)
			
	
	# Function to construct the layers given the inputs
	# in essence, a Neural Network
	def construct_layers(self, type_of_layer, number_of_neurons, activation_function):
		# Initialize 3 layer dictionaries (Currently 3, we can move to 4 in the near future)
		self.input_layers = {}
		self.hidden_layers = {}
		self.output_layers = {}
		
		# A dictionary for storing which layer is Input, Hidden or Output
		self.layer_category = {}
		
		# Append the Layer classes
		# Breadth First Search based Algorithm
		for index in range(self.number_of_layers):
			# All the calculation is based on the adjacency list and type of layer
			# 0 implies Input Layer
			if(type_of_layer[index] == 0):
				self.input_layers[index] = InputLayer(number_of_neurons[index], "Input Layer")
				self.layer_category[index] = "Input"
				
			# 1 implies a Static Layer
			elif(type_of_layer[index] == 1):
				# Combined input dimension of a layer
				# Dimensions of the layers that are giving input to the current layer
				input_dimensions = [number_of_neurons[i] for i in self.input_connections[index]]
				combined_input_dim = sum(input_dimensions)
				
				# Output or hidden, output layer does not have any elements in it's output_connections
				if(len(self.output_connections[index]) == 0):
					self.output_layers[index] = StaticLayer(combined_input_dim, number_of_neurons[index], activation_function[index], "Output Layer")
					self.layer_category[index] = "Output"
					
				else:
					self.hidden_layers[index] = StaticLayer(combined_input_dim, number_of_neurons[index], activation_function[index], "Hidden Layer")
					self.layer_category[index] = "Hidden"
				
			# 2 implies a CTR Layer
			elif(type_of_layer[index] == 2):
				# Combined input dimension of a layer
				# Dimensions of the layers that are giving input to the current layer
				input_dimensions = [number_of_neurons[i] for i in self.input_connections[index]]
				combined_input_dim = sum(input_dimensions)
				
				# Output or hidden, output layer does not have any elements in it's output_connections
				if(len(self.output_connections[index]) == 0):
					self.output_layers[index] = CTRNNLayer(combined_input_dim, number_of_neurons[index], 0.001, np.fill((number_of_layers[index], ), 0.001),activation_function[index], "Output Layer")
					self.layer_category[index] = "Output"
					
				else:
					self.hidden_layers[index] = CTRNNLayer(combined_input_dim, number_of_neurons[index], 0.001, np.fill((number_of_layers[index], ), 0.001),activation_function[index], "Hidden Layer")
					self.layer_category[index] = "Hidden"
				
				
			# 2 implies an RBF Layer, but not now!
	
	# Function to get output from input_vector
	# Input vector is a map that tells which indices are given the inputs	
	def forward_propagate(self, input_vector):
		# Output Dictionary to store the outputs of the layer
		# They can then be used criss-cross
		output_dict = {}
		
		# Input Layers
		for index, layer in self.input_layers.items():
			output_dict[index] = np.array(input_vector[index])
			
		# Hidden Layers
		for index, layer in self.hidden_layers.items():
			# Combine the inputs
			combined_input = np.array([])
			for input_index in self.input_connections[index]:
				combined_input = np.concatenate([combined_input, output_dict[input_index]])
			
			output_dict[index] = layer.forward_propagate(np.array(combined_input))
		
		
		# Return Dictionary for storing the final output of neural network
		return_dict = {}
			
		# Output Layers
		for index, layer in self.output_layers.items():
			# Combine the inputs
			combined_input = np.array([])
			for input_index in self.input_connections[index]:
				combined_input = np.concatenate([combined_input, output_dict[input_index]])
				
			return_dict[index] = layer.forward_propagate(np.array(combined_input))
		
		return return_dict
		
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
		
		
	
			
	
				
		
		

