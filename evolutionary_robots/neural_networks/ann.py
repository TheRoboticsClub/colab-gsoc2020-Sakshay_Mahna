"""Docstring for the ann.py module

This module implements Artificial Neural Networks
The Network can have variable number of layers,
variable number of neurons in each layer, variable
activation function and variable type of layer(Simple or Continuous)

"""

import numpy as np
import pickle
from graphviz import Digraph
from layers import StaticLayer, CTRNNLayer, RBFLayer

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
		An array specifying the type of layer, Static, CTR or RBF
		
	number_of_neurons: array_like
		An array of neurons in each layer
		
	activation_function: array_like
		An array of activation function objects
		
	adjacency_matrix: array_like
		Adjacency Matrix to specify the connections between layers
		
	name_of_layer: array_like
		An array specifying whether the layer is input, hidden, associative or output
		
	Methods
	-------
	
	"""
	
	def __init__(self, number_of_layers, type_of_layer, number_of_neurons, activation_function, adjacency_matrix, name_of_layer):
		# Class declarations
		self.number_of_layers = number_of_layers
		
		# Convert to Adjacency List
		self.matrix_to_list(adjacency_matrix)
		
		# Construct the layers
		self.construct_layers(type_of_layer, number_of_neurons, activation_function, name_of_layer)
	
	# Function to generate an adjacency list from
	# an adjacency matrix
	def matrix_to_list(self, adjacency_matrix):
		self.adjacency_list = []
		
		# Append to adjacency list when a 1 occurs
		for node in adjacency_matrix:
			try:
				self.adjacency_list.append(node.index(1))
			except:
				self.adjacency_list.append(0)
			
	
	# Function to construct the layers given the inputs
	# in essence, a Neural Network
	def construct_layers(self, type_of_layer, number_of_neurons, activation_function, name_of_layer):
		# Initialize a layer vector, that is a list of Layer objects
		self.layer_vector = []
		
		# Append the Layer classes
		for index in range(self.number_of_layers - 1):
			# 0 implies a Static Layer
			if(type_of_layer[index] == 0):
				self.layer_vector.append(StaticLayer(number_of_neurons[index], self.adjacency_list[index], activation_function[index], str(name_of_layer)))
				
			# 1 implies a CTR Layer
			elif(type_of_layer[index] == 1):
				self.layer_vector.append(CTRNNLayer(number_of_neurons[index], self.adjacency_list[index], 0.001, np.fill((number_of_layers[index], ), 0.001),activation_function[index], str(name_of_layer)))
				
			# 2 implies an RBF Layer, but not now!
	
	# Function to get output from input_vector		
	def forward_propagate(self, input_vector):
		# Convert to numpy array
		intermediate_output = np.array(input_vector)
		
		# Forward Propagate the outputs
		for layer in self.layer_vector:
			# Static Layer?
			if(isinstance(layer, StaticLayer)):
				intermediate_output = layer.forward_propagate(intermediate_output)
				
			# CTR Layer?
			elif(isinstance(layer, CTRNNLayer)):
				intermediate_output = layer.euler_step(intermediate_output)
				
		return intermediate_output
		
	# Function to save the layer parameters
	def save_parameters_to_file(self, file_name):
		"""
		Save the parameters of the Neural Network
		
		Using pickle, the list of layers is stored
		"""
		# Use pickle to save the layer_vector
		with open(file_name, 'wb') as f:
			pickle.dump(self.layer_vector, f)
			
	# Function to load the layer parameters
	def load_weights_from_file(self, file_name):
		""" Load the parameters of the Neural Network """
		# Use pickle to load the layer_vector
		with open(file_name, 'rb') as f:
			self.layer_vector = pickle.load(f)
	
	# Function to return the parameters in the form of a vector		
	def return_parameters_as_vector(self):
		# Initialize the output vector
		# Determine an individual layer's weight matrix in row major form and then it's bias
		# Then concatenate it with the previous output vector
		output = np.array([])
	
		for layer in self.layer_vector:
			output = np.concatenate(output, [layer.return_parameters()])
		
		return output
		
	# Function to load the parameters from a vector
	def load_parameters_from_vector(self, parameter_vector):
		# Convert to numpy array
		parameter_vector = np.array(parameter_vector)
		
		# Load the parameters layer by layer
		for layer_index in range(len(self.layer_vector)):
			self.layer_vector[layer_index].update_parameters(parameter_vector[layer_index])
					
					
	def generate_visual(self, filename, view=False):
		pass
		
		
	
			
	
				
		
		

