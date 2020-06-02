# This module implements the Static Neural Networks
# The fundamental idea behind neural networks is the concept of layers
# A collection of layers make up a Neural Network
# So, we implement a layer class and then a neural network class that makes use of it!

import numpy as np
import pickle
from graphviz import Digraph, Graph

# Note: The weight matrix works as follows:
# Each of the columns tune for the weights of a single input node
# Each of the rows tune for the weights of a single output node

# Static Layer, only forward connections are present in this layer
class StaticLayer:
	def __init__(self, input_dim, output_dim, activation_function, layer_name):
		# Set the layer name
		self.layer_name = layer_name
	
		# Initialize the weight and bias dimensions
		self.weight_dim = (output_dim, input_dim)
		self.bias_dim = (output_dim, 1)
		
		# Initialize the weight matrix
		self.weight_matrix = np.random.rand(*self.weight_dim)
		
		# Initialize the bias vector
		self.bias_vector = np.random.rand(output_dim)
		
		# Set the activation function
		self.activation_function = activation_function
		
	# Function to calculate the output vector
	# weight_matrix . input_vector + bias_vector
	def forward_propagate(self, input_vector):
		# Convert the input_vector to a numpy array
		#input_vector = np.array(input_vector)
		# The input is already converted to numpy array by the Neural Network
		
		# Output vector is obtained by dotting weight and input, then adding with bias
		output_vector = np.add(np.dot(self.weight_matrix, input_vector), self.bias_vector)
		
		# Activate the output
		output_vector = self.activation_function(output_vector)
		
		return output_vector
	
	# Function to set the weight matrix	
	def set_weight_matrix(self, weight_matrix):
		self.weight_matrix = weight_matrix
		
	# Function to return the weight matrix
	def get_weight_matrix(self):
		return self.weight_matrix
	
	# Function to set the bias vector	
	def set_bias_vector(self, bias_vector):
		self.bias_vector = bias_vector
		
	# Function to return the bias vector
	def get_bias_vector(self):
		return self.bias_vector
		
	# Function to return the layer name
	def get_name(self):
		return self.layer_name
		
	# Function to return the weight dimensions
	def get_weight_dim(self):
		return self.weight_dim
		
	# Function return the bias dimensions
	def get_bias_dim(self):
		return self.bias_dim
		
# Perceptron Class
class Perceptron:
	def __init__(self, input_dim, output_dim, activation_function):
		# Initializations
		self.input_dim = input_dim
		self.output_dim = output_dim
		
		# Contains a single static layer
		# Initialize that Layer
		self.layer = StaticLayer(input_dim, output_dim, activation_function, "Perceptron_Layer")
		
		# Construct a visual component
		# The visual component is constructed using Graphviz
		self.visual = Digraph(comment="Perceptron", graph_attr={'rankdir': "LR", 'splines': "line"}, node_attr={'fixedsize': "true", 'label': ""})
		#self.generate_visual()
		
	# Function to calculate the output given an input vector
	def forward_propagate(self, input_vector):
		# Convert the input_vector to numpy array
		input_vector = np.array(input_vector)
		
		# Get the output
		output_vector = self.layer.forward_propagate(input_vector)
		
		return output_vector
	
	# Function to save the layer weights
	def save_weights_to_file(self, file_name):
		# Use pickle to save the layer_vector
		with open(file_name, 'wb') as f:
			pickle.dump(self.layer, f)
			
	# Function to load the layer weights
	def load_weights_from_file(self, file_name):
		# Use pickle to load the layer_vector
		with open(file_name, 'rb') as f:
			self.layer = pickle.load(f)
			
	# Function to return the weights and bias in the form of a vector
	def return_weights_as_vector(self):
		# The vector we get from flattening the weight matrix
		# flatten() works in row major order
		weight_vector = self.layer.get_weight_matrix().flatten()
		
		# The vector we get from flattening the bias vector
		bias_vector = self.layer.get_bias_vector().flatten()
		
		# The output vector is concatenated form of weight_vector and bias_vector
		output = np.concatenate([weight_vector, bias_vector])
		
		return output
	
	# Function to load the weights and bias from vector
	def load_weights_from_vector(self, weight_vector):
		# Convert to numpy array
		weight_vector = np.array(weight_vector)
	
		# Get the dimensions of the weight matrix and bias vector
		weight_dim = self.layer.get_weight_dim()
		bias_dim = self.layer.get_bias_dim()
		
		# Get the interval at which weight and bias seperate
		interval = weight_dim[0] * weight_dim[1]
		
		# Seperate the weights and bias and then reshape them
		self.layer.set_weight_matrix(weight_vector[:interval].reshape(weight_dim))
		self.layer.set_bias_vector(weight_vector[interval:].reshape(bias_dim[0],))
	
	# Function to generate the visual representation	
	def generate_visual(self, filename, view=False):
		# We need two subgraphs
		subgraph_one = Digraph(name="cluster_0", graph_attr={'color': "white", 'label': "Layer 0"}, node_attr={'style': "solid", 'color': "blue4", 'shape': "circle"})
		for node_number in range(self.input_dim):
			subgraph_one.node("x" + str(node_number+1))
			
		subgraph_two = Digraph(name="cluster_1", graph_attr={'color': 'white', 'label': "Layer 1"}, node_attr={'style': "solid", 'color': "red2", 'shape': "circle"})
		for node_number in range(self.output_dim):
			subgraph_two.node("a" + str(node_number+1))
			
		# Declare subgraphs
		self.visual.subgraph(subgraph_one)
		self.visual.subgraph(subgraph_two)
		
		# Put the edges in the graph
		for input_node in range(self.input_dim):
			for output_node in range(self.output_dim):
				self.visual.edge('x'+str(input_node+1), 'a'+str(output_node+1))
		
		# Render the graph		
		self.visual.render('representations/' + filename + '.gv', view=view)
		
		

# Static Neural Network Class
class StaticNeuralNetwork:
	# The layer_dimensions is an array with the following layout
	# [[number_of_nodes_in_first_layer(input layer), activation_function], [number_of_nodes_in_second_layer, activation_function], ..., [number_of_output_nodes]]
	def __init__(self, layer_dimensions):
		# Initialize a layer_vector, that is a list of Layer objects
		self.layer_vector = []
		
		# Append the Layer classes
		for i in range(len(layer_dimensions) - 1):
			self.layer_vector.append(StaticLayer(layer_dimensions[i][0], layer_dimensions[i+1][0], layer_dimensions[i][1], "Layer_"+str(i)))
			
		# Number of layers
		self.number_of_layers = len(self.layer_vector)
		# Construct a visual component
		self.visual = Digraph(comment="Static Neural Network", graph_attr={'rankdir': "LR", 'splines': "line"}, node_attr={'fixedsize': "true", 'label': ""})
		# self.generate_visual(len(self.layer_vector))
		
	# Function to get output from input_vector
	def forward_propagate(self, input_vector):
		# Convert the input_vector to numpy array
		intermediate_output = np.array(input_vector)
		
		# Forward propagate for each layer
		for layer in self.layer_vector:
			intermediate_output = layer.forward_propagate(intermediate_output)
			
		return intermediate_output
		
	# Function to save the layer weights
	def save_weights(self, file_name):
		# Use pickle to save the layer_vector
		with open(file_name, 'wb') as f:
			pickle.dump(self.layer_vector, f)
			
	# Function to load the layer weights
	def load_weights(self, file_name):
		# Use pickle to load the layer_vector
		with open(file_name, 'rb') as f:
			self.layer_vector = pickle.load(f)
			
	# Function to return the weights and bias in the form of a vector
	def return_weights_as_vector(self):
		# Initialize the output vector
		# Determine an individual layer's weight matrix in row major form and then it's bias
		# Then concatenate it with the previous output vector
		output = np.array([])
	
		for layer in self.layer_vector:
			# The vector we get from flattening the weight matrix
			# flatten() works in row major order
			weight_vector = layer.get_weight_matrix().flatten()
			
			# The vector we get from flattening the bias vector
			bias_vector = layer.get_bias_vector().flatten()
			
			# The output vector is concatenated form of weight_vector and bias_vector
			output = np.concatenate([output, weight_vector, bias_vector])
		
		return output
	
	# Function to load the weights and bias from vector
	def load_weights_from_vector(self, weight_vector):
		# Convert to numpy array
		weight_vector = np.array(weight_vector)
	
		# Interval counter maintains the current layer index
		interval_counter = 0
		
		for layer in self.layer_vector:
			# Get the dimensions of the weight matrix and bias vector
			weight_dim = layer.get_weight_dim()
			bias_dim = layer.get_bias_dim()
			
			# Get the interval at which weight and bias seperate
			weight_interval = weight_dim[0] * weight_dim[1]
			
			# Get the interval at which the bias and next weight vector seperate
			bias_interval = bias_dim[0] * bias_dim[1]
			
			# Seperate the weights and bias and then reshape them
			layer.set_weight_matrix(weight_vector[interval_counter:interval_counter + weight_interval].reshape(weight_dim))
			interval_counter = interval_counter + weight_interval
			layer.set_bias_vector(weight_vector[interval_counter:interval_counter + bias_interval].reshape(bias_dim[0],))
			interval_counter = interval_counter + bias_interval
			
	
	# Function to generate the visual representation
	def generate_visual(self, filename, view=False):
		# We need many subgraphs
		for layer in range(self.number_of_layers):
			subgraph = Digraph(name="cluster_" + str(layer), graph_attr={'color': "white", 'label': "Layer " + str(layer)}, node_attr={'style': "solid", 'color': "black", 'shape': "circle"})
			
			# Get the weight dimensions for generating the nodes
			weight_dim = self.layer_vector[layer].get_weight_dim()
			
			for node_number in range(weight_dim[1]):
				subgraph.node("layer_" + str(layer) + str(node_number+1))
				
			# Declare subgraphs
			self.visual.subgraph(subgraph)
			
			
		# The final layer needs to be done manually
		subgraph = Digraph(name="cluster_" + str(self.number_of_layers), graph_attr={'color': "white", 'label': "Layer " + str(self.number_of_layers)}, node_attr={'style': "solid", 'color': "black", 'shape': "circle"})
		
		# Get the weight dimensions
		weight_dim = self.layer_vector[self.number_of_layers - 1].get_weight_dim()
		
		for node_number in range(weight_dim[0]):
			subgraph.node("layer_" + str(self.number_of_layers) + str(node_number+1))
			
		# Declare the subgraph
		self.visual.subgraph(subgraph)
		
		
		for layer in range(self.number_of_layers):
			# Get the weight dimensions for generating the nodes
			weight_dim = self.layer_vector[layer].get_weight_dim()
		
			# Put the edges in the graph
			for input_node in range(weight_dim[1]):
				for output_node in range(weight_dim[0]):
					self.visual.edge("layer_" + str(layer) + str(input_node+1), 'layer_' + str(layer + 1) + str(output_node+1))
		
		# Render the graph		
		self.visual.render('representations/' + filename + '.gv', view=view)	
		
		
	
		
		
		
	
		
		
