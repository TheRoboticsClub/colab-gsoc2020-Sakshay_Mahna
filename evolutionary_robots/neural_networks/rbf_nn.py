# This module implements the Radial Basis Function Network
# The network requires a different kind of layer
# to take input as a vector, compute the radial basis function
# output of that input vector and then pass it to a Static Layer
# to give the output

import numpy as np
import pickle
from graphviz import Digraph
from static_nn import StaticLayer
from activation_functions import linear_function
from nn import Layer, NeuralNetwork

# Radial Basis Function Layer
class RBFLayer(Layer):
	def __init__(self, input_dim, output_dim, distance_function, basis_function, parameter, layer_name):
		# Set the name
		self.layer_name = layer_name
	
		# Set the dimensions
		self.center_dim = (output_dim, input_dim)
		
		# Set the center of neurons
		self.center_matrix = np.random.rand(*self.center_dim)
		
		# Set the distance function
		self.distance_function = distance_function
		
		# Set the basis function and it's parameter
		self.basis_function = basis_function
		self.parameter = parameter
		
	# Forward Propagate
	def forward_propagate(self, input_vector):
		# Conver to numpy array
		input_vector = np.array(input_vector)
		
		# Compute the distance of the input and the centers
		# The distance function has to calculate a distance vector
		# based on the center matrix and input_vector
		distance_vector = self.distance_function(input_vector, self.center_matrix)
		
		# Calculate the output of the basis function
		output = self.basis_function(distance_vector, self.parameter)
		
		return output
		
	# Function to set the center matrix
	def set_center_matrix(self, center_matrix):
		self.center_matrix = center_matrix
		
	# Function to get the center matrix
	def get_center_matrix(self):
		return self.center_matrix
		
	# Function to return the dimensions of the center matrix
	def get_center_dim(self):
		return self.center_dim
		
	# Function to return the layer name
	def get_name(self):
		return self.layer_name
		
# Gaussian Radial Basis Network
class GaussianRBFNetwork(NeuralNetwork):
	def __init__(self, input_dim, hidden_dim, output_dim, beta=1):
		# Make the dimensions available
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
	
		# A combination of RBF Layer and StaticLayer
		self.rbf_layer = RBFLayer(input_dim, hidden_dim, self.distance_function, self.basis_function, beta, "RBF Layer")
		self.static_layer = StaticLayer(hidden_dim, output_dim, linear_function, "Static Layer")
		
		# The bias of the perceptron is zero
		self.static_layer.set_bias_vector(np.zeros(output_dim))
		
		# Set the visual
		self.visual = Digraph(comment="RBF Network", graph_attr={'rankdir': "LR", 'splines': "line"}, node_attr={'fixedsize': "true", 'label': ""})
		
	# Forward Propagation
	def forward_propagate(self, input_vector):
		# Convert to numpy
		input_vector = np.array(input_vector)
		
		# Calculate Outputs
		intermediate_output = self.rbf_layer.forward_propagate(input_vector)
		intermediate_output = self.static_layer.forward_propagate(intermediate_output)
		
		return intermediate_output
		
	# The distance function of the network
	def distance_function(self, input_vector, center_matrix):
		# Convert to numpy
		input_vector = np.array(input_vector)
		center_matrix = np.array(center_matrix)
		
		# Take the difference
		# Numpy handles it automatically
		difference = center_matrix - input_vector
		
		# Calculate the euclidean distance
		square = difference * difference
		euclidean_distance = np.sqrt(np.sum(square, axis=1))
		
		return euclidean_distance
		
	# The basis function of the network
	def basis_function(self, input_vector, beta):
		# Convert to numpy
		input_vector = np.array(input_vector)
		input_vector = beta * input_vector
		
		# Gaussian function
		square = input_vector * input_vector
		output = np.exp(-1 * square)
		
		return output
		
	# Function to save the layer weights
	def save_weights_to_file(self, file_name):
		# Use pickle to save the layer_vector
		with open(file_name, 'wb') as f:
			pickle.dump({'rbf': self.rbf_layer, 'static': self.static_layer}, f)
			
	# Function to load the layer weights
	def load_weights_from_file(self, file_name):
		# Use pickle to load the layer_vector
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
			self.rbf_layer = data['rbf']
			self.perceptron_layer = data['static']
			
	# Function to return the centers and weights in the form of a vector
	def return_weights_as_vector(self):
		# The vector we get from flattening the center matrix
		# flatten() works in row major order
		center_vector = self.rbf_layer.get_center_matrix().flatten()
		
		# The weight vector
		weight_vector = self.static.get_weight_matrix().flatten()
		
		# The output vector is concatenated form of center_vector and weight_vector
		output = np.concatenate([center_vector, weight_vector])
		
		return output
		
	# Function to load the weights and centers from vector
	def load_weights_from_vector(self, weight_vector):
		# Convert to numpy array
		weight_vector = np.array(weight_vector)
		
		# Get the dimensions of the center and weight matrix
		center_dim = self.rbf_layer.get_center_dim()
		weight_dim = self.static_layer.get_weight_dim()
		
		
		# Get the interval at which the center and weight seperate
		interval = center_dim[0] * center_dim[1]
		
		# Seperate the center matrix and weight matrix
		self.rbf_layer.set_center_matrix(weight_vector[:interval].reshape(center_dim))
		self.static_layer.set_weight_matrix(weight_vector[interval:].reshape(weight_dim))
		
	def generate_visual(self, filename, view=False):
		# We need three subgraphs
		subgraph_one = Digraph(name="cluster_0", graph_attr={'color': "white", 'label': "Input Layer"}, node_attr={'style': "solid", 'color': "blue4", 'shape': "circle"})
		for node_number in range(self.input_dim):
			subgraph_one.node("x" + str(node_number+1))
			
		subgraph_two = Digraph(name="cluster_1", graph_attr={'color': "white", 'label': "RBF Layer"}, node_attr={'style': "solid", 'color': "green", 'shape': "circle"})
		for node_number in range(self.hidden_dim):
			subgraph_two.node("r" + str(node_number+1))
			
		subgraph_three = Digraph(name="cluster_2", graph_attr={'color': 'white', 'label': "Static Layer"}, node_attr={'style': "solid", 'color': "red2", 'shape': "circle"})
		for node_number in range(self.output_dim):
			subgraph_three.node("y" + str(node_number+1))
			
		# Declare subgraphs
		self.visual.subgraph(subgraph_one)
		self.visual.subgraph(subgraph_two)
		self.visual.subgraph(subgraph_three)
		
		# Put the edges in the graph
		for input_node in range(self.input_dim):
			for hidden_node in range(self.hidden_dim):
				self.visual.edge('x'+str(input_node+1), 'r'+str(hidden_node+1))
				
		for hidden_node in range(self.hidden_dim):
			for output_node in range(self.output_dim):
				self.visual.edge('r'+str(hidden_node+1), 'y'+str(output_node+1))
		
		# Render the graph		
		self.visual.render('representations/' + filename + '.gv', view=view)
		
	
		
		
		

