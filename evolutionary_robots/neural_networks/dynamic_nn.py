# This module contains the Dynamic Neural Network class
# Two main constituents of the Dynamic Layer are Time Delay and Time Recurrence
# Time Delay, provides the current input as well as the previous input to the network
# Time Recurrence, provides the output of a layer to the input of a previous layer

# Note: The dynamic Neural Network implemented here only works for left-to-right case
# For a network with right_to_left connections, the user is recommended to construct
# a custom neural network without any abstractions!

import numpy as np
import pickle

# Class for Time Delay system
# The time delay network works by returning the weighted average of the input vectors
# delay_dim specifies the delay, a value of 1 gives the same Static behaviour
class TimeDelay:
	def __init__(self, input_dim, delay_dim):
		# Weight dimensions and input dimensions
		self.weight_dim = (1, delay_dim)
		self.input_matrix_dim = (delay_dim, input_dim)
		
		# Generate the weight vector and input matrix
		self.weight_vector = np.random.rand(*self.weight_dim)
		self.input_matrix = np.zeros(self.input_matrix_dim)
		
	# Forward propagate
	def forward_propagate(self, input_vector):
		# Convert to numpy array
		input_vector = np.array([input_vector])
		
		# Insert the input_vector and remove the oldest one
		self.input_matrix = np.append(self.input_matrix, input_vector, axis=0)
		self.input_matrix = np.delete(self.input_matrix, 0, axis=0)
		
		# Generate the output
		output = np.dot(self.weight_vector, self.input_matrix).flatten()
		
		return output
		
	# Function to return the weight vector
	def get_weight_vector(self):
		return self.weight_vector
		
	# Function to set the weight vector
	def set_weight_vector(self, weight_vector):
		self.weight_vector = weight_vector
		
	# Function to return the dimensions of weight vector
	def get_weight_vector_dim(self):
		return self.weight_dim
		

# Class for Time Recurrence system
# The time recurrence works the same as static layer, but we have to set the input vector by ourselves
# The input_dim are the dimensions of the later layer, which provides its output(right)
# The output_dim are the dimensions of the layer that uses those values(left)
class TimeRecurrence:
	def __init__(self, input_dim, output_dim):
		# Weight dimensions
		self.weight_dim = (output_dim, input_dim)
		
		# Initialize the weight matrix
		self.weight_matrix = np.random.rand(*self.weight_dim)
		
		# Initialize the input vector
		self.input_vector = np.zeros((input_dim,))
		
	# Forward propagate
	def forward_propagate(self):
		# Generate the output and return it
		output = np.dot(self.weight_matrix, self.input_vector)
		return output
		
	# Function to get the weight matrix
	def get_weight_matrix(self):
		return self.weight_matrix
		
	# Function to set the weight matrix
	def set_weight_matrix(self, weight_matrix):
		self.weight_matrix = weight_matrix
		
	# Function to set the input_vector
	def set_input_vector(self, input_vector):
		self.input_vector = input_vector
		
	# Function to return the dimensions of weight matrix
	def get_weight_matrix_dim(self):
		return self.weight_dim
		
# Class for Dynamic Layer
# Each dynamic layer has the provision to include the delay system, recurrent system and forward propagation system
class DynamicLayer:
	# input_dim and output_dim are for the forward propagation part
	# delay_dim is for the delay system
	# recurrent_dim is a list for the recurrent system, the dimensions are given from left to right
	def __init__(self, input_dim, output_dim, delay_dim, recurrent_dim, activation_function, layer_name):
		# Set the layer name
		self.layer_name = layer_name
		
		# Initialize the weight and bias dimensions
		self.weight_dim = (output_dim, input_dim)
		self.bias_dim = (output_dim, 1)
		
		# Initialize the weight and bias
		self.weight_matrix = np.random.rand(*self.weight_dim)
		self.bias_vector = np.random.rand(output_dim)
		
		# Initialize the delay system
		self.delay_system = TimeDelay(input_dim, delay_dim)
		
		# Initialize the recurrent system
		self.recurrent_system = []
		for dimension in recurrent_dim:
			self.recurrent_system.append(TimeRecurrence(dimension, output_dim))
			
		# Set the activation function
		self.activation_function = activation_function
			
	# Forward Propagate
	def forward_propagate(self, input_vector):
		# Convert to numpy array
		input_vector = np.array(input_vector)
		
		# Get the input from delay system
		input_vector = self.delay_system.forward_propagate(input_vector)
		
		# intermediate_output keeps storing the outputs and adding them
		intermediate_output = np.add(np.dot(self.weight_matrix, input_vector), self.bias_vector)
		
		# Get the outputs from recurrent system
		for recurrence in self.recurrent_system:
			intermediate_output = np.add(intermediate_output, recurrence.forward_propagate())
			
		# Activate the output
		intermediate_output = self.activation_function(intermediate_output)
		
		return intermediate_output
		
	# Function to set the input_vector of the recurrent layer
	def set_recurrent_input(self, input_vector, index):
		self.recurrent_system[index].set_input_vector(input_vector)
		
	# Function to set the weight matrix
	def set_weight_matrix(self, weight_matrix):
		self.weight_matrix = weight_matrix
		
	# Function to return the weight matrix
	def get_weight_matrix(self):
		return self.weight_matrix
		
	# Function to set the recurrent weight matrix
	def set_recurrent_weight_matrix(self, weight_matrix, index):
		self.recurrent_system[index].set_weight_matrix(weight_matrix)
		
	# Function to get the recurrent weight matrix
	def get_recurrent_weight_matrix(self, index):
		return self.recurrent_system[index].get_weight_matrix()
		
	# Function to set the delay weight
	def set_delay_weight_vector(self, weight_vector):
		self.delay_system.set_weight_vector(weight_vector)
		
	# Function to return the delay weight
	def get_delay_weight_vector(self):
		return self.delay_system.get_weight_vector()
		
	# Function to set the bias vector
	def set_bias_vector(self, bias_vector):
		self.bias_vector = bias_vector
		
	# Function to return the bias vector
	def get_bias_vector(self):
		return self.bias_vector
		
	# Function to return the delay weight dimensions
	def get_delay_weight_dim(self):
		return self.delay_system.get_weight_vector_dim()
		
	# Function to return the recurrent weight matrix dimensions
	def get_recurrent_weight_dim(self, index):
		return self.recurrent_system[index].get_weight_matrix_dim()
		
	# Function to return the static weight matrix dimensions
	def get_weight_matrix_dim(self):
		return self.weight_dim
		
	# Function to return the bias vector dimensions
	def get_bias_vector_dim(self):
		return self.bias_dim
		
# The Dynamic Neural Network class
class DynamicNeuralNetwork:
	# Layer dimensions follow the layout as:
	# [[nodes_in_layer_one, delay_dim, [list_of_connections], activation_function], [nodes_in_layer_two, delay_dim, [list_of_connections], activation_function], ...[nodes_in_output]]
	# The list_of_connections is provided in a left to right fashion
	def __init__(self, layer_dimensions):
		# Number of layers
		self.number_of_layers = len(layer_dimensions) - 1
		
		# Initialize a layer vector
		self.layer_vector = []
		
		# Parse the list of connections
		self.process_connections(layer_dimensions)
		
		# Generate a list of layers
		for i in range(self.number_of_layers):
			# For the generation of recurrent_dim list
			recurrent_dim = []
			for connection in self.input_connections[i]:
				# Append the dimensions of the output of the layer that is mentioned in the input_connections
				# The user specifies the layer whose output is required
				recurrent_dim.append(layer_dimensions[connection + 1][0])
			self.layer_vector.append(DynamicLayer(layer_dimensions[i][0], layer_dimensions[i+1][0], layer_dimensions[i][1], recurrent_dim, layer_dimensions[i][3], "Layer_"+str(i)))
			
	# Function to forward propagate the output
	def forward_propagate(self, input_vector):
		# A list of intermediate outputs will enable us to update the input_vector for each layer after calculation of the whole output
		self.intermediate_output = [np.array([])] * self.number_of_layers
		
		# The first output is calculated without the loop to start with the intermediate output list
		self.intermediate_output[0] = self.layer_vector[0].forward_propagate(input_vector)
		
		# The loop to generate the intermediate output list
		for layer in range(1, self.number_of_layers):
			self.intermediate_output[layer] = self.layer_vector[layer].forward_propagate(self.intermediate_output[layer-1])
			
		# Update the Neural Network inputs
		self.update_inputs()	
		
		# Return the output
		return self.intermediate_output[self.number_of_layers - 1]
		
	# Function to update the inputs
	def update_inputs(self):
		# Loop through the input_connections
		for i in range(len(self.input_connections)):
			# Variable to help update the input from left to right
			update_index = 0
			for connections in self.input_connections[i]:
				# Keep updating the recurrent_input in a left to right fashion
				self.layer_vector[i].set_recurrent_input(self.intermediate_output[connections].flatten(), update_index)
				update_index = update_index + 1
	
	# Function to process the connections that are input and output
	def process_connections(self, layer_dimensions):
		# Initialize the input and output connection list
		self.input_connections = [[]] * (self.number_of_layers)
		self.output_connections = [[]] * (self.number_of_layers)
		
		# If the current layer has a connection with a layer to it's right, then it is an input for that layer
		# If the current layer has a connection with a layer to it's left, then it is an output for that layer
		for i in range(self.number_of_layers):
			for j in layer_dimensions[i][2]:
				if j >= i:
					self.input_connections[i].append(j)
				else:
					self.output_connections[i].append(j) 
	
	# Function to save the layer weights
	def save_weights_to_file(self, file_name):
		# Use pickle to save the layer_vector
		# This even saves all the previous input we were working on!
		with open(filename, 'wb') as f:
			pickle.dump(self.layer_vector, f)
			
	# Function to load the layer weights
	def load_weights_from_file(self, file_name):
		# Use pickle to load the layer_vector
		with open(file_name, 'rb') as f:
			self.layer_vector = pickle.load(f)
			
	# Function to return all the weights and bias in the form of a vector
	def return_weights_as_vector(self):
		# The layout followed is the same that Static Network follows, with a few additions
		# weights of delay system + weights of recurrent system(quite varying length) + weights of static system + weights of bias
		
		# Initialize the output
		output = np.array([])
		
		for layer in range(self.number_of_layers):
			# The delay system uses a weight vector
			weight_vector_delay = self.layer_vector[layer].get_delay_weight_vector().flatten()
			
			# The recurrent weights need to be flattened and collected as well
			# They are taken from input side
			weight_vector_recurrent = np.array([])
			for index in range(len(self.input_connections[layer])):
				weight_vector_recurrent = np.concatenate([weight_vector_recurrent, self.layer_vector[layer].get_recurrent_weight_matrix(index).flatten()])
				
			# Get the static weight matrix
			weight_vector_static = self.layer_vector[layer].get_weight_matrix().flatten()
			
			# Get the bias vector
			bias_vector = self.layer_vector[layer].get_bias_vector().flatten()
			
			# concatenate everything
			output = np.concatenate([output, weight_vector_delay, weight_vector_recurrent, weight_vector_static, bias_vector])
			
		return output
		
	# Function to set all the weights and bias from a vector
	def load_weights_from_vector(self, weight_vector):
		# Same layout, therefore we need to extract and then load!
		# Convert to numpy array
		weight_vector = np.array(weight_vector)
		
		# Interval counter maintains the current layer index
		interval_counter = 0
		
		# Contrary to static neural network, in this case, we extract dimensions and then extract the weight simultaneously 
		for layer in range(self.number_of_layers):
			# Get the dimensions, interval and extract for delay
			delay_dim = self.layer_vector[layer].get_delay_weight_dim()
			delay_interval = delay_dim[0] * delay_dim[1]
			self.layer_vector[layer].set_delay_weight_vector(weight_vector[interval_counter:interval_counter+delay_interval].reshape(delay_dim))
			interval_counter = interval_counter + delay_interval
			
			# Get the dimensions, interval and extract for recurrent
			for index in range(len(self.input_connections[layer])):
				recurrent_dim = self.layer_vector[layer].get_recurrent_weight_dim(index)
				recurrent_interval = recurrent_dim[0] * recurrent_dim[1]
				self.layer_vector[layer].set_recurrent_weight_matrix(weight_vector[interval_counter:interval_counter+recurrent_interval].reshape(recurrent_dim), index)
				interval_counter = interval_counter + recurrent_interval
				
			# Get the dimensions, interval and extract for static weight
			weight_dim = self.layer_vector[layer].get_weight_matrix_dim()
			weight_interval = weight_dim[0] * weight_dim[1]
			self.layer_vector[layer].set_weight_matrix(weight_vector[interval_counter:interval_counter+weight_interval].reshape(weight_dim))
			interval_counter = interval_counter + weight_interval
			
			# Get the dimensions, interval and extract for bias vector
			bias_dim = self.layer_vector[layer].get_bias_vector_dim()
			bias_interval = bias_dim[0] * bias_dim[1]
			self.layer_vector[layer].set_bias_vector(weight_vector[interval_counter:interval_counter+bias_interval].reshape(bias_dim[0],))
			interval_counter = interval_counter + bias_interval
			
	
