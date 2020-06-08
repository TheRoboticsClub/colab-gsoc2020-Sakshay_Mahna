"""Docstring for the layers.py module.

This library contains the various layers that constitute
a Neural Network. Combining these layers in different ways
more Neural Networks can be generated.

"""

# Import numpy
import numpy as np
from activation_functions import ActivationFunction

# Static Layer, only forward connections are present in this layer #################################
class StaticLayer:
	"""
	Static Layer is the simplest of all layers
	Only forward connections are present in this layer
	The layer stores the weights and calculates the output
	based on matrix multiplication
	
	...
	
	Attributes
	----------
	input_dim: integer
		Specifies the number of input nodes
	
	output_dim: integer
		Specifies the number of output nodes
	
	activation_function: ActivationFunction class
		Specifies the activation function to be used
	
	layer_name: string
		Specifies the name of the layer 
		
	Methods
	-------
	forward_propagate(input_vector)
		Calculate the output of the Layer when input_vector is passed
		
	Additional Methods
	------------------
	set_weight_matrix(weight_matrix)
		Set the weight matrix
	
	get_weight_matrix()
		Get the current weight matrix
	
	set_bias_vector(bias_vector)
		Set the bias vector
		
	get_bias_vector()
		Get the	current bias vector
		
	get_layer_name()
		Get the name of the layer
		
	get_weight_dim()
		Get the dimensions of the weight matrix
		
	get_bias_dim()
		Get the dimensions of the bias vector
		
	set_activation_parameters(beta, theta)
		Set the parameters of activation function
		
	get_activation_parameters()
		Get the parameters of activation function
	"""
	def __init__(self, input_dim, output_dim, activation_function, layer_name):
		# Set the layer name
		self.layer_name = layer_name
	
		# Initialize the weight and bias dimensions
		self.weight_dim = (output_dim, input_dim)
		self.bias_dim = (output_dim, )
		
		# Initialize the weight matrix
		self.weight_matrix = np.random.rand(*self.weight_dim)
		
		# Initialize the bias vector
		self.bias_vector = np.random.rand(output_dim)
		
		# Set the activation function
		self.activation_function = activation_function
		# Also check if the activation function is an instance of the ActivationFunction
		if(not isinstance(self.activation_function, ActivationFunction)):
			raise TypeError("The activation function has to be an instance of ActivationFunction class")
		
	def forward_propagate(self, input_vector):
		"""
		Generate the output of the Layer when input_vector
		is passed to the Layer
		
		Parameters
		----------
		input_vector: array_like
			The input_vector to be passed to the Layer
			
		Returns
		-------
		output_vector: array_like
			The output_vector generated from the input_vector
			
		Raises
		------
		ValueException
			The input_vector should be of the dimension as specified by the user earlier
			
		Notes
		-----
		The input_vector has dimensions (input_dim, )
		The output_vector has dimensions (output_dim, )
		The bias_vector has dimensions (output_dim, )
		The weight_matrix has dimensions (output_dim, input_dim)
		Each of the columns of the weight_matrix tune for a single input node
		Each of the rows of the weight_matrix tune for a single output node
		
		The output_vector is generated using the formula
		output_vector = weight_matrix . input_vector + bias_vector
		"""
		
		# Convert the input vector to numpy array
		input_vector = np.array(input_vector)
		
		# Check for the proper dimensions of the input vector
		if(input_vector.shape != (self.weight_dim[1], )):
			raise ValueError("The dimensions of input vector do not match!")
		
		# Output vector is obtained by dotting weight and input, then adding with bias
		output_vector = np.add(np.dot(self.weight_matrix, input_vector), self.bias_vector)
		
		# Activate the output
		output_vector = self.activation_function.calculate_activation(output_vector)
		
		return output_vector
	
	# Function to set the weight matrix	
	def set_weight_matrix(self, weight_matrix):
		"""
		Set a user defined weight matrix
		
		Raises a Value Exception if the dimensions do not match
		"""
		if(weight_matrix.shape != self.weight_dim):
			raise ValueError("The dimensions of weight matrix do not match!")
		
		self.weight_matrix = weight_matrix
		
	# Function to return the weight matrix
	def get_weight_matrix(self):
		""" Get the weight matrix that the Layer is using """
		return self.weight_matrix
	
	# Function to set the bias vector	
	def set_bias_vector(self, bias_vector):
		"""
		Set a user defined bias vector
		
		Raises a Value Exception if the dimensions do not match
		"""
		if(bias_vector.shape != self.bias_dim):
			raise ValueError("The dimensions of bias vector do not match!")
			
		self.bias_vector = bias_vector
		
	# Function to return the bias vector
	def get_bias_vector(self):
		""" Get the bias vector that the Layer is using """
		return self.bias_vector
		
	# Function to return the layer name
	def get_layer_name(self):
		""" Get the name of the Layer """
		return self.layer_name
		
	# Function to return the weight dimensions
	def get_weight_dim(self):
		""" Get the dimensions of the weight matrix """
		return self.weight_dim
		
	# Function return the bias dimensions
	def get_bias_dim(self):
		""" Get the dimensions of the bias vector """
		return self.bias_dim
		
	# Function to set the activation parameters
	def set_activation_parameters(self, beta, theta):
		""" Set the parameters of activation function """
		self.activation_function.set_parameters(beta, theta)
	
	# Function to return the activation parameters(tuple)	
	def get_activation_parameters(self):
		""" Get the parameters of activation function """
		return self.activation_function.get_parameters()
		
# Dynamic Layer ##############################################################
# The time delay network works by returning the weighted average of the input vectors
# No activation is given at the moment!################
# delay_dim specifies the delay, a value of 1 gives the same Static behaviour
# Class for Time Delay System
class TimeDelay:
	"""
	Class to implement the TimeDelay system
	
	...
	
	Attributes
	----------
	input_dim: integer
		The input dimensions of the layer
		
	delay_dim: integer
		The amount of delay we require. Specifying 1 implies the system behaves normally
		
	Methods
	-------
	forward_propagate(input_vector)
		Calculate the weighted average of the previous inputs and the current output as specifed by delay
		
	
	Additional Methods
	------------------
	get_weight_vector()
		Function used to return the weight vector
	
	set_weight_vector()
		Function used to set the weight vector
		
	get_weight_vector_dim()
		Function used to return the dimensions of the weight vector
		
	Notes
	-----
	Instead of weight matrix, a weight vector is used in this system
	A weight vector is simply a weight matrix with the dimensions (1, delay_dim)
	"""
	def __init__(self, input_dim, delay_dim):
		# Weight dimensions and input dimensions
		self.weight_dim = (1, delay_dim)
		self.input_matrix_dim = (delay_dim, input_dim)
		
		# Generate the weight vector and input matrix
		self.weight_vector = np.random.rand(*self.weight_dim)
		self.input_matrix = np.zeros(self.input_matrix_dim)
		
	# Forward propagate
	def forward_propagate(self, input_vector):
		"""
		Function to generate the output of the system
		
		Parameters
		----------
		input_vector: array_like
			The input vector to be passed to the system
			
		Returns
		-------
		output: array_like
			THe output vector generated by the system
			
		Raises
		------
		ValueException
			The input vector should be of the dimensions as specified by the user earlier
			
		"""
		# Convert to numpy array
		input_vector = np.array(input_vector)

		# Check dimensions
		if(input_vector.shape != (self.input_matrix_dim[1], )):
			raise ValueError("The dimensions of the input vector do not match!")
		
		# Insert the input_vector and remove the oldest one
		self.input_matrix = np.insert(self.input_matrix, 0, input_vector, axis=0)
		self.input_matrix = np.delete(self.input_matrix, len(self.input_matrix) - 1, axis=0)
		
		# Generate the output
		output = np.dot(self.weight_vector, self.input_matrix).flatten()
		
		return output
		
	# Function to return the weight vector
	def get_weight_vector(self):
		""" Function to get the weight vector """
		return self.weight_vector
		
	# Function to set the weight vector
	def set_weight_vector(self, weight_vector):
		"""
		Function to set the weight vector
		Raises an exception when the dimensions do not match!
		"""
		if(weight_vector.shape != self.weight_dim):
			raise ValueError("THe dimensions of the weight vector do not match!")
		
		self.weight_vector = weight_vector
		
	# Function to return the dimensions of weight vector
	def get_weight_vector_dim(self):
		""" Function to return the dimensions of the weight vector """
		return self.weight_dim
		

# Class for Time Recurrence system
class TimeRecurrence:
	"""
	Class for Time Recurrence System.
	
	...
	
	Attributes
	----------
	input_dim: integer
		The input dimensions of the layer
		
	output_dim: integer
		The output dimensions of the layer
		
	Methods
	-------
	forward_propagate()
		Calculates the output generated by the layer, by using the input vector that is stored and not passed!
		
	Additional Methods
	------------------
	get_weight_matrix()
		Function to return the weight matrix used by the layer
	
	set_weight_matrix(weight_matrix)
		Function to set the weight matrix to be used by the layer
		
	set_input_vector(input_vector)
		Function to set the input vector
		
	get_weight_matrix_dim()
		Function to return the dimensions of the weight matrix
		
	Notes
	-----
	In this system, the input vector is kept stored by the system. When the output
	is to be generated, the stored input vector is used to generate that output.
	input_dim are the dimensions of the later layer, which provides its output(right)
	output_dim are the dimensions of the layer that uses those values(left)
	"""
	def __init__(self, input_dim, output_dim):
		# Weight dimensions
		self.weight_dim = (output_dim, input_dim)
		
		# Initialize the weight matrix
		self.weight_matrix = np.random.rand(*self.weight_dim)
		
		# Initialize the input vector
		self.input_vector = np.zeros((input_dim,))
		
	# Forward propagate
	def forward_propagate(self):
		""" Function to generate the output of the system """
		# Generate the output and return it
		output = np.dot(self.weight_matrix, self.input_vector)
		return output
		
	# Function to get the weight matrix
	def get_weight_matrix(self):
		""" Function to get the weight matrix """
		return self.weight_matrix
		
	# Function to set the weight matrix
	def set_weight_matrix(self, weight_matrix):
		"""
		Function to set the weight matrix
		Raises an exception when the dimensions do not match
		"""
		if(weight_matrix.shape != self.weight_dim):
			raise ValueError("THe dimensions of the weight matrix do not match!")
		self.weight_matrix = weight_matrix
		
	# Function to set the input_vector
	def set_input_vector(self, input_vector):
		""" 
		Function to set the input vector 
		Raises an exception when the dimensions do not match
		"""
		if(input_vector.shape != (self.weight_dim[1], )):
			raise ValueError("The dimensions of the input vector are not correct!")
		self.input_vector = input_vector
		
	# Function to return the dimensions of weight matrix
	def get_weight_matrix_dim(self):
		""" Function to return the dimensions of the weight matrix """
		return self.weight_dim
		
# Class for Dynamic Layer
class DynamicLayer:
	"""
	The Dynamic Layer class.
	This layer consists of the Static, Delay and Recurrent System
	
	...
	
	Attributes
	----------
	input_dim: integer
		The input dimensions of the layer
	
	output_dim: integer
		THe output dimensions of the layer
		
	recurrent_dim: array_like
		A list of output dimensions of other dynamic layers which are going
		to provide the recurrent relation. The dimensions are specified from
		left to right
		
	activation_function: ActivationFunction class
		Specifies the activation function to be used for the Static System
		
	layer_name: string
		The name of the layer
		
	Methods
	-------
	forward_propagate(input_vector)
		Generate the output of the layer given the input vector
		
	Additional Methods
	------------------
	set_recurrent_input(input_vector, index)
		Set the input_vector of recurrent system(given index)
		
	set_weight_matrix(weight_matrix)
		Sets the weight matrix of the static system
		
	get_weight_matrix()
		Returns the weight matrix of the static system
		
	set_recurrent_weight_matrix(weight_matrix, index)
		Sets the weight matrix of the recurrent system(given index)
		
	get_recurrent_weight_matrix(index)
		Returns the weight matrix of the recurrent system(given index)
		
	set_delay_weight_vector(weight_vector)
		Sets the weight vector of the delay system
		
	get_delay_weight_vector()
		Returns the weight vector of the delay system
		
	set_bias_vector(bias_vector)
		Sets the bias vector of the static system
		
	get_bias_vector()
		Returns the bias vector of the static system
		
	get_delay_weight_dim()
		Returns the dimensions of the weight matrix of delay system
		
	get_recurrent_weight_dim(index)
		Returns the dimensions of the weight matrix of recurrent system(given index)
		
	get_weight_matrix_dim()
		Returns the dimensions of the weight matrix of the static system
		
	get_bias_vector_dim()
		Returns the dimensions of the bias vector of the static system
		
	get_layer_name()
		Returns the name of the layer
		
	set_activation_parameters(beta, theta)
		Set the parameters of activation function
		
	get_activation_parameters()
		Get the parameters of activation function
	"""
	def __init__(self, input_dim, output_dim, delay_dim, recurrent_dim, activation_function, layer_name):
		# Set the layer name
		self.layer_name = layer_name
		
		# Initialize the weight and bias dimensions
		self.weight_dim = (output_dim, input_dim)
		self.bias_dim = (output_dim, )
		
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
		# Also check if the activation function is an instance of the ActivationFunction
		if(not isinstance(self.activation_function, ActivationFunction)):
			raise TypeError("The activation function has to be an instance of ActivationFunction class")
			
	# Forward Propagate
	def forward_propagate(self, input_vector):
		"""
		Function to generate the output of the layer
		
		Parameters
		----------
		input_vector: array_like
			THe input vector to be passed to the layer
			
		Returns
		-------
		intermediate_output: array_like
			The output vector generated by the layer
			
		Raises
		------
		ValueException
			The input vector should be of the correct dimensions
			
		Notes
		-----
		The output vector is generated as 
		Calculate the delayed input
		Calculate the static output along with addition of bias
		Calculate the weighted sum with recurrent inputs
		Calculate the activation
		"""
		# Convert to numpy array
		input_vector = np.array(input_vector)
		
		if(input_vector.shape != (self.weight_dim[1], )):
			raise ValueError("THe input dimensions do not match as specified earlier!")
		
		# Get the input from delay system
		input_vector = self.delay_system.forward_propagate(input_vector)
		
		# intermediate_output keeps storing the outputs and adding them
		intermediate_output = np.add(np.dot(self.weight_matrix, input_vector), self.bias_vector)
		
		# Get the outputs from recurrent system
		for recurrence in self.recurrent_system:
			intermediate_output = np.add(intermediate_output, recurrence.forward_propagate())
			
		# Activate the output
		intermediate_output = self.activation_function.calculate_activation(intermediate_output)
		
		return intermediate_output
		
	# Function to set the input_vector of the recurrent layer
	def set_recurrent_input(self, input_vector, index):
		"""
		Function to set the input vector of the recurrent system 
		Raise an exception if dimensions do not match
		"""
		# Get the weight dimensions from the recurrent system
		correct_dimensions = self.recurrent_system[index].get_weight_matrix_dim()
		
		# Check the dimensions
		if(input_vector.shape != (correct_dimensions[1], )):
			raise ValueError("THe dimensions do not match!")
		
		self.recurrent_system[index].set_input_vector(input_vector)
		
	# Function to set the weight matrix
	def set_weight_matrix(self, weight_matrix):
		"""
		Function to set the weight matrix of the static system
		Raise an exception if dimensions do not match
		"""
		if(weight_matrix.shape != self.weight_dim):
			raise ValueError("THe dimensions of the weight matrix do not match!")
			
		self.weight_matrix = weight_matrix
		
	# Function to return the weight matrix
	def get_weight_matrix(self):
		""" Function to return the weight matrix """
		return self.weight_matrix
		
	# Function to set the recurrent weight matrix
	def set_recurrent_weight_matrix(self, weight_matrix, index):
		"""
		Function to set the weight matrix of the recurrent system
		Raises an exception if dimensions do not match
		"""
		# Get the weight dimensions from the recurrent system
		correct_dimensions = self.recurrent_system[index].get_weight_matrix_dim()
		
		# CHeck the dimensions
		if(weight_matrix.shape != correct_dimensions):
			raise ValueError("The dimensions of weight matrix do not match!")
		
		self.recurrent_system[index].set_weight_matrix(weight_matrix)
		
	# Function to get the recurrent weight matrix
	def get_recurrent_weight_matrix(self, index):
		""" Function to return the recurrent weight matrix """
		return self.recurrent_system[index].get_weight_matrix()
		
	# Function to set the delay weight
	def set_delay_weight_vector(self, weight_vector):
		""" 
		Function to set the delay weight vector
		Raises an exception if the dimensions do not match
		"""
		
		# Get the dimensions from the delay system
		correct_dimensions = self.delay_system.get_weight_vector_dim()
		
		if(weight_vector.shape != correct_dimensions):
			raise ValueError("The dimensions of delay weight vector do not match!")
		
		self.delay_system.set_weight_vector(weight_vector)
		
	# Function to return the delay weight
	def get_delay_weight_vector(self):
		""" Function to return the delay weight vector """
		return self.delay_system.get_weight_vector()
		
	# Function to set the bias vector
	def set_bias_vector(self, bias_vector):
		"""
		Function to set the bias vector of the static system
		Raises an exception if the dimensions do not match!
		"""
		if(bias_vector.shape != self.bias_dim):
			raise ValueError("The dimensions of the bias vector do not match!")
		
		self.bias_vector = bias_vector
		
	# Function to return the bias vector
	def get_bias_vector(self):
		""" Function to return the bias vector """
		return self.bias_vector
		
	# Function to return the delay weight dimensions
	def get_delay_weight_dim(self):
		""" Function to return the dimensions of the delay weight vector """
		return self.delay_system.get_weight_vector_dim()
		
	# Function to return the recurrent weight matrix dimensions
	def get_recurrent_weight_dim(self, index):
		""" Function to return the dimensions of the recurrent weight matrix """
		return self.recurrent_system[index].get_weight_matrix_dim()
		
	# Function to return the static weight matrix dimensions
	def get_weight_matrix_dim(self):
		""" Function to return the dimensions of the static weight matrix """
		return self.weight_dim
		
	# Function to return the bias vector dimensions
	def get_bias_vector_dim(self):
		""" Function to return the dimensions of the static bias vector """
		return self.bias_dim
		
	# Function to set the activation parameters
	def set_activation_parameters(self, beta, theta):
		""" Set the parameters of activation function """
		self.activation_function.set_parameters(beta, theta)
	
	# Function to return the activation parameters(tuple)	
	def get_activation_parameters(self):
		""" Get the parameters of activation function """
		return self.activation_function.get_parameters()
		
	# Function to return the layer name
	def get_name(self):
		""" Function to return the name of the layer """
		return self.layer_name

# CTRNN Layer ################################################################
class CTRNNLayer:
	"""
	CTRNN Layer is used in the Continuous Time Recurrent Neural Network.
	CTRNN has to save the state of the previous output and then calculate
	the weighted average of the previous and the current output to get
	the total output
	
	...
	
	Attributes
	----------
	input_dim: integer
		The input dimension of the Layer
	
	output_dim: integer
		The output dimension of the layer
		
	time_interval: integer
		The time interval of the CTRNN Network
		
	time_constant: array_like
		The time_constants of the nodes in the Layer
	
	activation_function: ActivationFunction class
		SPecifies the activation function to be used
		
	layer_name: string
		THe name of the layer
		
	Methods
	-------
	euler_step(input_vector):
		Calculates the next step of the Layer based on first degree Euler Approximation
		
	Additional Methods
	------------------
	set_weight_matrix(weight_matrix)
		Sets the weight matrix used by the layer
		
	get_weight_matrix()
		Used to retreive the weight matrix used by the Layer
		
	set_bias_vector(bias_vector)
		Sets the bias vector used by the layer
		
	get_bias_vector()
		Retreives the bias vector used by the layer
		
	get_name()
		Returns the name of the layer
		
	get_weight_dim()
		Returns the dimension of the weight matrix
	
	get_bias_dim()
		Returns the dimension of the bias vector
		
	get_time_constants()
		Returns the time constants used by the layer
		
	"""
	# time_interval is a scalar
	# time_constant is a list, as it is different for each i-th neuron
	def __init__(self, input_dim, output_dim, time_interval, time_constant, activation_function, layer_name):
		# Name of the layer
		self.layer_name = layer_name
		
		# Weight and bias dimensions
		self.weight_dim = (output_dim, input_dim)
		self.bias_dim = (output_dim, )
		
		# Initialize the weight and bias
		self.weight_matrix = np.random.rand(*self.weight_dim)
		self.bias_vector = np.random.rand(output_dim)
		
		# Generate the weights for the weighted average
		self.time_constant = time_constant
		self.time_weight = np.asarray(float(time_interval) / np.array(time_constant))
		
		# Set the activation function
		self.activation_function = activation_function
		# Also check if the activation function is an instance of the ActivationFunction
		if(not isinstance(self.activation_function, ActivationFunction)):
			raise TypeError("The activation function has to be an instance of ActivationFunction class")
		
		# Set the previous state output, zero for initial
		self.previous_output = np.zeros(output_dim)
		
	# First order euler step
	def euler_step(self, input_vector):
		"""
		Function that calculates the next step based on the first degree Euler approximation
		
		Parameters
		----------
		input_vector: array_like
			THe input_vector to be passed to the layer
			
		Returns
		-------
		current_output: array_like
			The current state, or the output generated in this time step
			
		Raises
		------
		ValueException
			The input vector has to be of the correct dimensions
			
		Notes
		-----
		The input_vector is of dimensions (input_dim, )
		The output_vector is of dimensions (output_dim, )
		The weight_matrix is of dimensions (output_dim, input_dim)
		The bias vector is of dimensions (output_dim, )
		
		Each of the columns of the weight_matrix tune for a single input node
		Each of the rows of the weight_matrix tune for a single output node
		
		"""
	
		# Convert to numpy array
		input_vector = np.array(input_vector)
		
		if(input_vector.shape != (self.weight_dim[1], )):
			raise ValueError("The dimensions of input vector do not match!")
		
		# Get the current activation
		current_activation = np.add(np.dot(self.weight_matrix, input_vector), self.bias_vector)
		current_activation = self.activation_function.calculate_activation(current_activation)
		
		# Generate the current output
		# This equation is the first order euler solution
		current_output = self.previous_output * (1 - self.time_weight) + current_activation * self.time_weight
		
		# Save it!
		self.previous_output = current_output
		
		return current_output
		
	# Function to set the weight matrix	
	def set_weight_matrix(self, weight_matrix):
		"""
		Sets the weight matrix to be used by the layer
		Raise a value exception if dimensions do not match
		"""
		if(weight_matrix.shape != self.weight_dim):
			raise ValueError("The dimensions of the weight matrix do not match")
		self.weight_matrix = weight_matrix
		
	# Function to return the weight matrix
	def get_weight_matrix(self):
		""" Function used to return the weight matrix """
		return self.weight_matrix
	
	# Function to set the bias vector	
	def set_bias_vector(self, bias_vector):
		"""
		Sets the bias vector to be used by the layer
		Raise a value exception if dimensions do not match
		"""
		if(bias_vector.shape != self.bias_dim):
			raise ValueError("The dimensions of the bias vector do not match!")
		self.bias_vector = bias_vector
		
	# Function to return the bias vector
	def get_bias_vector(self):
		""" Function to return the bias vector """
		return self.bias_vector
		
	# Function to return the layer name
	def get_name(self):
		""" Function to return the name of the layer """
		return self.layer_name
		
	# Function to return the weight dimensions
	def get_weight_dim(self):
		""" Function to return the dimensions of the weight matrix """
		return self.weight_dim
		
	# Function return the bias dimensions
	def get_bias_dim(self):
		""" Function to return the dimensions of the bias vector """
		return self.bias_dim
	
	# Function to return the time constant list
	def get_time_constant(self):
		""" Function to return the list of time constants """
		return self.time_constant

		
# Radial Basis Function Layer ################################################
class RBFLayer:
	"""
	The Radial Basis Function Layer for RBF Networks
	This layer has additional parameters for specifying
	the center of each neuron, the distance function and the
	basis function. Activation Function is not present in this layer
	
	...
	
	Attributes
	----------
	input_dim: integer
		Specifies the number of input nodes
	
	output_dim: integer
		Specifies the number of nodes in output
		
	distance_function: function
		Specifies the distance function. The input parameters
		of the function are a numpy array with dimensions as
		input_dim and center matrix with dimensions as (rxc)
		output_dim x input_dim. It returns a numpy array with
		dimension as output_dim
		
	basis_function: function
		Specifies the basis function. The input parameters
		of the function are a numpy array with dimensions as
		output_dim and a parameter value which specifies the
		gain of the basis function. It returns a numpy array
		with dimensions as output_dim
		
	parameter: float
		The parameter of the basis function
		
	layer_name: string
		Specifies the name of the layer
		
	Methods
	-------
	forward_propagate(input_vector)
		Calculates the output of the layer given an input vector
		
	Additional Methods
	----------------
	set_center_matrix(center_matrix)
		Sets the centers used by the nodes of the layer
		
	get_center_matrix()
		Used to retreive the center matrix of the layer
		
	get_center_dim()
		Used to retreive the dimensions of the center matrix
		
	get_name()
		Used to return the name of the layer
	"""
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
		"""
		Generates the output of the Layer when an input vector is passed
		
		Parameters
		----------
		input_vector: array_like
			The input_vector to be passed to the Layer
			
		Returns
		-------
		output: array_like
			The output vector generated by the Layer
			
		Raises
		------
		ValueException
			The input_vector should be of the dimension as specified earlier
			
		Notes
		-----
		The input_vector has dimensions (input_dim, )
		The output has dimensions (output_dim, )
		The center matrix has dimensions (output_matrix, input_matrix)
		"""
	
		# Convert to numpy array
		input_vector = np.array(input_vector)
		
		if(input_vector.shape != (self.center_dim[1], )):
			raise ValueError("The dimensions of input vector do not match!")
		
		# Compute the distance of the input and the centers
		# The distance function has to calculate a distance vector
		# based on the center matrix and input_vector
		distance_vector = self.distance_function(input_vector, self.center_matrix)
		
		# Calculate the output of the basis function
		output = self.basis_function(distance_vector, self.parameter)
		
		return output
		
	# Function to set the center matrix
	def set_center_matrix(self, center_matrix):
		"""
		Set the center matrix used by the layer
		
		Raise a Value Exception if the dimensions of matrix do not match
		"""
		if(center_matrix.shape != self.center_dim):
			raise ValueError("The dimensions of the center matrix do not match!")
		
		self.center_matrix = center_matrix
		
	# Function to get the center matrix
	def get_center_matrix(self):
		""" Function to return the center matrix used by the layer """
		return self.center_matrix
		
	# Function to return the dimensions of the center matrix
	def get_center_dim(self):
		""" Function to return the dimensions of the center matrix """
		return self.center_dim
		
	# Function to return the layer name
	def get_name(self):
		""" Function to return the name of the layer """
		return self.layer_name
