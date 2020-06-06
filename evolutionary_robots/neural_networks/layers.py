"""Docstring for the layers.py module.

This library contains the various layers that constitute
a Neural Network. Combining these layers in different ways
more Neural Networks can be generated.

"""

# Import numpy
import numpy as np
from activation_functions import ActivationFunction

# Static Layer, only forward connections are present in this layer
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
	forward_propagate(self, input_vector)
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
