"""Docstring for the layers.py module.

This library contains the various layers that constitute
a Neural Network.

"""

# Import numpy
import numpy as np
from activation_functions import ActivationFunction
import warnings

# Simple Layer, simple feed forward connection #################################
class StaticLayer(object):
	"""
	Static Layer works by calculating the activation of each neuron
	using the feed forward algorithm
	
	...
	Attributes
	----------
	weight_matrix: numpy_matrix
		The weight matrix used to calculate the output using 
		feed forward algorithm
		This attribute can be changed
		
	weight_dim: tuple
		The dimensions of the weight matrix
		This attribute cannot be changed
		
	gain: array_like
		The gain array used to modify the sensor input that is taken
		This attribute can be changed
		
	layer_name: string
		The name of the layer
		This attribute cannot be changed
	 
	
	Parameters
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
		
	update_parameters(parameter_vector)
		Update the parameters based on a vector passed as argument
		
	return_parameters()
		Return the parameters in the form of a vector
		
	Additional Methods
	------------------
	set_activation_parameters(beta, theta)
		Set the parameters of activation function
		
	get_activation_parameters()
		Get the parameters of activation function
	"""
	def __init__(self, input_dim, output_dim, activation_function, layer_name):
		"""
		Initialization function of StaticLayer class		
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
		# Set the layer name
		self.__layer_name = layer_name
	
		# Initialize the weight and bias dimensions
		self.__weight_dim = (output_dim, input_dim)
		
		# Initialize the weight matrix
		self.weight_matrix = np.random.rand(*self.weight_dim)
		
		# Set the gain of the sensors values that are going to be input
		# as an associative layer or input layer
		self._gain = np.ones((output_dim, ))
		
		# Set the activation function
		self.__activation_function = activation_function
		# Also check if the activation function is an instance of the ActivationFunction
		if(not hasattr(self.__activation_function, 'calculate_activation')):
			raise TypeError("The activation class needs to contain a method calculate_activation for " + self.__layer_name)
		
	def forward_propagate(self, input_vector, sensor_input):
		"""
		Generate the output of the Layer when input_vector
		is passed to the Layer
		
		Parameters
		----------
		input_vector: array_like
			The input_vector to be passed to the Layer
			
		sensor_input: array_like
			The sensor input to be added
			
		Returns
		-------
		output_vector: array_like
			The output_vector according according to the output generated
			
		Raises
		------
		ValueException
			The input_vector should be of the dimension as specified by the user earlier
			
		Notes
		-----
		The input_vector has dimensions (input_dim, )
		The output_vector has dimensions (output_dim, )
		The sensor_input should have dimensions (output_dim, )
		The weight_matrix has dimensions (output_dim, input_dim)
		Each of the columns of the weight_matrix tune for a single input node
		Each of the rows of the weight_matrix tune for a single output node
		
		The output_vector is generated using the formula
		output_vector = activation(beta * (weight_matrix . input_vector - theta ) )
		"""
		
		try:
			# Convert the input vector to numpy array
			input_vector = np.array(input_vector)
			sensor_input = np.array(sensor_input)
			
			# Output vector is obtained by dotting weight and input, then adding with bias
			output_vector = np.dot(self.weight_matrix, input_vector)
			
			# Activate the output
			output_vector = self.__activation_function.calculate_activation(output_vector)
			
			# Add the sensor input
			output_vector = output_vector + np.multiply(self.gain, sensor_input)
			
			# Output 
			return output_vector
			
		except:
			raise ValueError("Please check dimensions of " + self.__layer_name)

		
	# Function to update the parameters
	def update_parameters(self, parameter_vector):
		"""
		Load the parameters of the Static Layer in the form of
		an array / vector
		
		Parameters
		----------
		parameter_vector: array_like
			The parameter vector follows the layout as
			[w_11, w_21, w_12, w_22, w_13, w_23, a_1g, a_2g, a_3g, a_1b, a_2b, a_3b, w_11, w_21, w_31, a_1g, ...]
			Here, w_ij implies the weight between ith input node and jth output node.
			a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of 
			ith output node. 
			
		Returns
		-------
		None
		
		Raises
		------
		ValueException
			The parameter array is shorter than required
			
		Warning
			The parameter array is not of the right dimension
			
		Notes
		-----
		The Exception will not always take place for parameter vectors shorter than required
		The layer will generate output or not also cannot be said so, therefore, a warning is enough
		"""
		# Convert to numpy array
		parameter_vector = np.array(parameter_vector)
	
		# Interval counter maintains the current layer index
		interval_counter = 0
		
		# Get the interval at which the weight and activation seperate
		weight_interval = self.weight_dim[0] * self.weight_dim[1]
		
		# Get the interval at which activatation parameters seperate
		activation_interval = self.weight_dim[0]
		
		# Seperate the weights and activation parameters and then reshape them
		# Numpy raises a None Type Exception, as it cannot reshape a None object
		# If such an excpetion occurs, raise a value error as our parameter_vector
		# is shorter than required
		try:
			self.weight_matrix = parameter_vector[interval_counter:interval_counter + weight_interval].reshape(self.weight_dim)
			interval_counter = interval_counter + weight_interval
			
			self.set_activation_parameters(parameter_vector[interval_counter:interval_counter+activation_interval], parameter_vector[interval_counter+activation_interval: interval_counter+2*activation_interval])
			interval_counter = interval_counter + 2 * activation_interval
			
		except:
			raise ValueError("The parameter_vector of " + self.__layer_name + " consists of elements less than required")
			
		# The interval counter should contain the number of elements in parameter_vector
		# Otherwise the user has specified parameters different than required
		if(len(parameter_vector) != interval_counter):
			warnings.warn("The number of parameters entered do not match the parameter vector of " + self.__layer_name)
			
	# Function to return the parameters
	def return_parameters(self):
		"""
		Return the parameters of the Static Layer in the form of
		a an array / vector.
		
		Parameters
		----------
		None
		
		Returns
		-------
		output: array_like
			The vector representation of the parameters of the Neural Network
			
		Raises
		------
		None
		
		Notes
		-----
		The numpy flatten function works in row major order.
		The parameter vector follows the layout as
		[w_11, w_21, w_12, w_22, w_13, w_23, a_1g, a_2g, a_3g, a_1b, a_2b, a_3b, w_11, w_21, w_31, a_1g, ...]
		Here, w_ij implies the weight between ith input node and jth output node
		a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of 
		ith output node.
		"""
		# Initialize the output vector
		# Determine an individual layer's weight matrix in row major form and activation parameters
		# Then concatenate it with the previous output vector
		output = np.array([])
		
		# The vector we get from flattening the weight matrix
		# flatten() works in row major order
		weight_vector = self.weight_matrix.flatten()
		
		# The vector of activation parameters
		activation_vector = self.get_activation_parameters()
		
		# The output vector is concatenated form of weight_vector and activation_vector
		output = np.concatenate([output, weight_vector, activation_vector])
		
		return output
	
	# Setters and Getters	
	@property
	def weight_matrix(self):
		""" The weight matrix 
			A user defined weight matrix can also be set
			Raises a Value Exception if the dimensions do not match
		"""
		return self.__weight_matrix
		
	@weight_matrix.setter
	def weight_matrix(self, weight_matrix):
		if(weight_matrix.shape != self.weight_dim):
			raise ValueError("The dimensions of weight matrix do not match for " + self.__layer_name)
		
		self.__weight_matrix = weight_matrix
		
	# Function to return the layer name
	@property
	def layer_name(self):
		""" The name of the Layer """
		return self.__layer_name
		
	# Function to return the weight dimensions
	@property
	def weight_dim(self):
		""" The dimensions of the weight matrix """
		return self.__weight_dim

	# For changing the gain values
	@property
	def gain(self):
		""" The gain for sensor input 
			This is an adjustable parameter
			Raises a Value Exception if the dimensions do not match
		"""
		
		return self._gain
		
	@gain.setter
	def gain(self, gain):
		if(self._gain.shape != self.__weight_dim[0]):
			raise ValueError("The dimensions of gain are not correct for " + self.__layer_name)
		
		self._gain = gain
		
	# Function to set the activation parameters
	def set_activation_parameters(self, beta, theta):
		""" Function to set the parameters of activation function """
		self.__activation_function.beta = beta
		self.__activation_function.theta = theta
	
	# Function to return the activation parameters(tuple)	
	def get_activation_parameters(self):
		""" Function to get the parameters of activation function """
		return np.concatenate([self.__activation_function.beta, self.__activation_function.theta])
		

# Dynamic Layer ################################################################
class DynamicLayer(object):
	"""
	Dynamic Layer is used in the Dynamic Neural Networks.
	Dynamic Layer has to save the state of the previous output 
	and then calculate the output by considering the previous 
	output and the current activations passed to it
	
	...
	Attributes
	----------
	weight_matrix: numpy_matrix
		The weight matrix used to calculate the output using 
		feed forward algorithm
		This attribute can be changed
		
	weight_dim: tuple
		The dimensions of the weight matrix
		This attribute cannot be changed
		
	time_constant: array_like
		The time constants of the neurons used for calculation of 
		dynamic network output
		This attribute can be changed
		
	time_dim: tuple
		The dimensions of the time constant array
		This attribute cannot be changed
		
	gain: array_like
		The gain array used to modify the sensor input that is taken
		This attribute can be changed
		
	layer_name: string
		The name of the layer
		This attribute cannot be changed
	
	Parameters
	----------
	input_dim: integer
		The input dimension of the Layer
	
	output_dim: integer
		The output dimension of the layer
	
	activation_function: ActivationFunction class
		SPecifies the activation function to be used
		
	layer_name: string
		The name of the layer
		
	Methods
	-------
	forward_propagate(input_vector)
		Calculates the output based on the input_vector
		
	update_parameters(parameter_vector)
		Updates the parameters of the layer according to 
		the parameter_vector argument
		
	return_parameters()
		Return the parameters of the layer
		
	Additional Methods
	------------------
	set_activation_parameters(beta, theta)
		Set the parameters of activation function
		
	get_activation_parameters()
		Get the parameters of activation function
		
	"""
	def __init__(self, input_dim, output_dim, activation_function, 
				 time_interval, time_constant, layer_name):
		"""
		Initialization function of DynamicLayer class		
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
		# Name of the layer
		self.__layer_name = layer_name
		
		# Weight and bias dimensions
		self.__weight_dim = (output_dim, input_dim)
		self.__time_dim = (output_dim, )
		
		# Initialize the weight and bias
		self.__weight_matrix = np.random.rand(*self.weight_dim)
		
		# Initialize the gain vector
		self._gain = np.ones((output_dim, ))
		
		# Generate the weights for the weighted average
		self.__time_interval = time_interval
		self.__time_constant = time_constant
		self.__time_weight = np.array(self.time_constant)
		
		# A check for the dimension of time constant list
		if(self.__time_weight.shape != self.__time_dim):
			raise ValueError("The dimension of time constant list is incorrect for " + self.__layer_name)
		
		# Set the activation function
		self.__activation_function = activation_function
		# Also check if the activation function is an instance of the ActivationFunction
		if(not hasattr(self.__activation_function, 'calculate_activation')):
			raise TypeError("The activation function needs to has an attribute calculate_activation for " + self.__layer_name)
		
		# Set the previous state output, zero for initial
		self.__previous_output = np.zeros(output_dim)
		
	# Function to calculate the output of the layer
	def forward_propagate(self, input_vector, sensor_input):
		"""
		Function that calculates the next step based on the previous
		output and current input
		
		Parameters
		----------
		input_vector: array_like
			THe input_vector to be passed to the layer
			
		sensor_input: array_like
			The values of the sensor that are to be added
			
		Returns
		-------
		current_output: array_like
			The output vector according to the output generated
			
		Raises
		------
		ValueException
			The input vector has to be of the correct dimensions
			
		Notes
		-----
		The input_vector is of dimensions (input_dim, )
		The output_vector is of dimensions (output_dim, )
		The weight_matrix is of dimensions (output_dim, input_dim)
		
		Each of the columns of the weight_matrix tune for a single input node
		Each of the rows of the weight_matrix tune for a single output node
		
		The mathematics working behind is:
		n_i[k] = n_i[k-1] + time_weight * (g * x_i[k] + sum_j{w * a_j[k-1]})
		a_i[k] = f(n_i[k])
		"""
		try:
			# Convert to numpy array
			input_vector = np.array(input_vector)
			
			# Get the current activation
			current_activation = np.dot(self.weight_matrix, input_vector)
			current_activation = current_activation + np.multiply(self.gain, sensor_input)
			current_activation = np.multiply(self.__time_weight, current_activation)
			
			# Generate the current output
			current_output = self.__previous_output + current_activation
			
			# Save it
			self.__previous_output = current_output
			
			# Calculate activation and return
			current_output = self.__activation_function.calculate_activation(current_activation)
			return current_output
			
		except:
			raise ValueError("Please check dimensions of " + self.__layer_name)
			
		
	# Function to update the parameters of the layer
	def update_parameters(self, parameter_vector):
		"""
		Load the parameters of the Dynamic Layer in the form of
		an array / vector
		
		Parameters
		----------
		parameter_vector: array_like
			The parameter vector follows the layout as
			[tc_1, tc_2, tc_3, w_11, w_21, w_12, w_22, w_13, w_23, a_1g, a_2g, a_3g, a_1b, a_2b, a_3b, w_11, w_21, w_31, b_1, ...]
			Here, w_ij implies the weight between ith input node and jth output node.
			a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
			tc_i is the time constant of ith neuron of the current layer 
			
		Returns
		-------
		None
		
		Raises
		------
		ValueException
			The parameter array is shorter than required
			
		Warning
			The parameter array is greater than required
			
		Notes
		-----
		The Exception will not always take place for parameter vectors shorter than required
		The layer will generate output or not also cannot be said so, therefore, a warning is enough
		"""
		# Convert to numpy array
		parameter_vector = np.array(parameter_vector)
	
		# Interval counter maintains the current layer index
		interval_counter = 0
		
		# Get the interval at which time constants seperate
		time_interval = self.time_dim[0]
		
		# Get the interval at which weight seperates
		weight_interval = self.weight_dim[0] * self.weight_dim[1]
		
		# Get the interval at which activation function parameters seperate
		activation_interval = self.weight_dim[0]
		
		# Seperate the weights and then reshape them
		# Numpy raises a None Type Exception, as it cannot reshape a None object
		# If such an excpetion occurs, raise a value error as our parameter_vector
		# is shorter than required
		try:
			self.__time_constant = parameter_vector[interval_counter:interval_counter + time_interval].reshape(self.__time_dim[0], )
			interval_counter = interval_counter + time_interval
			
			self.__weight_matrix = parameter_vector[interval_counter:interval_counter + weight_interval].reshape(self.__weight_dim)
			interval_counter = interval_counter + weight_interval
			
			self.set_activation_parameters(parameter_vector[interval_counter:interval_counter + activation_interval], 
											parameter_vector[interval_counter + activation_interval: interval_counter + 2 * activation_interval])
			interval_counter = interval_counter + 2 * activation_interval
			
		except:
			raise ValueError("The parameter_vector for " + self.__layer_name + " consists of elements less than required")
			
		# The interval counter should contain the number of elements in parameter_vector
		# Otherwise the user has specified parameters not equal to the ones required
		if(len(parameter_vector) != interval_counter):
			warnings.warn("The number of parameters entered do not match the parameter vector of " + self.__layer_name)
		
	# Function to return the parameters of a layer
	def return_parameters(self):
		"""
		Return the parameters of the Dynamic Neural Network in the form of
		a an array / vector.
		
		Parameters
		----------
		None
		
		Returns
		-------
		output: array_like
			The vector representation of the parameters of the Neural Network
			
		Raises
		------
		None
		
		Notes
		-----
		The numpy flatten function works in row major order.
		The parameter vector follows the layout as
		[tc_1, tc_2, tc_3, w_11, w_21, w_12, w_22, w_13, w_23, a_1g, a_2g, a_3g, a_1b, a_2b, a_3b, w_11, w_21, w_31, b_1, ...]
		Here, w_ij implies the weight between ith input node and jth output node
		a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
		tc_i is the time constant of ith neuron of the current layer
		"""
		# Initialize the output vector
		# Determine an individual layer's weight matrix in row major form and then activation function parameters
		# Then concatenate it with the previous output vector
		output = np.array([])
		
		# The vector we get from flattening time constants
		time_vector = self.time_constant.flatten()
	
		# The vector we get from flattening the weight matrix
		# flatten() works in row major order
		weight_vector = self.weight_matrix.flatten()
		
		# The vector of activation parameters
		activation_vector = self.get_activation_parameters()
		
		# The output vector is concatenated form of time_vector, weight_vector, bias_vector and activation_vector
		output = np.concatenate([output, time_vector, weight_vector, activation_vector])
		
		return output
		
	# Function to return the weight matrix
	@property
	def weight_matrix(self):
		""" The Weight Matrix of the layer
			The user can set the weight matrix to be used
			Raise a value exception if dimensions do not match
		"""
		return self.__weight_matrix
		
	# Function to set the weight matrix
	@weight_matrix.setter
	def set_weight_matrix(self, weight_matrix):
		if(weight_matrix.shape != self.weight_dim):
			raise ValueError("The dimensions of the weight matrix do not match for " + self.__layer_name)
		self.__weight_matrix = weight_matrix
		
	# Function to return the layer name
	@property
	def layer_name(self):
		""" The name of layer """
		return self.__layer_name
		
	# Function to return the weight dimensions
	@property
	def weight_dim(self):
		""" The dimensions of weight matrix """
		return self.__weight_dim
		
	# Function to return the time constant dimension
	@property
	def time_dim(self):
		""" The dimensions of time constant array """
		return self.__time_dim
	
	# Function to return the time constant list
	@property
	def time_constant(self):
		""" The time constant array 
			The user can set the time constants to be used
			Raises an exception if the dimensions do not match
		"""
		return self.__time_constant
		
	# Function to set the time constant list
	@time_constant.setter
	def time_constant(self, time_constant):
		self.time_constant = np.array(time_constant)
		
		if(self.time_constant.shape != self.time_dim):
			raise ValueError("The dimension of time constant list is not correct for " + self.__layer_name)
			
		# Calculate the weights based on the time constants and the time interval!
		self.__time_weight = self.time_constant
		
	# For changing the gain values
	@property
	def gain(self):
		""" The gain for sensor input """
		return self._gain
		
	@gain.setter
	def gain(self, gain):
		if(self.gain.shape != self.__bias_dim):
			raise ValueError("The dimensions of gain are not correct for " + self.__layer_name)
		
		self._gain = gain
		
	# Function to set the activation parameters
	def set_activation_parameters(self, beta, theta):
		""" Function to set the parameters of activation function """
		self.__activation_function.beta = beta
		self.__activation_function.theta = theta
	
	# Function to return the activation parameters(tuple)	
	def get_activation_parameters(self):
		""" Function to get the parameters of activation function """
		return np.concatenate([self.__activation_function.beta, self.__activation_function.theta])
