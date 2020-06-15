"""Docstring for the layers.py module.

This library contains the various layers that constitute
a Neural Network. Combining these layers in different ways
more Neural Networks can be generated.

"""

# Import numpy
import numpy as np
from activation_functions import ActivationFunction
import warnings

# Input Layer, only for taking inputs from the user
class InputLayer:
	"""
	Input Layer takes input from the user and sends
	it to the other layers ahead!
	
	...
	
	Attributes
	----------
	input_dim: integer
		Specifies the number of input nodes
	
	layer_name: string
		Specifies the name of the layer 
		
	delay(optional): integer
		Specifies the delay of the connection
		
	Methods
	-------
	forward_propagate(input_vector, delay)
		An identity function with an optional delay
		
	"""
	def __init__(self, input_dim, layer_name, delay = 2):
		# Private Class declarations
		self.__input_dim = (input_dim, )
		self.__layer_name = layer_name
		self.__gain = np.ones((input_dim, ))
		self.__delay = delay

		# Output matrix is used to work with delays
		self.__output_matrix = np.zeros((delay, input_dim))
		
	# Getters and Setters
	@property
	def input_dim(self):
		""" Getter for input_dim """
		return self.__input_dim
		
	@property
	def layer_name(self):
		""" Getter for layer name """
		return self.__layer_name
		
	@property
	def gain(self):
		""" Getter for gain """
		return self.__gain
		
	@gain.setter
	def gain(self, gain):
		self.__gain = gain
		
	# Function to take input from user
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
		output_matrix: array_like
			The output_vector generated from the input_vector
			
		Raises
		------
		ValueException
			The input_vector should be of the dimension as specified by the user earlier
			
		Notes
		-----
		The input_matrix stores the various values input in the various time frames
		The output vector is returned according to the specified delay
		"""
		try:
			# Multiply with the gain
			input_vector = np.multiply(np.array(input_vector), self.gain)
			# Insert the input_vector and remove the oldest one
			self.__output_matrix = np.insert(self.__output_matrix, 0, input_vector, axis=0)
			self.__output_matrix = np.delete(self.__output_matrix, self.__delay, axis=0)
			
		except:
			raise ValueError("Please check dimensions of " + self.layer_name)
			
		# Return according to the delay
		return self.__output_matrix

# Simple Layer, simple feed forward connection with a specified delay #################################
class SimpleLayer:
	"""
	Simple Layer works by calculating the activation of each neuron
	using the feed forward algorithm and supplies the output according
	to the delay of the layer
	
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
		
	delay(optional): integer
		Specifies the delay of the connection
		
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
	def __init__(self, input_dim, output_dim, activation_function, layer_name, delay = 2):
		# Set the layer name
		self.__layer_name = layer_name
	
		# Initialize the weight and bias dimensions
		self.__weight_dim = (output_dim, input_dim)
		self.__bias_dim = (output_dim, )
		self.__delay = delay
		
		# Initialize the weight matrix
		self.__weight_matrix = np.random.rand(*self.weight_dim)
		
		# Initialize the bias vector
		self.__bias_vector = np.random.rand(output_dim)
		
		# Initialize the output matrix
		self.__output_matrix = np.zeros((delay, output_dim))
		
		# Set the gain of the sensors values that are going to be input
		# as an associative layer
		self.__gain = np.ones((output_dim, ))
		
		# Set the activation function
		self.__activation_function = activation_function
		# Also check if the activation function is an instance of the ActivationFunction
		if(not hasattr(self.__activation_function, 'calculate_activation')):
			raise TypeError("The activation class needs to contain a method calculate_activation")
		
	def forward_propagate(self, input_vector, sensor_input):
		"""
		Generate the output of the Layer when input_vector
		is passed to the Layer
		
		Parameters
		----------
		input_vector: array_like
			The input_vector to be passed to the Layer
			
		sensor_input: array_like
			The sensor input to be added(for associative layer)
			
		Returns
		-------
		output_matrix: array_like
			The output_vector according according to the delay specified above
			
		Raises
		------
		ValueException
			The input_vector should be of the dimension as specified by the user earlier
			
		Notes
		-----
		The input_vector has dimensions (input_dim, )
		The output_vector has dimensions (output_dim, )
		The sensor_input should have dimensions (output_dim, )
		The bias_vector has dimensions (output_dim, )
		The weight_matrix has dimensions (output_dim, input_dim)
		Each of the columns of the weight_matrix tune for a single input node
		Each of the rows of the weight_matrix tune for a single output node
		
		The output_vector is generated using the formula
		output_vector = weight_matrix . input_vector + bias_vector
		"""
		
		try:
			# Convert the input vector to numpy array
			input_vector = np.array(input_vector)
			
			# Output vector is obtained by dotting weight and input, then adding with bias
			output_vector = np.add(np.dot(self.weight_matrix, input_vector), self.bias_vector)
			
			# Activate the output
			output_vector = self.__activation_function.calculate_activation(output_vector)
			
			# Add the sensor input
			output_vector = output_vector + np.multiply(self.gain, sensor_input)
			
			# Insert the input_vector and remove the oldest one
			self.__output_matrix = np.insert(self.__output_matrix, 0, output_vector, axis=0)
			self.__output_matrix = np.delete(self.__output_matrix, self.__delay, axis=0)
			
			# Output 
			return self.__output_matrix
			
		except:
			raise ValueError("Please check dimensions of " + self.layer_name)

		
	# Function to update the parameters
	def update_parameters(self, parameter_vector):
		"""
		Load the parameters of the Simple Layer in the form of
		an array / vector
		
		Parameters
		----------
		parameter_vector: array_like
			The parameter vector follows the layout as
			[w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, a_1g, a_1b, a_2g, a_2b, a_3g, a_3b, w_11, w_21, w_31, b_1, ...]
			Here, w_ij implies the weight between ith input node and jth output node. b_i is the bias for the ith output node.
			a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node. 
			
		Returns
		-------
		None
		
		Raises
		------
		ValueException
			The parameter array is shorter than required
			
		Warning
			The parameter array is greater than required
		"""
		# Convert to numpy array
		parameter_vector = np.array(parameter_vector)
	
		# Interval counter maintains the current layer index
		interval_counter = 0
		
		# Get the interval at which the weight and bias seperate
		weight_interval = self.weight_dim[0] * self.weight_dim[1]
		
		# Get the interval at which the bias and next weight vector seperate
		bias_interval = self.bias_dim[0]
		
		# Seperate the weights and bias and then reshape them
		# Numpy raises a None Type Exception, as it cannot reshape a None object
		# If such an excpetion occurs, raise a value error as our parameter_vector
		# is shorter than required
		try:
			self.weight_matrix = parameter_vector[interval_counter:interval_counter + weight_interval].reshape(self.weight_dim)
			interval_counter = interval_counter + weight_interval
			
			self.bias_vector = parameter_vector[interval_counter:interval_counter + bias_interval].reshape(self.bias_dim[0],)
			interval_counter = interval_counter + bias_interval
			
			self.set_activation_parameters(parameter_vector[interval_counter], parameter_vector[interval_counter + 1])
			interval_counter = interval_counter + 2
			
		except:
			raise ValueError("The parameter_vector consists of elements less than required")
			
		# The interval counter should contain the number of elements in parameter_vector
		# Otherwise the user has specified parameters more than required
		# Just a warning is enough
		if(len(parameter_vector) > interval_counter):
			warnings.warn("The parameter vector consists of elements greater than required")
			
	# Function to return the parameters
	def return_parameters(self):
		"""
		Return the parameters of the Simple Layer in the form of
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
		[w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, a_1g, a_1b, a_2g, a_2b, a_3g, a_3b, w_11, w_21, w_31, b_1, ...]
		Here, w_ij implies the weight between ith input node and jth output node. b_i is the bias for the ith output node.
		a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
		"""
		# Initialize the output vector
		# Determine an individual layer's weight matrix in row major form and then it's bias
		# Then concatenate it with the previous output vector
		output = np.array([])
		
		# The vector we get from flattening the weight matrix
		# flatten() works in row major order
		weight_vector = self.weight_matrix().flatten()
		
		# The vector we get from flattening the bias vector
		bias_vector = self.bias_vector().flatten()
		
		# The vector of activation parameters
		activation_vector = np.array(self.get_activation_parameters())
		
		# The output vector is concatenated form of weight_vector, bias_vector and activation_vector
		output = np.concatenate([output, weight_vector, bias_vector, activation_vector])
		
		return output
	
	# Setters and Getters	
	@property
	def weight_matrix(self):
		return self.__weight_matrix
		
	@weight_matrix.setter
	def weight_matrix(self, weight_matrix):
		"""
		Set a user defined weight matrix
		
		Raises a Value Exception if the dimensions do not match
		"""
		if(weight_matrix.shape != self.weight_dim):
			raise ValueError("The dimensions of weight matrix do not match!")
		
		self.__weight_matrix = weight_matrix
		
	@property
	def bias_vector(self):
		return self.__bias_vector
		
	# Function to set the bias vector
	@bias_vector.setter	
	def bias_vector(self, bias_vector):
		"""
		Set a user defined bias vector
		
		Raises a Value Exception if the dimensions do not match
		"""
		if(bias_vector.shape != self.bias_dim):
			raise ValueError("The dimensions of bias vector do not match!")
			
		self.__bias_vector = bias_vector
		
	# Function to return the layer name
	@property
	def layer_name(self):
		""" Get the name of the Layer """
		return self.__layer_name
		
	# Function to return the weight dimensions
	@property
	def weight_dim(self):
		""" Get the dimensions of the weight matrix """
		return self.__weight_dim
		
	# Function return the bias dimensions
	@property
	def bias_dim(self):
		""" Get the dimensions of the bias vector """
		return self.__bias_dim

	# For changing the gain values
	@property
	def gain(self):
		""" Getter for gain """
		return self.__gain
		
	@gain.setter
	def gain(self, gain):
		self.__gain = gain
		
	# Function to set the activation parameters
	def set_activation_parameters(self, beta, theta):
		""" Set the parameters of activation function """
		self.__activation_function.beta = beta
		self.__activation_function.theta = theta
	
	# Function to return the activation parameters(tuple)	
	def get_activation_parameters(self):
		""" Get the parameters of activation function """
		return self.__activation_function.beta, self.__activation_function.theta
		

# CTR Layer ################################################################
class CTRLayer:
	"""
	CTR Layer is used in the Continuous Time Recurrent Neural Network.
	CTR has to save the state of the previous output and then calculate
	the weighted average of the previous and the current output to get
	the total output
	
	...
	
	Attributes
	----------
	input_dim: integer
		The input dimension of the Layer
	
	output_dim: integer
		The output dimension of the layer
	
	activation_function: ActivationFunction class
		SPecifies the activation function to be used
		
	layer_name: string
		THe name of the layer
		
	delay(optional): integer
		Set the delay of the output connection
		
	Methods
	-------
	forward_propagate(input_vector)
		Calculates the output based on the input_vector
		
	update_parameters(parameter_vector)
		Updates the parameters of the layer according to the parameter_vector argument
		
	return_parameters()
		Return the parameters of the layer
		
	Additional Methods
	------------------
	set_activation_parameters(beta, theta)
		Set the parameters of activation function
		
	get_activation_parameters()
		Get the parameters of activation function
		
	"""
	def __init__(self, input_dim, output_dim, activation_function, layer_name, delay = 2):
		# Name of the layer
		self.__layer_name = layer_name
		
		# Weight and bias dimensions
		self.__weight_dim = (output_dim, input_dim)
		self.__bias_dim = (output_dim, )
		self.__time_dim = (output_dim, )
		
		# Initialize the weight and bias
		self.__weight_matrix = np.random.rand(*self.weight_dim)
		self.__bias_vector = np.random.rand(output_dim)
		self.__delay = delay
		
		# Initialize the output matrix
		self.__output_matrix = np.zeros((delay, output_dim))
		
		# Initialize the gain vector
		self.__gain = np.ones((output_dim, ))
		
		# Generate the weights for the weighted average
		self.__time_interval = 1
		self.__time_constant = np.ones((output_dim, ))
		self.__time_weight = np.asarray(float(time_interval) / np.array(time_constant))
		
		# A check for the dimension of time constant list
		if(self.__time_weight.shape != self.__time_dim):
			raise ValueError("The dimension of time constant list is incorrect")
		
		# Set the activation function
		self.__activation_function = activation_function
		# Also check if the activation function is an instance of the ActivationFunction
		if(not hasattr(self.__activation_function, 'calculate_activation')):
			raise TypeError("The activation function needs to has an attribute calculate_activation")
		
		# Set the previous state output, zero for initial
		self.__previous_output = np.zeros(output_dim)
		
	# First order euler step
	# Private function
	def _euler_step(self, input_vector, sensor_input, delay=0):
		"""
		Function that calculates the next step based on the first degree Euler approximation
		
		Parameters
		----------
		input_vector: array_like
			THe input_vector to be passed to the layer
			
		sensor_input: array_like
			The values of the sensor that are to be added(associative layer)
			
		Returns
		-------
		output_matrix: array_like
			The delayed output
			
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
		
		try:
			# Convert to numpy array
			input_vector = np.array(input_vector)
			
			# Get the current activation
			current_activation = np.add(np.dot(self.weight_matrix, input_vector), self.bias_vector)
			current_activation = self.__activation_function.calculate_activation(current_activation)
			current_activation = current_activation + np.multiply(self.gain, sensor_input)
			
			# Generate the current output
			# This equation is the first order euler solution
			current_output = self.__previous_output * (1 - self.__time_weight) + current_activation * self.__time_weight
			
			# Insert the input_vector and remove the oldest one
			self.__output_matrix = np.insert(self.__output_matrix, 0, current_output, axis=0)
			self.__output_matrix = np.delete(self.__output_matrix, self.__delay, axis=0)
			
			# Save it!
			self.__previous_output = current_output
			
			return self.__output_matrix
			
		except:
			raise ValueError("Please check dimensions of " + self.layer_name)
		
	# Just a wrapper function for euler_step to maintain uniformity
	def forward_propagate(input_vector, sensor_input):
		"""
		Function to generate output of the input vector
		Just a wrapper function for euler_step()
		"""
		return euler_step(input_vector, sensor_input)
		
	# Function to update the parameters of the layer
	def update_parameters(self, parameter_vector):
		"""
		Load the parameters of the CTRNN Layer in the form of
		an array / vector
		
		Parameters
		----------
		parameter_vector: array_like
			The parameter vector follows the layout as
			[tc_1, tc_2, tc_3, w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, a_1g, a_1b, a_2g, a_2b, a_3g, a_3b, w_11, w_21, w_31, b_1, ...]
			Here, w_ij implies the weight between ith input node and jth output node. b_i is the bias for the ith output node.
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
		"""
		# Convert to numpy array
		parameter_vector = np.array(parameter_vector)
	
		# Interval counter maintains the current layer index
		interval_counter = 0
		
		# Get the interval at which time constants seperate
		time_interval = self.time_dim[0]
		
		# Get the interval at which weight and bias seperate
		weight_interval = self.weight_dim[0] * self.weight_dim[1]
		
		# Get the interval at which the bias and next weight vector seperate
		bias_interval = self.bias_dim[0]
		
		# Seperate the weights and bias and then reshape them
		# Numpy raises a None Type Exception, as it cannot reshape a None object
		# If such an excpetion occurs, raise a value error as our parameter_vector
		# is shorter than required
		try:
			self.time_constant(parameter_vector[interval_counter:interval_counter + time_interval].reshape(self.time_dim[0], ))
			interval_counter = interval_counter + time_interval
			
			self.weight_matrix(parameter_vector[interval_counter:interval_counter + weight_interval].reshape(self.weight_dim))
			interval_counter = interval_counter + weight_interval
			
			self.bias_vector(parameter_vector[interval_counter:interval_counter + bias_interval].reshape(self.bias_dim[0],))
			interval_counter = interval_counter + bias_interval
			
			self.set_activation_parameters(parameter_vector[interval_counter], parameter_vector[interval_counter + 1])
			interval_counter = interval_counter + 2
			
		except:
			raise ValueError("The parameter_vector consists of elements less than required")
			
		# The interval counter should contain the number of elements in parameter_vector
		# Otherwise the user has specified parameters more than required
		# Just a warning is enough
		if(len(parameter_vector) > interval_counter):
			warnings.warn("The parameter vector consists of elements greater than required")
		
	# Function to return the parameters of a layer
	def return_parameters(self):
		"""
		Return the parameters of the CTRNN Neural Network in the form of
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
		[tc_1, tc_2, tc_3, w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, a_1g, a_1b, a_2g, a_2b, a_3g, a_3b, w_11, w_21, w_31, b_1, ...]
		Here, w_ij implies the weight between ith input node and jth output node. b_i is the bias for the ith output node.
		a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
		tc_i is the time constant of ith neuron of the current layer
		"""
		# Initialize the output vector
		# Determine an individual layer's weight matrix in row major form, it's bias and then activation function parameters
		# Then concatenate it with the previous output vector
		output = np.array([])
		
		# The vector we get from flattening time constants
		time_vector = self.time_constant.flatten()
	
		# The vector we get from flattening the weight matrix
		# flatten() works in row major order
		weight_vector = self.weight_matrix().flatten()
		
		# The vector we get from flattening the bias vector
		bias_vector = self.bias_vector().flatten()
		
		# The vector of activation parameters
		activation_vector = np.array(self.get_activation_parameters())
		
		# The output vector is concatenated form of time_vector, weight_vector, bias_vector and activation_vector
		output = np.concatenate([output, time_vector, weight_vector, bias_vector, activation_vector])
		
		return output
		
	# Function to return the weight matrix
	@property
	def weight_matrix(self):
		""" Function used to return the weight matrix """
		return self.__weight_matrix
		
	# Function to set the weight matrix
	@weight_matrix.setter
	def set_weight_matrix(self, weight_matrix):
		"""
		Sets the weight matrix to be used by the layer
		Raise a value exception if dimensions do not match
		"""
		if(weight_matrix.shape != self.weight_dim):
			raise ValueError("The dimensions of the weight matrix do not match")
		self.__weight_matrix = weight_matrix
		
	# Function to return the bias vector
	@property
	def bias_vector(self):
		""" Function to return the bias vector """
		return self.__bias_vector
		
	# Function to set the bias vector
	@bias_vector.setter	
	def bias_vector(self, bias_vector):
		"""
		Sets the bias vector to be used by the layer
		Raise a value exception if dimensions do not match
		"""
		if(bias_vector.shape != self.bias_dim):
			raise ValueError("The dimensions of the bias vector do not match!")
		self.__bias_vector = bias_vector
		
	# Function to return the layer name
	@property
	def layer_name(self):
		""" Function to return the name of the layer """
		return self.__layer_name
		
	# Function to return the weight dimensions
	@property
	def weight_dim(self):
		""" Function to return the dimensions of the weight matrix """
		return self.__weight_dim
		
	# Function return the bias dimensions
	@property
	def bias_dim(self):
		""" Function to return the dimensions of the bias vector """
		return self.__bias_dim
		
	# Function to return the time constant dimension
	@property
	def time_dim(self):
		""" Function to return the dimensions of the time constant vector """
		return self.__time_dim
		
	@property
	def time_interval(self):
		return self.__time_interval
		
	@time_interval.setter
	def time_interval(self, time_interval):
		self.__time_interval = time_interval
		
		# Change the time weight as well
		self.__time_weight = np.asarray(float(self.time_interval) / self.time_constant)
	
	# Function to return the time constant list
	@property
	def time_constant(self):
		""" Function to return the list of time constants """
		return self.__time_constant
		
	# Function to set the time constant list
	@time_constant.setter
	def time_constant(self, time_constant):
		"""
		Function to set the time constant list of neurons
		Raises an excpetion if the dimensions do not match!
		"""
		self.time_constant = np.array(time_constant)
		
		if(self.time_constant.shape != self.time_dim):
			raise ValueError("The dimension of time constant list is not correct!")
			
		# Calculate the weights based on the time constants and the time interval!
		self.__time_weight = np.asarray(float(self.time_interval) / self.time_constant)
		
	# For changing the gain values
	@property
	def gain(self):
		""" Getter for gain """
		return self.__gain
		
	@gain.setter
	def gain(self, gain):
		self.__gain = gain
		
	# Function to set the activation parameters
	def set_activation_parameters(self, beta, theta):
		""" Set the parameters of activation function """
		self.__activation_function.beta = beta
		self.__activation_function.theta = theta
	
	# Function to return the activation parameters(tuple)	
	def get_activation_parameters(self):
		""" Get the parameters of activation function """
		return self.__activation_function.beta, self.__activation_function.theta
