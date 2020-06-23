"""Docstring for the activation_functions.py module.

This is a library of activation functions that the user can choose from!
These are passed as a parameter to the neural network. The following
activation functions are present in this library

- Identity Activation
- Linear Activation
- Step Activation
- Sigmoid Activation
- Hyperbolic Tangent Activation
- ReLU Activation
- Maximum Value Activation

References
https://www.geeksforgeeks.org/activation-functions/
https://miro.medium.com/max/1192/1*4ZEDRpFuCIpUjNgjDdT2Lg.png
http://www.robolabo.etsit.upm.es/asignaturas/irin/transparencias/redesNeuronales.pdf


To generate your own activation function, inherit from the Activation
Function class and put the function calculation procedure inside
the calculate_activation() method

"""

# Import the numpy module
import numpy as np


class ActivationFunction(object):
	"""
    Class of Activation Function

	...
	Parameters
	----------
	beta(optional) : float / array_like
	    The gain of the activation function
	theta(optional) : float / array_like
		The bias/offset of the activation function
		
	* An array can be used for beta and theta, if we want different activation for different neuron

	Attributes
	----------
	beta(optional) : float / array_like
	    The gain of the activation function
	theta(optional) : float / array_like
		The bias/offset of the activation function
		
	* An array can be used for beta and theta, if we want different activation for different neuron

	Methods
	-------
	calculate_activation(input_vector)
	    Calculate the activation of an input vector
	    
	"""  
	def __init__(self, beta = 1, theta = 0):
		"""
		Initialization function of ActivationFunction class		
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
		# Set the protected class variables
		self.beta = beta
		self.theta = theta
		
	@property
	def beta(self):
		""" Beta parameter defining the gain"""
		return self._beta
		
	@beta.setter
	def beta(self, beta):
		self._beta = np.array(beta)
		
	@property
	def theta(self):
		""" Theta parameter defining the offset"""
		return self._theta
		
	@theta.setter
	def theta(self, theta):
		self._theta = np.array(theta)
		
	def calculate_activation(self, x):
		# Different implementation of different functions
		pass


# The identity actvation class
class IdentityActivation(ActivationFunction):
	# Inherit the docstring of parent class
	__doc__ = ActivationFunction.__doc__
	
	def calculate_activation(self, x):
		""" Returns the input as it is """
		
		# Calculate the argument
		argument = x
		
		return argument

# The linear activation class
class LinearActivation(ActivationFunction):
	# Inherit the docstring of parent class
	__doc__ = ActivationFunction.__doc__

	def calculate_activation(self, x):
		""" https://en.wikipedia.org/wiki/Identity_function """
		
		# Calculate the argument of the function
		argument = self.beta * (x - self.theta)
		
		# Return as it is
		return argument

# The step activation class
class StepActivation(ActivationFunction):
	# Inherit the docstring of parent class
	__doc__ = ActivationFunction.__doc__

	def calculate_activation(self, x):
		""" https://en.wikipedia.org/wiki/Heaviside_step_function """
		
		# The parameter beta is of no use, hence recalculate the argument
		argument = x - self.theta

		# Replace all the elements greater than or equal to 0
		argument[argument >= 0] = 1

		# Replace all the elements less than 0
		argument[argument < self.theta] = 0
		
		return argument
		
# The sigmoidal activation class
class SigmoidActivation(ActivationFunction):
	# Inherit the docstring of parent class
	__doc__ = ActivationFunction.__doc__

	def calculate_activation(self, x):
		""" https://en.wikipedia.org/wiki/Logistic_function """
		
		# Calculate the argument of the function
		argument = self.beta * (x - self.theta)
		
		# According to the formula
		z = 1 / (1 + np.exp(-argument))
		
		return z
	
# Hyperbolic Tangent activation class
class TanhActivation(ActivationFunction):
	# Inherit the docstring of parent class
	__doc__ = ActivationFunction.__doc__

	def calculate_activation(self, x):
		""" https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent """
		
		# Calculate the argument of the function
		argument = self.beta * (x - self.theta)
		
		# According to the formula
		z = np.tanh(argument)
		
		return z
	
# Rectified linear unit activation class
class ReluActivation(ActivationFunction):
	# Inherit the docstring of parent class
	__doc__ = ActivationFunction.__doc__

	def calculate_activation(self, x):
		""" https://en.wikipedia.org/wiki/Rectifier_(neural_networks) """
		
		# Calculate the argument of the function
		argument = self.beta * (x - self.theta)
		
		# Replace all elements less than 0
		argument[argument < 0] = 0
		
		# The others remain as such!

		return argument
		
# Maximum activation class
class MaximumActivation(ActivationFunction):
	# Inherit the docstring of parent class
	__doc__ = ActivationFunction.__doc__

	def calculate_activation(self, x):
		""" 
		Finds the maximum element from the input vector.
		Beta and Theta parameters are not used in this activation
		
		"""
	
		# Get the maximum element from x
		maximum = np.max(x)
		
		# Make others equal to zero
		x[x != maximum] = 0
		
		# Make the maximum equal to 1
		x[x == maximum] = 1
		
		return x
		


