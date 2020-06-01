# This is a library of activation functions that the user can choose from!
# These are passed as a parameter to the neural network, hence can be
# self designed by the user as well!

# References
# https://www.geeksforgeeks.org/activation-functions/
# https://miro.medium.com/max/1192/1*4ZEDRpFuCIpUjNgjDdT2Lg.png
import numpy as np


# The linear activation function
def linear_function(x):
	# Return as it is
	return x

# If the input is less than 0, the output is 0
# otherwise the output is 1
def step_function(x):
	# Replace all the elements greater than or equal to 0
	x[x >= 0] = 1

	# Replace all the elements less than 0
	x[x < 0] = 0
	
	return x
		
# The sigmoidal activation function
def sigmoid_function(x):
	# According to the formula
	z = 1 / (1 + np.exp(-x))
	
	return z
	
# Hyperbolic Tangent activation function
def tanh_function(x):
	# According to the formula
	z = np.tanh(x)
	
	return z
	
# Rectified linear unit activation function
def relu_function(x):
	# Replace all elements less than 0
	x[x < 0] = 0
	
	# The others remain as such!
	
	return x
	
# The leaky relu activation function
def leaky_relu_function(x, a=0.1):
	# Replace all elements less than 0
	x[x < 0] = a * x[x < 0]
	
	# The others remain as such!
	
	return x
		
# Maximum function
def maximum_function(x):
	# Get the maximum element from x
	maximum = np.max(x)
	
	# Make others equal to zero
	x[x != maximum] = 0
	
	# Make the maximum equal to 1
	x[x == maximum] = 1
	
	return x
		


