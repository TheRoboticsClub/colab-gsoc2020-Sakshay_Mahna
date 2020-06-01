# Tests for activation functions

from neural_networks import activation_functions
import numpy as np


# Linear Function
print("Linear Function: ")
a = np.array([1, -1, 0.1, -2.1])
b = activation_functions.linear_function(a)
print(b)

# Step Function
print("Step Function: ")
a = np.array([1, -1, 0.1, -2.1])
b = activation_functions.step_function(a)
print(b)

# Sigmoid Function
print("Sigmoid Function: ")
a = np.array([1, -1, 0.1, -2.1])
b = activation_functions.sigmoid_function(a)
print(b)

# Tanh Function
print("Tanh Function: ")
a = np.array([1, -1, 0.1, -2.1])
b = activation_functions.tanh_function(a)
print(b)

# RELU function
print("ReLu Function: ")
a = np.array([1, -1, 0.1, -2.1])
b = activation_functions.relu_function(a)
print(b)

# Leaky Relu function
print("Leaky Relu function: ")
a = np.array([1, -1, 0.1, -2.1])
b = activation_functions.leaky_relu_function(a)
print(b)

# Maximum Function
print("Maximum function: ")
a = np.array([1, -1, 0.1, -2.1])
b = activation_functions.maximum_function(a)
print(b)




