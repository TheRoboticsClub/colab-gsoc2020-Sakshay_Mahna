import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
import numpy as np
from neural_networks.activation_functions import LinearActivation

print("Complex Recurrent Example")
# There are 5 Layers in this network
# The output of a hidden layer depends only on the input layer
# 2 other hidden layers depend on the output of the previous hidden layer
# The 2 hidden layers are then connected to another hidden layer
# That final hidden layer is connected with the first hidden layer as a recurrent connection
# The output layer is then connected to the final hidden layer
nn = ArtificialNeuralNetwork([
				[2, 0, None, [], [1]], 																				# Layer 0 (Input Layer)
				[2, 1, LinearActivation(), [(0, False), (4, False)], [2, 3, 4]], 				# Layer 1 (Hidden Layer 1)
				[2, 1, LinearActivation(), [(1, False)], [4]], 							# Layer 2 (Hidden Layer 2)
				[2, 1, LinearActivation(), [(1, False)], [4]], 							# Layer 3 (Hidden Layer 3)
				[1, 1, LinearActivation(), [(2, False), (1, False), (3, False)], [1, 5]], 			# Layer 4 (Hidden Layer 4)
				[1, 1, LinearActivation(), [(4, False)], []]							# Layer 5 (Output Layer)
			     ])

# Weights are such that the recurrence is not able to show itself!
# Loading the parameters from a list
parameter_vector = [
			[], 						# Parameters for Layer 0
			[1, 1, 0, 1, 1, 0, 0, 0, 1, 0], 		# Parameters for Layer 1 (The weights of the recurrence with Layer 4 are taken as 0)
			[1, 1, 1, 1, 0, 0, 1, 0], 			# Parameters for Layer 2 
			[1, 1, 1, 1, 0, 0, 1, 0], 			# Parameters for Layer 3
			[1, 1, 1, 1, 1, 1, 0, 1, 0], 			# Parameters for Layer 4
			[1, 0, 1, 0]					# Parameters for Layer 5
		   ]
nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
input_dict = {
		0: np.array([1.0, 1.0])			# Input to Layer 0
	     }
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)

# Set the gain of Layer 3 according to the number of neurons in that layer
nn.set_gain(3, [2, 2])

# Input the Neural Network through a dictionary
input_dict = {
		0: np.array([1.0, 1.0]), 		# Input to Layer 0
		3: np.array([1.0, 1.0])			# Associative Input to Layer 3
	     }
output = nn.forward_propagate(input_dict)
print(output)

