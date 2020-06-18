import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
import numpy as np
from neural_networks.activation_functions import LinearActivation

# ANN with 4 Layers, 2 are hidden and the other 2 are input and output
# The hidden layers take input from the input layer and output them to the output layer
# The hidden layers are connected with each other
# The output of one hidden layer is dependent on the previous output of the other hidden layer and vice versa

print("Static Recurrent ANN: First Way (Correct Way)")
nn = ArtificialNeuralNetwork([
								[2, 0, None, [], [1, 2]], 											# Layer 0 (Input Layer)
								[1, 1, LinearActivation(), [(0, False), (2, False)], [2, 3]], 		# Layer 1 (Hidden Layer)
								[1, 1, LinearActivation(), [(0, False), (1, True)], [1, 3]], 		# Layer 2 (Hidden Layer)
								[1, 1, LinearActivation(), [(1, False), (2, False)], []]			# Layer 3 (Output Layer)
							])

# Loading the parameters from a list
parameter_vector = [
						[], 						# Parameters for Layer 0
						[1, 1, 1, 0, 1, 0], 		# Parameters for Layer 1
						[1, 1, 1, 0, 1, 0], 		# Parameters for Layer 2
						[1, 1, 0, 1, 0]				# Parameters for Layer 3
				   ]
nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
input_dict = {
				0: np.array([1.0, 1.0])				# Input for Layer 0
			 }
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)

# The difference lies where we have put the delay between Layer 1 and Layer 2
print("Static Recurrent ANN: Second Way (Wrong Way)")
nn = ArtificialNeuralNetwork([
								[2, 0, None, [], [1, 2]], 											# Layer 0
								[1, 1, LinearActivation(), [(0, False), (2, True)], [2, 3]], 		# Layer 1 (The connection input from Layer 2 is delayed)
								[1, 1, LinearActivation(), [(0, False), (1, False)], [1, 3]], 		# Layer 2 (The connection input from Layer 1 is not delayed)
								[1, 1, LinearActivation(), [(1, False), (2, False)], []]			# Layer 3
							])

# Loading the parameters from a list
parameter_vector = [
						[], 						# Parameters for Layer 0
						[1, 1, 1, 0, 1, 0], 		# Parameters for Layer 1
						[1, 1, 1, 0, 1, 0], 		# Parameters for Layer 2
						[1, 1, 0, 1, 0]				# Parameters for Layer 3
				   ]
nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
input_dict = {
				0: np.array([1.0, 1.0])				# Input for Layer 0
			 }
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)
