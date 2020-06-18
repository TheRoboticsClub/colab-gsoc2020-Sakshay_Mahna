import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
import numpy as np
from neural_networks.activation_functions import LinearActivation

print("Static ANN")
# Simple ANN with 2 input 3 output and Linear Activation
nn = ArtificialNeuralNetwork([
				[2, 0, None, [], [1]],					# Layer 0
				[3, 1, LinearActivation(), [(0, False)], []]		# Layer 1
			     ])

# Loading the parameters from a list
parameter_vector = [
			[],							# Layer 0 
			[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, 1, 0]		# Layer 1
		   ]

nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
input_dict = {
		0: np.array([1.0, 1.0])		# Input for Layer 0
	     }
			 
output = nn.forward_propagate(input_dict)
print(output)


########################################################################
print("CTRNN")
# CTRNN with 2 input 3 output and Linear Activation
# And time constants are 1, 1, 1
nn = ArtificialNeuralNetwork([
				[2, 0, None, [], [1]],					# Layer 0
				[3, 2, LinearActivation(), [(0, False)], []]		# Layer 1
			     ])

# Loading the parameters from a list
parameter_vector = [
			[], 								# Layer 0
			[1, 1, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, 1, 0]		# Layer 1
		   ]
				   
nn.load_parameters_from_vector(parameter_vector)


# Input the Neural Network through a dictionary
input_dict = {
		0: np.array([1.0, 1.0])			# Input for Layer 0
	     }

output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)



