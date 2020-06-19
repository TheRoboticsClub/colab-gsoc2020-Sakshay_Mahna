import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
import numpy as np
from neural_networks.activation_functions import LinearActivation

print("Static ANN")
# Simple ANN with 2 input 3 output and Linear Activation
inputLayer = Layer(2, 0, None, [], [1])			# Input Layer
outputLayer = Layer(3, 1, LinearActivation(), [0], [])	# Output Layer

nn = ArtificialNeuralNetwork([
				inputLayer,		# Layer 0
				outputLayer		# Layer 1
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
# The input Layer is same, therefore no changes
# The output layer type is changed, 
outputLayer.type_of_layer = 2
nn = ArtificialNeuralNetwork([
				inputLayer,		# Layer 0
				outputLayer		# Layer 1
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



