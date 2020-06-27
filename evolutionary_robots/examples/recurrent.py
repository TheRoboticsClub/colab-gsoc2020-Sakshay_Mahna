# This example shows the use of delayed connections
# Delayed connections are useful in the case where we have recurrent connections

import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
import numpy as np
from neural_networks.activation_functions import LinearActivation, IdentityActivation

# ANN with 4 Layers, 2 are hidden and the other 2 are input and output
# The input layer consists of 2 neurons, takes input from SENSOR and gives output to hiddenLayer1 and hiddenLayer2
# Both the hidden layers are connected to each other and the outputLayer, they have 1 neuron each with Linear activation
# The outputLayer contains 1 neuron with Linear activation and gives output to MOTOR actuator

# The format followed is Layer(name_of_layer, number_of_neurons, activation_function, sensor_input, [list_of_output_connections])
inputLayer = Layer("inputLayer", 2, IdentityActivation(), "SENSOR", ["hiddenLayer1", "hiddenLayer2"])		# Input Layer
hiddenLayer1 = Layer("hiddenLayer1", 1, LinearActivation(), "", ["hiddenLayer2", "outputLayer"])		# Hidden Layer 1
hiddenLayer2 = Layer("hiddenLayer2", 1, LinearActivation(), "", ["hiddenLayer1", "outputLayer"])		# Hidden Layer 2
outputLayer = Layer("outputLayer", 1, LinearActivation(), "", ["MOTOR"])					# Output Layer

print("Static Recurrent ANN")
nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0 (Input Layer)
				hiddenLayer1, 		# Layer 1 (Hidden Layer)
				hiddenLayer2, 		# Layer 2 (Hidden Layer)
				outputLayer		# Layer 3 (Output Layer)
			     ], "DYNAMIC")
			     
# Visualize the network
nn.visualize('repr/recurrent')

# Loading the parameters from a list
parameter_vector = [
			[], 				# Parameters for Layer 0
			[1, 1, 1, 1, 1, 0], 		# Parameters for Layer 1
			[1, 1, 1, 1, 1, 0], 		# Parameters for Layer 2
			[1, 1, 1, 1, 0]			# Parameters for Layer 3
		   ]
nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
input_dict = {
		"SENSOR": np.array([1.0, 1.0])		# Input for SENSOR
	     }
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)

