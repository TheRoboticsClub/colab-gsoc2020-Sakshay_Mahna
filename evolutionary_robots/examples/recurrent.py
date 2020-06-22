import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
import numpy as np
from neural_networks.activation_functions import LinearActivation, IdentityActivation

# ANN with 4 Layers, 2 are hidden and the other 2 are input and output
# The hidden layers take input from the input layer and output them to the output layer
# The hidden layers are connected with each other
# The output of one hidden layer is dependent on the previous output of the other hidden layer and vice versa

inputLayer = Layer("inputLayer", 2, "STATIC", IdentityActivation(), "SENSOR", ["hiddenLayer1", "hiddenLayer2"])				# Input Layer
hiddenLayer1 = Layer("hiddenLayer1", 1, "STATIC", LinearActivation(), "", ["hiddenLayer2", "outputLayer"])		# Hidden Layer 1
hiddenLayer2 = Layer("hiddenLayer2", 1, "STATIC", LinearActivation(), "", ["hiddenLayer1", "outputLayer"])		# Hidden Layer 2
outputLayer = Layer("outputLayer", 1, "STATIC", LinearActivation(), "", ["MOTOR"])		# Output Layer

# Delay the input connection of Hidden Layer 2 that is received from Hidden Layer 1
hiddenLayer1.delayed_connections = ["hiddenLayer2"]

print("Static Recurrent ANN: First Way (Correct Way)")
nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0 (Input Layer)
				hiddenLayer1, 		# Layer 1 (Hidden Layer)
				hiddenLayer2, 		# Layer 2 (Hidden Layer)
				outputLayer		# Layer 3 (Output Layer)
			     ])

# Loading the parameters from a list
parameter_vector = [
			[], 				# Parameters for Layer 0
			[1, 1, 1, 1, 0], 		# Parameters for Layer 1
			[1, 1, 1, 1, 0], 		# Parameters for Layer 2
			[1, 1, 1, 0]			# Parameters for Layer 3
		   ]
nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
input_dict = {
		"SENSOR": np.array([1.0, 1.0])		# Input for Layer 0
	     }
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)

####################################################
# The difference lies where we have put the delay between Layer 1 and Layer 2
# Remove the delay from Hidden Layer 2 and put it in Hidden Layer 1
hiddenLayer1.delayed_connections = []
hiddenLayer2.delayed_connections = ["hiddenLayer1"]

print("Static Recurrent ANN: Second Way (Wrong Way)")
nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0
				hiddenLayer1, 		# Layer 1 (The connection input from Layer 2 is delayed)
				hiddenLayer2, 		# Layer 2 (The connection input from Layer 1 is not delayed)
				outputLayer		# Layer 3
			     ])

# Loading the parameters from a list
parameter_vector = [
			[], 				# Parameters for Layer 0
			[1, 1, 1, 1, 0], 		# Parameters for Layer 1
			[1, 1, 1, 1, 0], 		# Parameters for Layer 2
			[1, 1, 1, 0]			# Parameters for Layer 3
		    ]
nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
input_dict = {
		"SENSOR": np.array([1.0, 1.0])		# Input for Layer 0
	     }
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)
