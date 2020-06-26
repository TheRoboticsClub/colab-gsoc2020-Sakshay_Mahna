# A complex example with more number of layers
# Including Input Layer, Hidden Layer, Associative Layer and Output Layer

import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
import numpy as np
from neural_networks.activation_functions import LinearActivation, IdentityActivation

print("Complex Recurrent Example")
# There are 5 Layers in this network
# The input layer consists of 2 neurons, takes input from SENSOR1 and outputs to hiddenLayer1
# The hiddenLayer1 consists of 2 neurons with Linear Activation and outputs to hiddenLayer2, hiddenLayer3 and outputLayer
# THe hiddenLayer2 consists of 2 neurons with LinearActivation and outputs to outputLayer
# The hiddenLayer3 consists of 2 neurons with LinearActivation, takes input from SENSOR2 and outputs to outputLayer(in essence an associative layer)
# The outputLayer consists of 1 neuron with LinearActivation and outputs to hiddenLayer1(recurrent connection) and MOTORLEFT


# The format followed is Layer(name_of_layer, number_of_neurons, type_of_layer, activation_function, sensor_input, [list_of_output_connections])
inputLayer = Layer("inputLayer", 2, "STATIC", IdentityActivation(), "SENSOR1", ["hiddenLayer1"])
hiddenLayer1 = Layer("hiddenLayer1", 2, "STATIC", LinearActivation(), "", ["hiddenLayer2", "hiddenLayer3", "outputLayer"])
hiddenLayer2 = Layer("hiddenLayer2", 2, "STATIC", LinearActivation(), "", ["outputLayer"])
hiddenLayer3 = Layer("hiddenLayer3", 2, "STATIC", LinearActivation(), "SENSOR2", ["outputLayer"])
outputLayer = Layer("outputLayer", 1, "STATIC", LinearActivation(), "", ["hiddenLayer1", "MOTORLEFT"])

# Add the recurrent connection
outputLayer.delayed_connections = ["hiddenLayer1"]

nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0 (Input Layer)
				hiddenLayer1, 		# Layer 1 (Hidden Layer 1)
				hiddenLayer2, 		# Layer 2 (Hidden Layer 2)
				hiddenLayer3, 		# Layer 3 (Hidden Layer 3)
				outputLayer		# Layer 4 (Output Layer)
			     ])
			     
nn.visualize('repr/complex', True)

# Weights are such that the recurrence is not able to show itself!
# Loading the parameters from a list
parameter_vector = [
			[], 					# Parameters for Layer 0
			[1, 1, 0, 1, 1, 0, 1, 1, 0, 0], 	# Parameters for Layer 1 (The weights of the recurrence with Layer 4 are taken as 0)
			[1, 1, 1, 1, 1, 1, 0, 0], 		# Parameters for Layer 2 
			[1, 1, 1, 1, 1, 1, 0, 0], 		# Parameters for Layer 3
			[1, 1, 1, 1, 1, 1, 1, 0], 		# Parameters for Layer 4
		   ]
nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
input_dict = {
		"SENSOR1": np.array([1.0, 1.0])		# Input to SENSOR1
	     }
	     
# By default the input to SENSOR2 is taken as 0
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)

# Input the Neural Network through a dictionary
input_dict = {
		"SENSOR1": np.array([1.0, 1.0]), 	# Input for SENSOR1
		"SENSOR2": np.array([1.0, 1.0])		# Associative Input for SENSOR2
	     }
output = nn.forward_propagate(input_dict)
print(output)

