# The Layers can be passed to the ANN class in any order
# The Neural Network will automatically generate the layers
# by looking at the connections of the layers specified in the interface

import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
import numpy as np
from neural_networks.activation_functions import LinearActivation, IdentityActivation

# An ANN with 3 Layers, which are input, hidden and output
# The input layer consists of 2 neurons, takes input from SENSOR and outputs to hiddenLayer and outputLayer
# The hidden layer consists of 2 neurons, with a linear activation and outputs to outputLayer
# The output layer consists of 2 neurons with linear activation and outputs to GRIPPERS


# The format followed is Layer(name_of_layer, number_of_neurons, type_of_layer, activation_function, sensor_input, [list_of_output_connections])
inputLayer = Layer("inputLayer", 2, "STATIC", IdentityActivation(), "SENSOR", ["hiddenLayer", "outputLayer"])
hiddenLayer = Layer("hiddenLayer", 2, "STATIC", LinearActivation(), "", ["outputLayer"])
outputLayer = Layer("outputLayer", 2, "STATIC", LinearActivation(), "", ["GRIPPERS"])

print("Static ANN example of order")
# The order entered here is not correct
# Yet the connections and output generated will be the same
nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0 (Input Layer)
				outputLayer, 		# Layer 1 (Output Layer)
				hiddenLayer		# Layer 2 (Hidden Layer)
			     ])

# Visualize the network
# Both the visualizations will be the same
nn.visualize('repr/order1', True)

# Loading the parameters from a list			
parameter_vector = [
			[], 								# Parameters for Layer 0
			[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1, 0, 0], 		# Parameters for Layer 1
			[1, 1, 1, 1, 1, 1, 0, 0]					# Parameters for Layer 2
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


###################################################################
print("Static ANN example of order")
# Different order of initialization, which is the correct one
nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0 (Input Layer)
				hiddenLayer, 		# Layer 1 (Hidden Layer)
				outputLayer		# Layer 2 (Output Layer)
			     ])

# Visualize the network
# Both the visualizations will be the same
nn.visualize('repr/order2', True)
					
# Loading the parameters from a list
parameter_vector = [
			[], 							# Parameters for Layer 0
			[1, 1, 1, 1, 1, 1, 0, 0], 				# Parameters for Layer 1
			[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1, 0, 0]	# Parameters for Layer 2
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
