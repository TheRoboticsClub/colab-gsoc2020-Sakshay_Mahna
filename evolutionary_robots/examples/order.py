# This example shows the concept of order of execution
# The Layers should be passed to the ANN class in correct sequence
# Otherwise the outputs will not be as desired

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

print("Static ANN example of order: Wrong Way")
# The order entered here is not correct
# This may result in wrong results
nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0 (Input Layer)
				outputLayer, 		# Layer 1 (Output Layer)
				hiddenLayer		# Layer 2 (Hidden Layer)
			     ])

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
print("Static ANN example of order: Correct Way")
# Different order of initialization, which results in a different order of initialization
nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0 (Input Layer)
				hiddenLayer, 		# Layer 1 (Hidden Layer)
				outputLayer		# Layer 2 (Output Layer)
			     ])

nn.visualize('repr/order', True)
					
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
