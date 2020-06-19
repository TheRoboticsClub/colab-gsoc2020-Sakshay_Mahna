import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
import numpy as np
from neural_networks.activation_functions import LinearActivation

# An ANN with 3 Layers, which are input, hidden and output
# The output layer depends on the output of input and hidden layers
inputLayer = Layer(2, 0, None, [], [1, 2])
hiddenLayer = Layer(2, 1, LinearActivation(), [0], [1])		# Connections are made according to the initialization of Neural Network object
outputLayer = Layer(2, 1, LinearActivation(), [0, 2], [])

print("Static ANN example of order: Wrong Way")
nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0 (Input Layer)
				outputLayer, 		# Layer 1 (Output Layer)
				hiddenLayer		# Layer 2 (Hidden Layer)
			     ])

# Loading the parameters from a list			
parameter_vector = [
			[], 								# Parameters for Layer 0
			[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 0, 1, 0], 		# Parameters for Layer 1
			[1, 1, 1, 1, 0, 0, 1, 0]					# Parameters for Layer 2
		    ]
nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
input_dict = {
		0: np.array([1.0, 1.0])		# Input to Layer 0
	     }
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)


##########################################################
print("Changed order of execution")
# Same initialization, but changing the order of execution before calculating the output
nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0 (Input Layer)
				outputLayer, 		# Layer 1 (Output Layer)
				hiddenLayer		# Layer 2 (Hidden Layer)
			      ])

# Loading the parameters from a list							
parameter_vector = [
			[], 								# Parameters for Layer 0
			[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 0, 1, 0], 		# Parameters for Layer 1
			[1, 1, 1, 1, 0, 0, 1, 0]					# Parameters for Layer 2
		   ]
nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
input_dict = {
		0: np.array([1.0, 1.0])		# Input to Layer 0
	     }

# The current order of execution is according to the order as specified during iniitialization of the ANN
# Layer 0 -> Layer 1 -> Layer 2
nn.order_of_execution = [0, 2, 1]			# This changes the order of execution, Layer 0 -> Layer 2 -> Layer 1
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)



###################################################################
print("Static ANN example of order: Correct Way")
# Different initialization, which results in a different order of initialization
hiddenLayer.output_connections = [2]
outputLayer.input_connections = [0, 1]
nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0 (Input Layer)
				hiddenLayer, 		# Layer 1 (Hidden Layer)
				outputLayer		# Layer 2 (Output Layer)
			     ])
							
# Loading the parameters from a list
parameter_vector = [
			[], 							# Parameters for Layer 0
			[1, 1, 1, 1, 0, 0, 1, 0], 				# Parameters for Layer 1
			[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 0, 1, 0]	# Parameters for Layer 2
		   ]
nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
input_dict = {
		0: np.array([1.0, 1.0])		# Input to Layer 0
	     }
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)
