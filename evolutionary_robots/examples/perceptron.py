# This example shows the basic usage of the API
# As to how to generate a Static Perceptron and a Dynamic Perceptron

import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
import numpy as np
from neural_networks.activation_functions import LinearActivation, IdentityActivation

print("Static ANN")
# Simple ANN with 2 Layers, both of them static
# The input layer consists of 2 neurons and takes input from CAMERA sensor
# The output layer consists of 3 neurons with Linear activation and gives output to MOTOR actuators

# The format followed is Layer(name_of_layer, number_of_neurons, activation_function, sensor_input, [list_of_output_connections])
inputLayer = Layer("inputLayer", 			# Name of the Layer
		   2,					# Number of neurons
		   IdentityActivation(), 		# Activation Function(Input Layers should use IdentityActivation function)
		   "CAMERA", 				# The sensor from which input is taken
		   ["outputLayer"]			# The list of output connections
		   )			

				 	
outputLayer = Layer("outputLayer",			# Name of the Layer
		    3,					# Number of neurons
		    LinearActivation(), 		# Activation Function
		    "", 				# The sensor from which input is taken(no sensor input in this case)
		    ["MOTOR"]				# The list of output connections(this is connected to MOTOR actuator)
		    )	

nn = ArtificialNeuralNetwork([
				inputLayer,		# Layer 0
				outputLayer		# Layer 1
			     ], "STATIC")	# Type of Network
	
# Visualize the network		     
nn.visualize('repr/static_perceptron', True)

# Loading the parameters from a list
parameter_vector = [
			[],						# Layer 0 (Any parameters given to input layers are ignored, therefore do not specify) 
			[						# Layer 1
				0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 		# The weight matrix
				1, 1, 1, 				# The gain of the activation
				-1, -1, -1				# The bias of the activation
			]		
		   ]

nn.load_parameters_from_vector(parameter_vector)

# Input the Neural Network through a dictionary
# The input dictionary is keyed according to the sensor inputs provided for the network
input_dict = {
		"CAMERA": np.array([1.0, 1.0])		# Input for the Sensor
	     }
			 
output = nn.forward_propagate(input_dict)
print(output)


########################################################################
print("CTRNN")
# CTRNN with 2 layers
# The input layer consists of 2 neurons and takes input from CAMERA sensor, input layer is always considered as static
# The output layer consists of 3 neurons with Linear activation and gives output to MOTOR actuators

# CTRNN is a Dynamic Network
nn = ArtificialNeuralNetwork([
				inputLayer,		# Layer 0
				outputLayer		# Layer 1
			     ], "DYNAMIC")	# CTRNN

# Loading the parameters from a list
parameter_vector = [
			[], 					# Layer 0
			[					# Layer 1
			 1, 1, 1,				# Time Constants
			 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,		# Weight Matrix
			 1, 1, 1,				# Gain of activation
			 -1, -1, -1				# Bias of activation
			]		
		   ]
				   
nn.load_parameters_from_vector(parameter_vector)

# Visualize the network
nn.visualize('repr/dynamic_perceptron', True)

# Input the Neural Network through a dictionary
input_dict = {
		"CAMERA": np.array([1.0, 1.0])			# Input for Sensor
	     }

output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)



