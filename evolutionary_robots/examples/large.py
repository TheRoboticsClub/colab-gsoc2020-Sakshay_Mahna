# An example showing the creation of a Large Neural Network

import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
import numpy as np
from neural_networks.activation_functions import LinearActivation, IdentityActivation

print("Large Recurrent Example")
# There are 30 layers in this network
# There are 28 hidden layers and 2 input and output respectively
# All the Layers have a single neuron each
# The 28 hidden layers take input from the input layer and give that to the output layer

# In order to make large number of layers we use a loop structure
layers = [None] * 30

# The format followed is Layer(name_of_layer, number_of_neurons, type_of_layer, activation_function, sensor_input, [list_of_output_connections])
layers[0] = Layer("layer0", 1, "STATIC", IdentityActivation(), "SENSOR", ["layer" + str(i) for i in range(1, 29)])

# Defining Layers from 1 to 28 using loop
for i in range(1, 29):
	layers[i] = Layer("layer" + str(i), 1, "STATIC", LinearActivation(), "", ["layer29"])
	
# Defining the final layer as CTRNN with 2 neurons
layers[29] = Layer("layer29", 2, "DYNAMIC", LinearActivation(), "", ["ACTUATORS"])

# Assuming we want 1 neuron for the output and convert to Simple Layer, the structure can be changed as
layers[29].type_of_layer = "STATIC"
layers[29].number_of_neurons = 1

# Constructing the Network
nn = ArtificialNeuralNetwork(layers)		# Simply passing the layers 2d matrix

# Considering we do not change the weights

# Input the Neural Network through a dictionary
input_dict = {
		"SENSOR": np.array([1.0])		# Input to Layer 0
	     }
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)

