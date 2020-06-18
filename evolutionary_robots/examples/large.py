import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
import numpy as np
from neural_networks.activation_functions import LinearActivation, SigmoidActivation

print("Large Recurrent Example")
# There are 30 layers in this network
# There are 28 hidden layers and 2 input and output respectively
# All the Layers have a single neuron each
# The 28 hidden layers take input from the input layer and give that to the output layer

# In order to make large number of layers we use a loop structure
layers = [None] * 30
layers[0] = [1, 0, None, [], [i for i in range(1, 29)]]
print("The Layer 0 structure is: ")
print(layers[0])

# Defining Layer 1 and Layer 2 manually
layers[1] = [1, 1, LinearActivation(), [(0, False)], [29]]
layers[2] = [1, 1, LinearActivation(), [(0, False)], [29]]

# Defining Layers from 3 to 28 using loop
for i in range(3, 29):
	layers[i] = [1, 1, LinearActivation(), [(0, False)], [29]]
	
# Defining the final layer as CTRNN with 2 neurons
layers[29] = [2, 2, LinearActivation(), [(i, False) for i in range(1, 29)], []]
print("The Layer 29 structure is: ")
print(layers[29])


# Assuming we want 1 neuron for the output and convert to Simple Layer, the structure can be changed as
layers[29][0] = 1
layers[29][1] = 1 

# Constructing the Network
nn = ArtificialNeuralNetwork(layers)		# Simply passing the layers 2d matrix

# Considering we do not change the weights

# Input the Neural Network through a dictionary
input_dict = {
		0: np.array([1.0])		# Input to Layer 0
	     }
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)

