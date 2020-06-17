import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
import numpy as np
from neural_networks.activation_functions import LinearActivation

print("Static ANN")
# Simple ANN with 2 input 3 output and Linear Activation
nn = ArtificialNeuralNetwork([[2, 0, None, [], [1]], [3, 1, LinearActivation(), [(0, False)], []]])

parameter_vector = [[], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, 1, 0]]
nn.load_parameters_from_vector(parameter_vector)

input_dict = {0: np.array([1.0, 1.0])}
output = nn.forward_propagate(input_dict)

print(output)

print("CTRNN")
# CTRNN with 2 input 3 output and Linear Activation
# And time constants are 1, 1, 1
nn = ArtificialNeuralNetwork([[2, 0, None, [], [1]], [3, 2, LinearActivation(), [(0, False)], []]])

parameter_vector = [[], [1, 1, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, 1, 0]]
nn.load_parameters_from_vector(parameter_vector)

input_dict = {0: np.array([1.0, 1.0])}
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)



