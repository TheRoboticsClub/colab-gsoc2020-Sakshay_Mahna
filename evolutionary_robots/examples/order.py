import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
import numpy as np
from neural_networks.activation_functions import LinearActivation

print("Static ANN example of order: Wrong Way")
nn = ArtificialNeuralNetwork([[2, 0, None, [], [1, 2]], [2, 1, LinearActivation(), [(0, False), (2, False)], []], [2, 1, LinearActivation(), [(0, False)], [1]]])

parameter_vector = [[], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 0]]
nn.load_parameters_from_vector(parameter_vector)

input_dict = {0: np.array([1.0, 1.0])}
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)

print("Static ANN example of order: Correct Way")
nn = ArtificialNeuralNetwork([[2, 0, None, [], [1, 2]], [2, 1, LinearActivation(), [(0, False)], [2]], [2, 1, LinearActivation(), [(0, False), (1, False)], []]])
parameter_vector = [[], [1, 1, 1, 1, 0, 0, 1, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 0, 1, 0]]
nn.load_parameters_from_vector(parameter_vector)

input_dict = {0: np.array([1.0, 1.0])}
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)
