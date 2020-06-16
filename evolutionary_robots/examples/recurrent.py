import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
import numpy as np
from neural_networks.activation_functions import LinearActivation

print("Static Recurrent ANN: First Way")
nn = ArtificialNeuralNetwork([[2, 0, None, [], [1, 2]], [1, 1, LinearActivation(), [(0, False), (2, False)], [2, 3]], [1, 1, LinearActivation(), [(0, False), (1, True)], [1, 3]], [1, 1, LinearActivation(), [(1, False), (2, False)], []]])

parameter_vector = [[], [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 0], [1, 1, 0, 1, 0]]
nn.load_parameters_from_vector(parameter_vector)

input_dict = {0: np.array([1.0, 1.0])}
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)

print("Static Recurrent ANN: Second Way (Wrong Way)")
nn = ArtificialNeuralNetwork([[2, 0, None, [], [1, 2]], [1, 1, LinearActivation(), [(0, False), (2, True)], [2, 3]], [1, 1, LinearActivation(), [(0, False), (1, False)], [1, 3]], [1, 1, LinearActivation(), [(1, False), (2, False)], []]])

parameter_vector = [[], [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 0], [1, 1, 0, 1, 0]]
nn.load_parameters_from_vector(parameter_vector)

input_dict = {0: np.array([1.0, 1.0])}
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)
