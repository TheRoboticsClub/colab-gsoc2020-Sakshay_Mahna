import sys
sys.path.append('./../')

from neural_networks.ann import ArtificialNeuralNetwork
import numpy as np
from neural_networks.activation_functions import LinearActivation

print("Complex Recurrent Example")
nn = ArtificialNeuralNetwork([[2, 0, None, [], [1]], [2, 1, LinearActivation(), [(0, False), (4, False)], [2, 3, 4]], [2, 1, LinearActivation(), [(1, False)], [4]], [2, 1, LinearActivation(), [(1, False)], [4]], [1, 1, LinearActivation(), [(2, False), (1, False), (3, False)], [1, 5]], [1, 1, LinearActivation(), [(4, False)], []]])

# Weights are such that the recurrence is not able to show itself!
parameter_vector = [[], [1, 1, 0, 1, 1, 0, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 0, 1, 0], [1, 0, 1, 0]]
nn.load_parameters_from_vector(parameter_vector)

input_dict = {0: np.array([1.0, 1.0])}
output = nn.forward_propagate(input_dict)
print(output)
output = nn.forward_propagate(input_dict)
print(output)

nn.set_gain(3, [2, 2])
input_dict = {0: np.array([1.0, 1.0]), 3: np.array([1.0, 1.0])}
output = nn.forward_propagate(input_dict)
print(output)

