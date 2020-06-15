from ann import ArtificialNeuralNetwork
import numpy as np
from activation_functions import LinearActivation

nn = ArtificialNeuralNetwork([[2, 0, LinearActivation(), [], [1, 2]], [1, 1, LinearActivation(), [(0, False), (2, True)], [2, 3]], [1, 1, LinearActivation(), [(0, False), (1, True)], [1, 3]], [1, 1, LinearActivation(), [(1, False), (2, False)], []]])

input_dict = {0: np.array([1.0, 1.0]), 1: np.array([0.0]), 2: np.array([0.0]), 3: np.array([0.0])}
output = nn.forward_propagate(input_dict)

print(output)
