from neural_networks import activation_functions
from neural_networks.ctrnn import CTRNN
import numpy as np

print("Continuous Time Recurrent Neural Network Test: ")
# The CTRNN network has 2 input nodes, 3 hidden nodes and 1 output node
# THe time constants for first activation are [2, 3, 5]
# The time constant for second activation is [2]
# The time interval chosen is 0.6
nn = CTRNN([[2, [2, 3, 5], activation_functions.sigmoid_function], [3, [2], activation_functions.sigmoid_function], [1]], 0.6)

# Weights given in the same format as Static Weights
nn.load_weights_from_vector(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 1, 0.7, 0.8, 0.9, 0]))
nn.generate_visual("CTRNN 1", True)
print(nn.forward_propagate([1, 1]))
print(nn.forward_propagate([1, 1]))

