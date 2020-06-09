from neural_networks import activation_functions
from neural_networks.rbf_nn import GaussianRBFNetwork
import numpy as np

print("Radial Basis Function Network Test: ")
# 2 Input nodes
# 2 Hidden nodes
# 1 Output node
nn = GaussianRBFNetwork(2, 2, 1)
# The center of first node is (0.25, 0.75)
# The center of second node is (0.0, 1.0)
# The weights of the second layer are [[0.5], [0.5]]
nn.load_weights_from_vector(np.array([0.25, 0.75, 0.0, 1.0, 0.5, 0.5]))
nn.generate_visual("RBFNetwork 1", True)
print(nn.forward_propagate([1, 1]))

