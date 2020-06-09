# This part is specific to each user, not recommended to be copied
import sys
sys.path.append('./../')


# Import the required libraries
from neural_networks.rbf_nn import GaussianRBFNetwork

# Example: Generate an RBF Network with 2 input neurons, 2 hidden layers and 1 output neuron
nn = GaussianRBFNetwork(2, 2, 1)

# The center of first node is (0.25, 0.75)
# The center of second node is (0.0, 1.0)
# The weights of second layer are [[0.5], [0.5]]
nn.load_parameters_from_vector([0.25, 0.75, 0.0, 1.0, 0.5, 0.5])		# center, weights

# Generate output to [1, 1]
output = nn.forward_propagate([1, 1])

print(output)
