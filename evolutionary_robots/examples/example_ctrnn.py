# This part is specific to each user, not recommended to be copied
import sys
sys.path.append('./../')


# Import the required libraries
from neural_networks.ctrnn import CTRNN
from neural_networks.activation_functions import SigmoidActivation

# Example
# The CTRNN network has 2 input nodes, 3 hidden nodes and 1 output node
# THe time constants for first activation are [2, 3, 5]
# The time constant for second activation is [2]
# The time interval chosen is 0.6
nn = CTRNN([[2, [2, 3, 5], SigmoidActivation()], [3, [2], SigmoidActivation()], [1]], 0.6)

# The weights are as shown in format and README
nn.load_parameters_from_vector([2, 3, 5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 1, 1, 0, 2, 0.7, 0.8, 0.9, 0, 1, 0])

# Generate output to [1, 1]
output = nn.forward_propagate([1, 1])
print(output)

# Second time to see recurrence in action
output = nn.forward_propagate([1, 1])
print(output)
