# This part is specific to each user, not recommended to be copied
import sys
sys.path.append('./../')


# Import the required libraries
from neural_networks.dynamic_nn import DynamicNeuralNetwork
from neural_networks.activation_functions import SigmoidActivation, LinearActivation

# Example 1: Generate a Dynamic Neural Network with 3 Layers
# The input layer consists of single neuron with 2 units of delay
# The second layer consist of two neurons and linear activation
# The output layer consists of a single neuron with sigmoid activation
# The output of second layer is connected to itself and the first layer
# The output of final layer is connected to the first layer
nn = DynamicNeuralNetwork([[1, 2, [1, 2], LinearActivation()], [2, 1, [0, 1], LinearActivation()], [2, 1, [0], SigmoidActivation()], [1]])

# Load the parameters
# The weights are as specified in the format and the image in README
nn.load_parameters_from_vector([2, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.1, 0.2, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0.3, 0.4, 0.5, 0.6, 0, 1, 1, 0, 1, 0.7, 0.8, 0, 1, 0])

# Generate output to [0.5]
output = nn.forward_propagate([0.5])
print(output)

# Second time to see the recurrence and delay in action
output = nn.forward_propagate([0.5])
print(output)


# Example 2: Generate a Dynamic Neural Network with 3 Layers
# The input layer consists of a single neuron with linear activation
# The second layer consists of three neurons with linear activation
# The third layer consists of two neurons with linear activation
# There are no recurrent connections
nn = DynamicNeuralNetwork([[2, 1, [], LinearActivation()], [3, 1, [], LinearActivation()], [2]])

# Load the parameters
# The weights are as specified in the format
# Note: Since we do not have any recurrent relations, therefore the recurrent weights are not specified
nn.load_parameters_from_vector([1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 1, 1, 0, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, 0])

# Generate output to [1, 1]
output = nn.forward_propagate([1, 1])
print(output)

# Second time to see there were no recurrent connections
output = nn.forward_propagate([1, 1])
print(output)
