# This part is specific to each user, not recommended to be copied
import sys
sys.path.append('./../')


# Import the required libraries
from neural_networks.static_nn import StaticNeuralNetwork
from neural_networks.activation_functions import SigmoidActivation, LinearActivation

# Example 1: Generate a Perceptron with 2 input nodes, 1 output node with Sigmoid Activation function
perceptron = StaticNeuralNetwork([[2, SigmoidActivation()], [1]])

# Load the specific parameters
# The weights for the input node are 0.5 and 0.5 and the bias is 1. The parameters of the Sigmoid function are 2(beta) and 1(theta)
perceptron.load_parameters_from_vector([0.5, 0.5, 1, 2, 1])

# Calculate the output, with input vector as 2 and 1
output = perceptron.forward_propagate([2, 1])

print(output)

# Example 2: Generate a MultiLayerPerceptron with 2 input neurons, 2 hidden layers with 3 neurons each and 1 output neuron, with Linear Activation at each step, with default parameters
nn = StaticNeuralNetwork([[2, LinearActivation()], [3, LinearActivation()], [3, LinearActivation()], [1]])

# Load the parameters
# The weights are as specified in the format and the image in README
nn.load_parameters_from_vector([0, 0.25, 0.5, 0.5, 0.75, 1.0, 1, 0, 0, 1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0, 1, 1, 0, 0.75, 0.5, 0.25, 0, 1, 0])

# Generate output to [1, 1]
output = nn.forward_propagate([1, 1])

print(output)
