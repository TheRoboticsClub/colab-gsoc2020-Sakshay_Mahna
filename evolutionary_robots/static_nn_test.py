from neural_networks import activation_functions
from neural_networks.static_nn import Perceptron, StaticNeuralNetwork
import numpy as np


print("Perceptron:")
nn = Perceptron(2, 1, activation_functions.linear_function)
nn.load_weights_from_vector([0.5, 0.5, 1])		# The format followed is weights and then bias
print("The weights are: " + str(nn.return_weights_as_vector()))
input_vector = [1, 1]
print("The input is: " + str(input_vector))
output = nn.forward_propagate(input_vector)			# Pass input as an array
print("The output is: " + str(output))
nn.generate_visual('Perceptron')
print("\n")

print("Static Neural Network: ")
nn = StaticNeuralNetwork([[2, activation_functions.linear_function], [3, activation_functions.linear_function], [1]])
nn.load_weights_from_vector(np.array([0.5, 0, 0.5, 0, 0.5, 0, 1, 0, 0, 0.5, 0, 0, 1]))	# The format followed is weight matrix of a layer and then bias
# The columns of the weight matrix, tune for the weight of an input neuron
# The rows of the weight matrix, tune for the weight of an output neuron
vector = nn.return_weights_as_vector()
print("The weights are: " + str(vector))
input_vector = [1, 0.5]
print("The input is: " + str(input_vector))
output = nn.forward_propagate(input_vector)
print("The output is: " + str(output))
nn.generate_visual('MutliLayerPerceptron')
print("\n")

print("Static Test 1")
nn = StaticNeuralNetwork([[2, activation_functions.linear_function], [3, activation_functions.linear_function], [3, activation_functions.linear_function], [1]])
nn.load_weights_from_vector(np.array([0, 0.25, 0.5, 0.5, 0.75, 1.0, 1, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0, 1, 0.75, 0.5, 0.25, 0]))
input_vector = [1, 1]
print("The input is: " + str(input_vector))
output = nn.forward_propagate(input_vector)
print("The output is: " + str(output))
nn.generate_visual("StaticTest1", True)
print("\n")

print("Static Test 2")
nn = StaticNeuralNetwork([[1, activation_functions.linear_function], [10, activation_functions.linear_function], [2]])
nn.load_weights_from_vector(np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 0, 0]))
input_vector = [1]
print("The input is: " + str(input_vector))
output = nn.forward_propagate(input_vector)
print("The output is: " + str(output))
nn.generate_visual("StaticTest2")
