from neural_networks import activation_functions
from neural_networks.static_nn import Perceptron, StaticNeuralNetwork
import numpy as np


print("Perceptron:")
# A perceptron with 2 inputs and 1 output and a linear activation function
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
# A MultiLayerPercpetron with 2 input nodes, 3 hidden nodes and 1 output node
nn = StaticNeuralNetwork([[2, activation_functions.linear_function], [3, activation_functions.linear_function], [1]])
nn.load_weights_from_vector(np.array([0.5, 0, 0.5, 0, 0.5, 0, 1, 0, 0, 0.5, 0, 0, 1]))	# The format followed is weight matrix of a layer and then bias
# [w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, w_11, w_21, w_31, b_1]. Here, w_ij implies the weight between ith input node and jth output node. b_i is the bias for the ith output node
vector = nn.return_weights_as_vector()
print("The weights are: " + str(vector))
input_vector = [1, 0.5]
print("The input is: " + str(input_vector))
output = nn.forward_propagate(input_vector)
print("The output is: " + str(output))
nn.generate_visual('MutliLayerPerceptron')
print("\n")

print("Static Test 1")
# Static Neural Network with 2 input nodes, 3 nodes in first hidden layer, 3 nodes in second hidden layer and 1 output node
nn = StaticNeuralNetwork([[2, activation_functions.linear_function], [3, activation_functions.linear_function], [3, activation_functions.linear_function], [1]])
nn.load_weights_from_vector(np.array([0, 0.25, 0.5, 0.5, 0.75, 1.0, 1, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0, 1, 0.75, 0.5, 0.25, 0]))
input_vector = [1, 1]
print("The input is: " + str(input_vector))
output = nn.forward_propagate(input_vector)
print("The output is: " + str(output))
nn.generate_visual("StaticTest1", True)
print("\n")

print("Static Test 2")
# Static Neural Network with 1 input node, 10 nodes in hidden layer and 2 output nodes
nn = StaticNeuralNetwork([[1, activation_functions.linear_function], [10, activation_functions.linear_function], [2]])
nn.load_weights_from_vector(np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 0, 0]))
input_vector = [1]
print("The input is: " + str(input_vector))
output = nn.forward_propagate(input_vector)
print("The output is: " + str(output))
nn.generate_visual("StaticTest2")
