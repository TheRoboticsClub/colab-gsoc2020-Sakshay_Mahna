from neural_networks import activation_functions
from neural_networks.dynamic_nn import DynamicNeuralNetwork


print("Dynamic Neural Networks")
print("\n")

print("Input layer: 2 nodes and Output layer: 1 node")
print("Time Delay: 2 units; Input connected with output layer as recurrence")
nn = DynamicNeuralNetwork([[2, 2, [0], activation_functions.linear_function], [1]])
nn.load_weights_from_vector([1, 1, 1, 1, 0, 0])		# The format followed is weights of delay system + weights of recurrent system(quite varying length) + weights of static system + weights of bias
print(nn.forward_propagate([1, 1]))			# Forward propagation 3 times to check the memory system
print(nn.forward_propagate([1, 1]))
print(nn.forward_propagate([1, 1]))
print(nn.forward_propagate([1, 1]))

print("\n")
print("Input layer: 2 nodes and Output layer: 1 node")
print("Time Delay: 1 unit; Input connected with output layer as recurrence")
nn = DynamicNeuralNetwork([[2, 1, [0], activation_functions.linear_function], [1]])
nn.load_weights_from_vector([1, 1, 1, 0, 0])		# The format followed is weights of delay system + weights of recurrent system(quite varying length) + weights of static system + weights of bias
print(nn.forward_propagate([1, 1]))			# Forward propagation 3 times to check the memory system
print(nn.forward_propagate([1, 1]))
print(nn.forward_propagate([1, 1]))
print(nn.forward_propagate([1, 1]))

print("\n")
print("The same Static Neural Network in static_nn_test.py")
nn = DynamicNeuralNetwork([[2, 1, [], activation_functions.linear_function], [3, 1, [], activation_functions.linear_function], [1]])
nn.load_weights_from_vector([1, 0.5, 0, 0.5, 0, 0.5, 0, 1, 0, 0, 1, 0.5, 0, 0, 1])
#			  delay, weights of layer one, bias,   delay, weights of layer two bias
# Here the recurrence is skipped as it's weights are not required
print(nn.forward_propagate([1, 1]))
