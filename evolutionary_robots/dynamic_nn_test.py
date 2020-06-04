from neural_networks import activation_functions
from neural_networks.dynamic_nn import DynamicNeuralNetwork


print("Dynamic Neural Networks")
print("\n")

print("Input layer: 2 nodes and Output layer: 1 node")
print("Time Delay: 2 units; Input connected with output layer as recurrence")
# DNN with 2 input nodes, with 2 units of delay along with self recurrence and 1 output node
nn = DynamicNeuralNetwork([[2, 2, [0], activation_functions.linear_function], [1]])
nn.load_weights_from_vector([1, 1, 1, 1, 0, 0])		# The format followed is weights of delay system + weights of recurrent system(quite varying length) + weights of static system + weights of bias
nn.generate_visual("Dynamic 2", True)
print(nn.forward_propagate([1, 1]))			# Forward propagation 3 times to check the memory system
print(nn.forward_propagate([1, 1]))
print(nn.forward_propagate([1, 1]))
print(nn.forward_propagate([1, 1]))

print("\n")
print("Input layer: 2 nodes and Output layer: 1 node")
print("Time Delay: 1 unit; Input connected with output layer as recurrence")
# DNN with 2 input nodes, with 1 unit of delay along with self recurrence and 1 output node
nn = DynamicNeuralNetwork([[2, 1, [0], activation_functions.linear_function], [1]])
nn.load_weights_from_vector([1, 1, 1, 0, 0])		# The format followed is weights of delay system + weights of recurrent system(quite varying length) + weights of static system + weights of bias
nn.generate_visual("Dynamic 1", True)
print(nn.forward_propagate([1, 1]))			# Forward propagation 3 times to check the memory system
print(nn.forward_propagate([1, 1]))
print(nn.forward_propagate([1, 1]))
print(nn.forward_propagate([1, 1]))

print("\n")
print("The same Static Neural Network in static_nn_test.py")
# Static Neural Network implemented using DNN
# 2 input nodes, 3 hidden nodes and 1 output node
nn = DynamicNeuralNetwork([[2, 1, [], activation_functions.linear_function], [3, 1, [], activation_functions.linear_function], [1]])
nn.load_weights_from_vector([1, 0.5, 0, 0.5, 0, 0.5, 0, 1, 0, 0, 1, 0.5, 0, 0, 1])
#			  delay, weights of layer one, bias,   delay, weights of layer two bias
# Here the recurrence is skipped as it's weights are not required
nn.generate_visual("StaticDynamic", True)
print(nn.forward_propagate([1, 1]))

print("\n")
print("Dynamic Test 1")
# DNN with 4 Layers, layer 0, layer 1, layer 2 and output layer
# 1 input node, with 2 units of delay and connected with outputs of layer 1 and layer 2. Output of layer 2 is in essence the output layer
# 2 hidden nodes, with 1 unit of delay(no delay in essence) connected to input of layer 0 and output of layer 1(self recurrence)
# 2 hidden nodes, with 1 unit of delay connected to input of layer 0
# 1 output node
nn = DynamicNeuralNetwork([[1, 2, [1, 2], activation_functions.linear_function], [2, 1, [0, 1], activation_functions.linear_function], [2, 1, [0], activation_functions.sigmoid_function], [1]])
# Each connected is given based on the output
# If we have an input connection(example, in case of layer 0), the outputs of the specified layer will be taken
# If we have an output connection(example, in case of layer 2), the outputs of this layer will be connected to the specified layer
nn.load_weights_from_vector([2, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.1, 0.2, 1, 0, 1, 1, 1, 1, 1, 0.3, 0.4, 0.5, 0.6, 0, 1, 1, 0.7, 0.8, 0])
# The weights and bias are specified with an image in tests directory
print(nn.forward_propagate([0.5]))
print(nn.forward_propagate([0.5]))
nn.generate_visual('DynamicTest1', True)



