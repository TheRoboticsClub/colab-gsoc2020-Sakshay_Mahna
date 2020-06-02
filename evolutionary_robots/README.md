# Evolutionary Robots
Exercises on Evolutionary Robotics

## Neural Networks

### Expectations
The Neural Network Class will provide the student with a template of a Static or Dynamic Neural Network. Following are the features of the module:

- A template for variable number of layers, number of nodes in those layers and their activation function
- Load and retreive the weights of the Neural Network
- Load and Save the weights of the Neural Network to an external file
- A visual representation of the Neural Network in terms of the graph diagrams

### Logic and Code
The fundamental concept behind Neural Networks is that of Layers. Neural Networks constitute a connection of layers. Given an input vector, the layers forward propagate the output to the next layers, finally giving the output. The output for each layer is given by matrix operations of multiplication and addition. Apart from this, an activation function is used to provide non-linear behaviour to our network.

Keeping the above things in mind the logic of the code is as follows. All the code is present in `neural_networks` directory.

- `activation_functions.py`: This file contains the most commonly used activation functions. They are implemented as python functions. Simply importing them and executing by giving them the appropriate **input_vectors** makes them work! Some sample test cases are provided in `activation_test.py`.

- `static_nn.py`: This file contains the Static Neural Network implementations. Each Neural Network is a collection of Layers. Therefore, a **layer abstraction** is provided which are used by the Neural Network class. The visual representations of the Neural Networks can be viewed by calling the function `network.generate_visual(file_name, view)`. Specify the name of the file as file_name and view is a boolean, to tell whether to view the network file or not. All the generated visuals go to the `representations` directory. The file contains **two** classes which are to be used by us. These are Perceptron and StaticNeuralNetwork. Sample test cases for both are provided in `static_nn_test.py`.
	- **Perceptron**: The input format for Percpetron is `network = Perceptron(input_dimension, output_dimension, activation_function)`. *input_dimension* is the number of input nodes and *output_dimension* is the number of output nodes. *activation_function* is a function, that can be implemented by the student or taken from the activation_functions module. Arbitrary weights can be assigned in the form of vectors(useful later on!) by using the function `network.load_weights_from_vector([weights_of_layer_one, bias_of_layer_one])`. The calculation is performed by the function `network.forward_propagate(input_vector)`.

	- **StaticNeuralNetwork**: The input formate for StaticNeuralNetwork is `network = StaticNeuralNetwork([[nodes_in_layer_one(input), activation_function_in_layer_one], [nodes_in_layer_two, activation_function], ..., [nodes_in_output]])`. Consider a Neural Network with 2 input nodes, 3 nodes in layer one and 1 output node, to load arbitrary weights to the neural network in the form of a vector, is done by `network.load_weights_from_vector([w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, w_11, w_21, w_31, b_1])`. Here, w_ij implies the weight between ith input node and jth output node. The input follows the same distribution as Perceptron, `weights_of_layer_one, bias_of_layer_one, weights_of_layer_two, bias_of_layer_two, ...`. The calculation is performed by the function `network.forward_propagate(input_vector)`.
	
### References
[GeeksForGeeks](https://www.geeksforgeeks.org/activation-functions/)

[Article on Medium](https://towardsdatascience.com/complete-guide-of-activation-functions-34076e95d044)

[NeuroLab project](https://github.com/zueve/neurolab)

[Visual Representation of Neural Networks](https://tgmstat.wordpress.com/2013/06/12/draw-neural-network-diagrams-graphviz/)
