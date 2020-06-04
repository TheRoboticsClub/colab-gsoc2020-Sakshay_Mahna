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

- `static_nn.py`: This file contains the Static Neural Network implementations. Each Neural Network is a collection of Layers. Therefore, a **layer abstraction** is provided which are used by the Neural Network class. The visual representations of the Neural Networks can be viewed by calling the function `network.generate_visual(file_name, view)`. Specify the name of the file as file_name and view is a boolean, to tell whether to view the network file or not. All the generated visuals go to the `representations` directory. `static_nn.py` contains **two** classes which are to be used by us. These are Perceptron and StaticNeuralNetwork. Sample test cases for both are provided in `static_nn_test.py`.
	- **Perceptron**: The input format for Percpetron is `network = Perceptron(input_dimension, output_dimension, activation_function)`. *input_dimension* is the number of input nodes and *output_dimension* is the number of output nodes. *activation_function* is a function, that can be implemented by the student or taken from the activation_functions module. Arbitrary weights can be assigned in the form of vectors(useful later on!) by using the function `network.load_weights_from_vector([weights_of_layer_one, bias_of_layer_one])`. The calculation is performed by the function `network.forward_propagate(input_vector)`.

	- **StaticNeuralNetwork**: The input formate for StaticNeuralNetwork is `network = StaticNeuralNetwork([[nodes_in_layer_one(input), activation_function_in_layer_one], [nodes_in_layer_two, activation_function], ..., [nodes_in_output]])`. Consider a Neural Network with 2 input nodes, 3 nodes in layer one and 1 output node, to load arbitrary weights to the neural network in the form of a vector, is done by `network.load_weights_from_vector([w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, w_11, w_21, w_31, b_1])`. Here, w_ij implies the weight between ith input node and jth output node. The input follows the same distribution as Perceptron, `weights_of_layer_one, bias_of_layer_one, weights_of_layer_two, bias_of_layer_two, ...`. The calculation is performed by the function `network.forward_propagate(input_vector)`.
	
- `dynamic_nn.py`: This file contains Dynamic Neural Network implementations. Each Neural Network is a collection of Dynamic Layer. Each dynamic layer consists of the delay system, recurrent system and the static system. The static system is the same as the StaticLayer. The recurrent system needs to get the output from a different layer and integrate it with the layer under consideration. The delay system works by generating a weighted average of the input vector supplied currently and in the previous iteration. The input format for DynamicNeuralNetwork is `network = DynamicNeuralNetwork([[nodes_in_layer_one, delay_dim, [list_of_connections], activation_function], [nodes_in_layer_two, delay_dim, [list_of_connections], activation_function], ...[nodes_in_output]])`. The weight matrix follows the same convention as the *StaticNeuralNetwork*, additionaly the other weights are formatted as `network.load_weights_from_vector([weights_of_delay, weights_of_recurrence, static_weights, weights_of_bias])`. The calculation is performed by the function `network.forward_propagate(input_vector)`. Some sample test cases are provided in `dynamic_nn_test.py`. The visual representations of the Neural Networks can be viewed by calling the function `network.generate_visual(file_name, view)`. Specify the name of the file as file_name and view is a boolean, to tell whether to view the network file or not. All the generated visuals go to the `representations` directory. The recurrent connections are denoted in dotted and red color. The Delay is specifed above the layer.

- `ctrnn.py`: This file contains Continuous Time Recurrent Neural Network implementation. A new layer called CTRNNLayer, is used that generates the weighted average of the previous output and the current output based on the weights that are derived from the time interval and the time constants. The Layer implements **first order euler solution** of the continuous differential equation. The input format for CTRNN is `network=CTRNN([[nodes_in_layer_one, list_of_time_constants, activation_function], [nodes_in_layer_two, list_of_time_constants, activation_function], ..., [nodes in output]], time_interval)`. The weight matrix follows the same format as the Static Neural Network `network.load_weights_from_vector()`. The calculation is performed by the function `network.forward_propagate(input_vector)`. A sample test case is provided in `ctrnn_test.py`. The visual representations of the Neural Networks can be viewed by calling the function `network.generate_visual(file_name, view)`. Specify the name of the file as file_name and view is a boolean, to tell whether to view the network file or not. All the generated visuals go to the `representations` directory. The time constants of the neurons are specified in green in their respective nodes

- `rbf_nn.py`: This file contains Radial Basis Function Neural Network implementation. The Neural Network is formed using a combination of a RBFLayer and a StaticLayer. Euclidean Distance is used as the distance function and an implementation of the Gaussian Radial Basis Network is provided. The Network consists of only 3 layers. The input format of the RBFNetwork is `network = RBFNetwork(input_dimension, hidden_dimension, output_dimension, beta(optional))`. `beta` specifies the value of the Gaussian Parameter. The weight vector is to be passed as a concatenation of the centers of the neurons and the weights. The centers are to be passed from top to bottom and the weights follow the same pattern as Static Layer. The weights can be changed by `network.load_weights_from_vector(center_matrix_in_row_major_format + weight_matrix_flattened)`. The calculation is performed by the function `network.forward_propagate(input_vector)`. A sample test case is provided in `rbf_test.py`. The visual representations of the Neural Networks can be viewed by calling the function `network.generate_visual(file_name, view)`. Specify the name of the file as file_name and view is a boolean, to tell whether to view the network file or not. All the generated visuals go to the `representations` directory.
	
	
### Tests
Some sample tests can be found in the `tests` directory.

### References
[GeeksForGeeks](https://www.geeksforgeeks.org/activation-functions/)

[Article on Medium](https://towardsdatascience.com/complete-guide-of-activation-functions-34076e95d044)

[NeuroLab project](https://github.com/zueve/neurolab)

[Visual Representation of Neural Networks](https://tgmstat.wordpress.com/2013/06/12/draw-neural-network-diagrams-graphviz/)

[CTRNN](https://neat-python.readthedocs.io/en/latest/ctrnn.html)

[RBF](https://en.wikipedia.org/wiki/Radial_basis_function_network)
