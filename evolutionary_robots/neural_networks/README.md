# Neural Networks API

## How to create a Neural Network?

**Import the Activation Function and Neural Network** along with Numpy(optional). The `neural_networks.activation_functions` consist of a library of activation functions. The `neural_networks.ann` contatins the ArtificialNeuralNetwork class.

```python
from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.activation_functions import SigmoidActivation
import numpy as np
```

**Initialize the Artificial Neural Network Object** For instance, we have a Perceptron with 2 input nodes and 1 output node along with a sigmoid activaiton function. Pass a **list of list** of Layer Parameters while initializing the Neural Network object.

```python
percpetron = ArtificialNeuralNetwork([[2, 0, None, [], [1]], [1, 1, SigmoidActivation(), [(0, False)], []]])
```

**Load self-generated parameters into the Neural Network** Assume, in this case the weights of the input are 0.5 and 0.5, the bias is 1. And the gain of the Sigmoid Activation is 1 and the offset is 0. Note that the parameters are to be given in a format specific to each Neural Network Class.

```python
perceptron.load_parameters_from_vector([0.5, 0.5, 1, 1, 0])
```

**Generate the output** In this case, the inputs to both the nodes is 1.

```python
output = perceptron.forward_propagate({0: [1, 1]})
```

For more examples, check the [examples directory](./../examples).

The following covers the specifics:

## Activation Function

`activation_functions.py` consists of a collection of activation functions. The activation functions available are:

| Activation Function   | Class to import     | Applicable Parameters |
|-----------------------|---------------------|-----------------------|
| Linear Activation     | LinearActivation()  | Beta and Theta		  |
| Step Activation		| StepActivation()	  | Theta				  |
| Sigmoid Activation	| SigmoidActivation() | Beta and Theta		  |
| Hyperbolic Tangent	| TanhActivation()	  | Beta and Theta		  |
| ReLU Activation		| ReluActivation()	  | Beta and Theta		  |
| Maximum Value			| MaximumActivation() | None				  |

**Note**: In case of activation functions with parameters less than 2. The format of parameter input of Neural Networks remains the same. Therefore, they are to be given any arbitrary values(for correct syntax). These values do not affect the activation function parameters.

**There are two ways to set the parameters of the Activation Function** Either while initializing the object or while explicitly by a function call. The default values of beta and theta are taken to be as 1 and 0 respectively.

```python
# Default Initialization
activation = LinearActivation()

# While initializing the object. Set beta as 2 and theta as 1 
activation = LinearActivation(2, 1)

# Explicitly assigining the values of beta as 1 and theta as 0
activation.beta = 1
activation.theta = 0
```

**Calculate the activation as follows** The input to the activation function should be a numpy array. Building on from the example above.

```python
print(activation.calculate_activation(np.array([-1, 0, 1])))
# The output is a numpy array [-1, 0, 1]
```

## Neural Network

`ann.py` consists of an ArtificialNeuralNetwork class which allows the creation of Static and Dynamic Neural Networks.

Methods supported by the ArtificialNeuralNetwork class are:

**Generate a Neural Network** with a given number of layers, number of neurons in each layer, the input and output connections and the activation function. The format of the list to be passed is given in the [next section](#format-for-initialization).

```python
nn = ArtificialNeuralNetwork(a_list_with_appropriate_format_for_initialization, time_constant_dictionary, time_interval)
```

The `time_constant_dictionary` is an optional parameter that specifies a list of time constants for the continuous layer. By default, the time constants are assigned a value of 1.

The `time_interval` is an optional parameter that specifies the time interval of the network. This is useful in the case when a continuous layer is used, otherwise this is ignored. By default, the time interval is 0.01.

**Generate the output of Neural Network** An input dictionary specifying the input specific to each layer is passed to the `forward_propagate` function of the ArtificialNeuralNetwork class. The input for both the input and associative layers is passed through this input dictionary. Input for any layer not passed is assumed to be zero.

The order of indices of the layers specified during initialization of the network is taken as the order of execution. A seperate user defined order of execution can also be followed while calculating the output of the network.

```python
# Order of execution according to the initialization indices
nn.forward_propagate(input_dictionary)

# Order of execution according to the user
nn.order_of_execution = [2, 3, 1, 0]
nn.forward_propagate(input_dictionary)
```

**Save and load the parameters from a file**

```python
# Save the parameters to a file in some specific directory
nn.save_parameters_to_file(path_to_directory_with_file_name_in_quotes)

# Load the parameters from a file in some specific directory
nn.load_parameters_from_file(path_to_directory_with_file_name_in_quotes)
```

**Return and load the parameters in the form of a list**. The format of parameter list for each Neural Network is given in the [next section](##format-for-parameter-vector).

```python
# Return the parameters
nn.return_parameters_as_vectors()

# Load the parameters
nn.load_parameters_from_vector(a_list_with_appropriate_format)
```

### Format for Initialization

```
[[number_of_neurons, type_of_layer, activation_function, input_connections, output_connections], ...]	*for each layer
```

**number_of_neurons** specify the number of neurons in a particular layer. In other words, these are the input dimensions of the layer

**type_of_layer** specifies the type of layer. There are 3 types of layer for the Neural Network. *0 implies input layer, 1 implies simple layer and 2 implies a continuous time recurrent layer*.

**activation_function** specifies the activation function class. This is usually taken from the activation functions library. In order to allow an outside class, it should possess a method called `calculate_activation` that takes in a single numpy array and outputs the activation result.

**input_connections** is a list of tuples. The first element specifies the index of the layers from which the current layer takes input. The second element is a boolean that tells whether to delay the input by next iteration or not. In order to construct specific networks, the delay element is set True.

Additionally, the order of specification of the input connections determines the format for the parameter vector that sets the weights of the network. 

**output_connections** is a list of integers specifying the index of the layers which the current layer provides output to.



### Format for parameter vector

```
[list_of_parameters_of_layer_0, list_of_parameters_of_layer_1, list_of_parameters_of_layer_2, ...]
# Here, w_ij implies the weight between ith input node and jth output node. b_i is the bias for the ith output node.
# a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
```

The parameter vector for setting the weights is a list of list. Each layer follows a particular pattern of how the weights are set from the vector.

**Input Layer** Input Layer takes no parameters. Therefore, the parameter list for input layers is kept as empty

**Simple Layer** The format followed is:

```python
[w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, a_1g, a_1b, a_2g, a_2b, a_3g, a_3b, w_11, w_21, w_31, b_1, ...]
# Here, w_ij implies the weight between ith input node and jth output node. b_i is the bias for the ith output node.
# a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
```

The jth output node corresponds to the index of the neuron of the layer under consideration.
The ith input node corresponds to the index of the neuron of the layer specified in the order of `input_connections` above.

**Continuous Time Recurrent Layer** The format followed is:

```python
[tc_1, tc_2, tc_3, w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, a_1g, a_1b, a_2g, a_2b, a_3g, a_3b, w_11, w_21, w_31, b_1, ...]
# Here, w_ij implies the weight between ith input node and jth output node. b_i is the bias for the ith output node.
# a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
# tc_i is the time constant of ith neuron of the current layer
```

The i and j format are similar to the Simple Layer.
