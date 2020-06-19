# Neural Networks API

## How to create a Neural Network?

**Import the Activation Function and Neural Network** along with Numpy(optional). The `neural_networks.activation_functions` consist of a library of activation functions. The `neural_networks.ann` contains the ArtificialNeuralNetwork class. The `neural_networks.interface` contain user interface classes.

```python
from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.activation_functions import SigmoidActivation
from neural_networks.interface import Layer
import numpy as np
```

**Initialize the Artificial Neural Network Object** For instance, we have a Perceptron with 2 input nodes and 1 output node along with a sigmoid activation function. Generate two layer objects, which are going to serve as the input and output layers.

```python
# The format is number of neurons, type of layer, activation function, input connections and output connections
inputLayer = Layer(2, 0, None, [], [1])
outputLayer = Layer(1, 1, SigmoidActivation(), [0], [])
percpetron = ArtificialNeuralNetwork([
                                      inputLayer,                                  # Input Layer
                                      outputLayer           # Output Layer
                                      ])
```

**Load self-generated parameters into the Neural Network** Assume, in this case the weights corresponding to second layer are 0.5 and 0.5, the bias is 1. And the gain of the Sigmoid Activation is 1 and the offset is 0. Note that the parameters are to be given in a format specific to each layer of the network.

```python
perceptron.load_parameters_from_vector([
                                        [],                         # Input Layer
                                        [0.5, 0.5, 1, 1, 0]         # Output Layer
                                       ])
```

**Generate the output** In this case, the inputs to both the nodes of input layer is 1.

```python
output = perceptron.forward_propagate({
                                       0: [1, 1]        # Input Layer
                                      })
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

**There are two ways to set the parameters of the Activation Function** Either while initializing the object or explicitly. The default values of beta and theta are taken to be 1 and 0 respectively.

```python
# Default Initialization
activation = LinearActivation()

# While initializing the object. Set beta as 2 and theta as 1 
activation = LinearActivation(2, 1)

# Explicitly assigining the values of beta as 1 and theta as 0
activation.beta = 1
activation.theta = 0
```

**Calculate the activation** The input to the activation function should be a numpy array. Building on from the example above.

```python
print(activation.calculate_activation(np.array([-1, 0, 1])))
# The output is a numpy array [-1, 0, 1]
```

## Layer

`interface.py` consists classes that define easier user interface. Layer class is present in this module.

Layer class allows easier initialization and management of layers of the Neural Network. The parameters of the layers can be passed during initialization or by changing the attributes of the object.

```python
# Initializing Layer Object
layer = Layer(number_of_neurons, type_of_layer, activation_function, list_of_input_connections, list_of_output_connections)

# Initializing with attributes or changing the attributes
layer = Layer()
layer.number_of_neurons = 1
layer.type_of_layer = 2
layer.activation_function = SigmoidActivation()
layer.input_connections = [1, 2]
layer.output_connections = [0]
```

**number_of_neurons** specifies the number of neurons in a particular layer.

**type_of_layer** specifies the type of layer. There are 3 types of layer for the Neural Network. *0 implies input layer, 1 implies simple layer and 2 implies a continuous time recurrent layer*.

- The *input layer* simply takes input of a sensor and returns them with an optional gain.

- The *simple layer* is a simple feed forward layer that calculates the output through forward propagation. Using specific input connections, this layer can behave as a recurrent layer as well. In addition to it, optional sensor values can also be passed to the layer(from input_dict) to make it behave as associative layer as well.

- The *continuous time recurrent layer* is a feed forward layer that calculates the output using first order euler approximation. Similar to the *simple layer*, this layer can be made to behave accordingly.

**activation_function** specifies the activation function class. This is usually taken from the activation functions library. In order to allow an outside class, it should possess a method called `calculate_activation` that takes in a single numpy array and outputs the activation result.

**input_connections** is a list of integers specifying the index of the layers which the current layer takes input from. An input layer is determined by it's input connections. A layer having type of layer equal to 1 is an input layer, any **input connections specified for this layer are simply ignored**. Additionally, the order of specification of the input connections **determines the format of the parameter vector** that sets the weights of the network. 

**output_connections** is a list of integers specifying the index of the layers which the current layer provides output to. An output layer is determined by it's output connections. **A layer having no output connections is considered as the output layer**, and the output dictionary returned by `forward_propagate()` returns the output of such layers.

**delay_connections** is an optional attribute that has to be set explicitly apart from initialization. It is a list of integers specifying the layer indices from which the input to current layer is to be delayed by one iteration of the network. By default, delay_connections is an empty list.

**time_constants** is an optional attribute that has to be set explicitly apart from initialization. It is a list of integers specifying the time constants of the neurons of the layer. By default, time constants are taken to be one.

**gains** is an optional attribute that has to be set explicitly apart from initialization. It is a list of integers specifying the gain provided to each input of the neuron. By default, the gains are taken as 1.

```python
layer = Layer()
layer.input_connections([1, 2])		# Layer takes input from layer indexed as 1 and 2
layer.delay_connections([2])		# Layer delays the input taken from 2 by one iteration

layer.time_constants = [2, 2]
layer.gains = [1, 1]
```


## Neural Network

`ann.py` consists of an ArtificialNeuralNetwork class which allows the creation of Static and Dynamic Neural Networks.

Methods supported by the ArtificialNeuralNetwork class are:

**Generate a Neural Network** with a given number of layers, number of neurons in each layer, the input and output connections and the activation function.

```python
nn = ArtificialNeuralNetwork(a_list_of_layer_object, time_interval)
```

The `time_interval` is an optional parameter that specifies the time interval of the network. This is useful in the case when a continuous layer is used, otherwise this is ignored. By default, the time interval is 0.01.

**Generate the output of Neural Network** An input dictionary specifying the input specific to each layer is passed to the `forward_propagate` function of the ArtificialNeuralNetwork class. The input for both the input and associative layers is passed through this input dictionary. Input for any layer not passed is assumed to be zero.

The order of indices of the layers specified during initialization of the network is taken as the **order of execution**. A seperate user defined order of execution can also be followed while calculating the output of the network. The Neural Network in another sense is a computational graph. The order in which the computations are done is specified by numbering the layers in ascending order starting from 0. In this way, the order of indices of layers during initialization would be the same as the order of execution.

The output is in the form of a dictionary. The dictionary is keyed according to the index of the layer as specified during initialization.

```python
# Order of execution according to the initialization indices
nn.forward_propagate(input_dictionary)

# Order of execution according to the user
nn.order_of_execution = [2, 3, 1, 0]
nn.forward_propagate(input_dictionary)
```

In order to set the gains of a particular layer, the following function calls are requried:

```python
# Set the gain of layer_0
nn.set_gain(0, np.array([1, 1, 1...]))

# Return the gains that were set
nn.get_gain(0)
```

By default, the gains are equal to 1.

**Save and load the parameters from a file**

```python
# Save the parameters to a file in some specific directory
nn.save_parameters_to_file(path_to_directory_with_file_name_in_quotes)

# Load the parameters from a file in some specific directory
nn.load_parameters_from_file(path_to_directory_with_file_name_in_quotes)
```

**Return and load the parameters in the form of a list**. The format of parameter list for each Neural Network is given in the [next section](#format-for-parameter-vector).

```python
# Return the parameters
nn.return_parameters_as_vectors()

# Load the parameters
nn.load_parameters_from_vector(a_list_with_appropriate_format)
```

**View the output of each layer** The output of each and every layer(previous and current) is avaialable as class variable of the Neural Network class

```python
nn.output_matrix
```


### Format for parameter vector

```python
[list_of_parameters_of_layer_0, list_of_parameters_of_layer_1, list_of_parameters_of_layer_2, ...]
```

The parameter vector for setting the weights is a list of list. Each layer follows a particular pattern of how the weights are set from the vector.

#### Input Layer

Input Layer takes no parameters. Therefore, the parameter list for input layers is kept as empty

#### Simple Layer 

The format followed is:

```python
[w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, a_1g, a_1b, a_2g, a_2b, a_3g, a_3b, w_11, w_21, w_31, b_1, ...]
# Here, w_ij implies the weight between ith input node and jth output node. b_i is the bias for the ith output node.
# a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
```

The jth output node corresponds to the index of the neuron of the layer under consideration.
The ith input node corresponds to the index of the neuron of the layer specified in the order of `input_connections` above.

#### Continuous Time Recurrent Layer

The format followed is:

```python
[tc_1, tc_2, tc_3, w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, a_1g, a_1b, a_2g, a_2b, a_3g, a_3b, w_11, w_21, w_31, b_1, ...]
# Here, w_ij implies the weight between ith input node and jth output node. b_i is the bias for the ith output node.
# a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
# tc_i is the time constant of ith neuron of the current layer
```

The i and j format are similar to the Simple Layer.
