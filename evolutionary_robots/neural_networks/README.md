# Neural Networks API

## How to create a Neural Network?

**Import the Activation Function and Neural Network** along with Numpy(optional). The `neural_networks.activation_functions` consist of a library of activation functions. The `neural_networks.ann` contains the ArtificialNeuralNetwork class. The `neural_networks.interface` contains the user interface Layer class.

```python
from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.activation_functions import SigmoidActivation, IdentityActivation
from neural_networks.interface import Layer
import numpy as np
```

**Initialize the Artificial Neural Network Object** For instance, we have a Static Perceptron with 2 input nodes and 1 output node along with a sigmoid activation function. In terms of the API, this will result in 2 layers. The input layer having 2 neurons, taking input from a sensor and the output layer having 1 neuron, providing the output to some actuator.

**The input layer should always have an identity activation function**

```python
# The format is: Layer(name of layer, number of neurons, activation function, sensor input, output connections)
inputLayer = Layer("inputLayer", 2, IdentityActivation(), "SENSOR", ["outputLayer"])
outputLayer = Layer("outputLayer", 1, SigmoidActivation(), "", ["ACTUATOR"])
percpetron = ArtificialNeuralNetwork([
                                      inputLayer,           # Input Layer
                                      outputLayer           # Output Layer
                                      ], "STATIC")	    # Static Neural Network
```

**Load self-generated parameters into the Neural Network** Assume, in this case the weights corresponding to second layer are 0.5 and 0.5. And the gain of the Sigmoid Activation is 1 and the offset is 0. Note that the parameters are to be given in a format specific to each layer of the network.

```python
perceptron.load_parameters_from_vector([
                                        [],                         # Input Layer
                                        [			    
                                        	0.5, 0.5,	    # Weight Matrix of Output Layer
                                        	1, 0		    # Activation Parameters of Output Layer
                                        ]         
                                       ])
```

**Generate the output** In this case, the inputs to both the nodes of SENSOR is 1.

```python
output = perceptron.forward_propagate({
                                       "SENSOR": [1, 1]        # Input to SENSOR
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
| Identity Function		| IdentityActivation()| None				  |

**Note**: In case of activation functions with parameters less than 2. The format of parameter input of Neural Networks remains the same. Therefore, they are to be given any arbitrary values(for correct syntax). These values affect neither the activation function parameters nor their result.

**There are two ways to set the parameters of the Activation Function** Either while initializing the object or explicitly. The default values of beta and theta are taken to be 1 and 0 respectively.

```python
# Default Initialization
activation = LinearActivation()

# While initializing the object. Set beta as 2 and theta as 1 
activation = LinearActivation(2, 1)

# Explicitly assigining the values of beta as 1 and theta as 0
activation.beta = 1
activation.theta = 0

# Assigning values of beta and theta for each neuron seperately(3 neurons in our case)
activation.beta = [1, 2, 3]
activation.theta = [1, 2, 3]
```

**Calculate the activation** The input to the activation function should be a numpy array. Building on from the example above.

```python
print(activation.calculate_activation(np.array([-1, 0, 1])))
# The output is a numpy array [-1, 0, 1]
```

## Layer

`interface.py` consists of classes that define easier user interface. Layer class is present in this module.

Layer class allows easier initialization and management of layers of the Neural Network. The parameters of the layers can be passed during initialization or by changing the attributes of the object.

```python
# Initializing Layer Object
layer = Layer(name_of_layer, number_of_neurons, activation_function, sensor_input, list_of_output_connections)

# Initializing with attributes or changing the attributes
layer = Layer(layer_name)		# The layer name attribute is a required parameter for initialization
layer.number_of_neurons = number_of_neurons
layer.activation_function = activation_function
layer.sensor_input = sensor_input
layer.output_connections = list_of_output_connections
```

**name_of_layer** specifies the name of the layer. This parameter is a string. The name of each layer should be unique and is a required parameter while initializing the Layer object. The names can be any string, such as : "layer0", "inputLayer" or "layer_0"

**number_of_neurons** specifies the number of neurons in a particular layer. This parameter is an integer data type.

**activation_function** specifies the activation function class. This is usually taken from the activation functions library. In order to allow an outside class, it should possess a method called `calculate_activation` that takes in a single numpy array and outputs the activation result.

**sensor_input** is a string specifying the sensor from which the layer is to take input. This sensor string determines the keys of the input dictionary that we pass to forward propagate function to calculate the output of the network. It can be any string such as: "CAMERA", "SENSOR" or "INFRARED" . Conventionally, these should be in CAPTIALS.

**output_connections** is a list of strings specifying the name of the layers which the current layer provides output to. For layers that output to an actuator or hardware, they have an additional member for that as well. The output dictionary returned by `forward_propagate()` returns the output of these hardware. These hardware outputs can have any name. Conventionally, these outputs should be in CAPITALS. For example: ["layer_0", "layer_1"] or ["layer0", "HARDWARE"]

## Neural Network

`ann.py` consists of an ArtificialNeuralNetwork class which allows the creation of Static and Dynamic Neural Networks.

Methods supported by the ArtificialNeuralNetwork class are:

**Generate a Neural Network** with a given number of layers, number of neurons in each layer, the sensor input and output connections and the activation function.

```python
nn = ArtificialNeuralNetwork(a_list_of_layer_object, type_of_network, time_interval)
```

The `list_of_layer_object` parameter is a list of `Layer()` objects. In general, different orders of initialization of layers will always generate the same network. However, **the elements of parameter vector will need to be changed accordingly**. For a better understanding check the example [order.py](./../examples/order.py)

The `type_of_network` parameter is a string specifying whether we want to use a Static or Dynamic Neural Network. A Static Neural Network is a simple feed forward neural network without any memory. A Dynamic Neural Network is a memory based network which is having all the inputs to each layer delayed by one iteration(time step). To use static net, we pass the string "STATIC" and to use dynamic net, we pass the string "DYNAMIC".

The `time_interval` is an optional parameter that specifies the time interval of the network. This is useful in the case when a Dynamic Network is used, otherwise this is ignored. By default, the time interval is 0.01.

[perceptron.py](./../examples/perceptron.py), [order.py](./../examples/order.py) and [large.py](./../examples/large.py) show examples of Static Neural Network.

[recurrent.py](./../examples/recurrent.py) and [complex.py](./../examples/complex.py) show examples of Dynamic Neural Network.

**Generate the output of Neural Network** An input dictionary specifying the input specific to each sensor is passed to the `forward_propagate` function of the ArtificialNeuralNetwork class. The sensor input for both the input and associative layers is passed through this input dictionary. Input for any sensor not passed is assumed to be zero.

```python
# For instance we have two sensors that serve as the input
input_dictionary = {
	"SENSOR1": [1, 2],
	"SENSOR2": [1, 1]
}

nn.forward_propagate(input_dictionary)
```

The output is in the form of a dictionary. The dictionary is keyed according to the output hardware specified in the `Layer()` interface (output connections)

**Save and load the parameters from a file**

```python
# Save the parameters to a file in some specific directory
nn.save_parameters_to_file(path_to_directory_with_file_name_in_quotes)

# Load the parameters from a file in some specific directory
nn.load_parameters_from_file(path_to_directory_with_file_name_in_quotes)
```

**Return and load the parameters in the form of a list**. The format of parameter list for each Neural Network is given in the [next section](#format-for-parameter-vector). 

**If any layer serves as an input layer, the weights specified in the parameter vector will be ignored for that layer.** Therefore, they should not be passed

```python
# Return the parameters
nn.return_parameters_as_vectors()

# Load the parameters
nn.load_parameters_from_vector(a_list_with_appropriate_format)
```

**View the output of each layer** The output of each and every layer(previous and current) is avaialable as a class variable of the Neural Network class

```python
nn.output_matrix
```

**Visualize the network** The computational graph of the Neural Network can also be graphically visualized

```python
nn.visualize(file_name, show)
# file_name is the path to the file that we want to generate
# show is a boolean to determine whether we want to view the file or not
``` 


### Format for parameter vector

```python
[list_of_parameters_of_layer_0, list_of_parameters_of_layer_1, list_of_parameters_of_layer_2, ...]
```

The parameter vector for setting the weights is a list of list. The type of Neural Network decides the layer that are to be used in the network. Each layer follows a particular pattern of how the weights are set from the vector.

#### Static Neural Network

For Static Neural Networks, all the layers follow the given format:

```python
[w_11, w_21, w_12, w_22, w_13, w_23, a_1g, a_2g, a_3g, a_1b, a_2b, a_3b, w_11, w_21, w_31, ...]
# Here, w_ij implies the weight between ith input node and jth output node.
# a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
```

#### Dynamic Neural Network

For Dynamic Neural Networks, all the layers follow the given format:

```python
[tc_1, tc_2, tc_3, w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, a_1g, a_2g, a_3g, a_1b, a_2b, a_3b, w_11, w_21, w_31, ...]
# Here, w_ij implies the weight between ith input node and jth output node.
# a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
# tc_i is the time constant of ith neuron of the current layer
```
