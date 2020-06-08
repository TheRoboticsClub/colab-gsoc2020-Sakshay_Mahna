# Evolutionary Robots

## Description
Exercises on Evolutionary Robotics for JdeRobot Robotics Academy

## Prerequisites
The libraries are developed and tested in Python 2.7.17, ROS Melodic, Ubuntu 18.04. These are the system requirements that JdeRobot currently specifies for users to run their Robotics-Academy software.

The dependencies of the project are:
```
numpy==1.16.5
graphviz==0.14
```

In order to install the dependencies, use the command:
```bash
pip install -r requirements.txt
```

## Installation
- Clone the Github Repository

```bash
git clone https://github.com/TheRoboticsClub/colab-gsoc2020-Sakshay_Mahna
```

- Follow the API Reference and Examples to use the library.

## API Reference

### How to create a Neural Network?

Import the Activation Function and Neural Networks along with Numpy(optional). The `neural_networks.activation_functions` consist of a library of activation functions. The `neural_networks.static_nn` contatins the StaticNeuralNetwork class.

```python
from neural_networks.static_nn import StaticNeuralNetwork
from neural_networks.activation_functions import SigmoidActivation
import numpy as np
```

Initialize the Neural Network Object. For instance, we have a Perceptron with 2 input nodes and 1 output node along with a sigmoid activaiton function. Pass a **list of list** of Layer Parameters while initializing the Neural Network object.

```python
percpetron = StaticNeuralNetwork([[2, SigmoidActivation(1, 0)], [1]])
```

Load self-generated parameters into the Neural Network. Assume, in this case the weights of the input are 0.5 and 0.5, the bias is 1. And the gain of the Sigmoid Activation is 1 and the offset is 0. Note that the parameters are to be given in a format specific to each Neural Network Class.

```python
perceptron.load_parameters_from_vector([0.5, 0.5, 1, 1, 0])
```

Generate the output. In this case, the inputs to both the nodes is 1.

```python
output = perceptron.forward_propagate([1, 1])
```

For more examples, check the `examples` directory.

The following covers the specifics of each of the Neural Network classes.

### Activation Function

`activation_functions.py` consists of a collection of activation functions. The activation functions available are:

| Activation Function   | Class to import     | Applicable Parameters |
|-----------------------|---------------------|-----------------------|
| Linear Activation     | LinearActivation()  | Beta and Theta		  |
| Step Activation		| StepActivation()	  | Theta				  |
| Sigmoid Activation	| SigmoidActivation() | Beta and Theta		  |
| Hyperbolic Tangent	| TanhActivation()	  | Beta and Theta		  |
| ReLU Activation		| ReluActivation()	  | Beta and Theta		  |
| Maximum Value			| MaximumActivation() | None				  |

**Note**: In case of activation functions with parameters less than 2. The format of parameter input of Neural Networks remains the same. Therefore, they are to be given any arbitrary values(for correct syntax). These values do not affect the activation function.

There are two ways to set the parameters of the Activation Function. Either while initializing the object or while explicitly by a function call. The default values of beta and theta are taken to be as 1 and 0 respectively.

```python
# Default Initialization
activation = LinearActivation()

# While initializing the object. Set beta as 2 and theta as 1 
activation = LinearActivation(2, 1)

# Explicitly assigining the values of beta as 1 and theta as 0
activation.set_parameters(1, 0)
```

The input to the activation function should be a numpy array. Calculate the activation as follows. Building on from the example above.

```python
print(activation.calculate_activation(np.array([-1, 0, 1])))
# The output is a numpy array [-1, 0, 1]
```

### Neural Network
The supported neural networks are:

| Neural Networks   	| Class to import     	|
|-----------------------|-----------------------|
| Static 			    | StaticNeuralNetwork() |


Methods supported by all the Neural Networks

Generate a Neural Network with a given number of layers, number of neurons in each layer and the activation function. The format of the list to be passed is given in the next section for each neuron.

```python
nn = StaticNeuralNetwork(a_list_with_appropriate_format_for_initialization)
```

Generate the output of a Neural Network. The input vector to be passed is a list, of dimensions as specified by the user during initialization.

```python
nn.forward_propagate(input_vector)
```

Save and load the parameters of a Neural Network from a file.

```python
# Save the parameters to a file in some specific directory
nn.save_parameters_to_file(path_to_directory_with_file_name_in_quotes)

# Load the parameters from a file in some specific directory
nn.load_parameters_from_file(path_to_directory_with_file_name_in_quotes)
```

Return and load the parameters of a Neural Network in the form of a list. The format of parameter list for each Neural Network is given in the next section

```python
# Return the parameters
nn.return_parameters_as_vectors()

# Load the parameters
nn.load_parameters_from_vector(a_list_with_appropriate_format)
```

Generate the visual representation of the Neural Network. The representation is generated in the form of pdf and saved in the directory `representations`

```python
# See the representation right away
nn.generate_visual(filename, True)

# Do not see the representation
nn.generate_visual(filename)
```


### Static Neural Network

**Format for Initialization**

```
[[number_of_nodes_in_first_layer(input layer), activation_class], [number_of_nodes_in_second_layer, activation_class], ..., [number_of_output_nodes]]
```

**Format for parameter vector**

```
[w_11, w_21, w_12, w_22, w_13, w_23, b_1, b_2, b_3, a_1g, a_1b, a_2g, a_2b, a_3g, a_3b, w_11, w_21, w_31, b_1, ...]
Here, w_ij implies the weight between ith input node and jth output node. b_i is the bias for the ith output node.
a_ib is the bias activation parameter of ith output node and a_ig is the gain activation parameter of ith output node.
```




