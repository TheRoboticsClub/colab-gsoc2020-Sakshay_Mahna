# Examples

### Example 1

`perceptron.py` Contains an example of 2 layers(Input and Output) with Linear Activation. The input layer consists of 2 neurons and the output layer consists of 3 neurons. The output layer is Simple or Continuous Time Recurrent.

### Example 2

`recurrent.py` Contains an example of recurrent relation between 2 hidden layers. The input consists of 1 neuron. There are 2 hidden layers with 1 neuron each and an output layer with 1 neuron. The hidden layers are assumed to be at the same level. In order to make these 2 layers communicate with each other, a delay has to be specified for one of the connection according to the order of execution. The example shows the correct and the wrong method.

### Example 3

`order.py` Contains an example of **order of execution** and how to adjust it. There are 3 layers in the network, input, hidden and output. All of them consist of 2 neurons each. The output layer of the network is expected to take input from the input and hidden layer and the hidden layer is supposed to take input from the input layer. Therefore, the correct order of execution is input -> hidden -> output. The example shows the correct and the wrong order of execution.

### Example 4

`complex.py` Contains an example showing how to design a complex hypothetical Neural Network, which takes associative inputs as well in accordance with a gain. There are 6 layers in total, of which 4 are hidden and 1 input and 1 output layer. The 5th layer is connected as a recurrence to the 2nd layer. However, the weights are adjusted such that the recurrence is not able to show it's effect.

### Example 5

`large.py` Contains an example showing how to design a Large Static Neural Network, which has a large number of layers. There are 30 Layers in total, 2 of which are input and output and the other 28 are hidden. In order to pass the parameters of the network a loop structure is used.
