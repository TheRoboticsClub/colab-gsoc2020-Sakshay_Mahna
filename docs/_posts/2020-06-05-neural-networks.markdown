---
layout: post
title:  "Week #1: Neural Networks"
date:   2020-06-05 10:30:00 +0530
categories: neural-networks
comments: true
---
This week's task involved coding the Neural Network library.

## Where does this fit in the project?
The Neural Network library should contain some templates of configurable Neural Networks. The networks can be configured in terms of the number of layers, number of nodes and the activation function. While solving the exercise, the student is expected to **evolve** the weights of the Neural Network of the robot, to allow it to learn the task at hand! The Neural Network; student uses, can be our configurable templates, some implementation of their own or some other *third party* library. Our library provides access to the Static and Dynamic Neural Networks along with some other networks like the **Continuous Time Recurrent Neural Network** or **Radial Basis Function Network**. Our library is built particularly for training the weights through Genetic Algorithms. Hence, easier to fit in the solution of the exercise.

![How Neural Networks should be seen!](https://miro.medium.com/max/1890/1*wHLVHCmwKAgWK9a4npHhEA.jpeg)

*How Neural Networks should be seen!*

## Outcome of this Week
The Neural Network library is **almost** complete, apart from some bug testing and modifications. These modifications and bugs can be tackled while going through the project and adapting according to the overall requirments. The library currently consists of the following:

- Activation Function Library
- Static Neural Network Template
- Dynamic Neural Network Template
- Continuous Time Recurrent Neural Network Template
- Radial Basis Function Network Template

These networks have been tested on various random inputs, fed along with variable weights and bias. Each network has different weight parameters, such as time constants for CTRNN, center vector for RBFNetworks and so on! To provide the weight parameters to the network a **vector** is to be passed, which more or less incorporates the behaviour of **Genes** in our exercises. Moreover, for the GUI template of the JdeRobot exercise, the visual representation of each of the network is also provided. The representation depicts the layers, nodes and the recurrent connections. 

## Logic of the Code
Most of the inspiration and logic is owed to [NeuroLabs](https://github.com/zueve/neurolab) and this [Graphviz Post](https://tgmstat.wordpress.com/2013/06/12/draw-neural-network-diagrams-graphviz/). A Layer forms the foundation of a Neural Network. Hence, for each of the Network consists of a collection of particular kinds of layers. For example, SNN networks consist of Static Layer, CTRNN networks consist of an Euler Stepping layer and so on. Hence, a Neural Network class consists of a collection of Layer classes. The Neural Network propagates the input forward into the layers, to generate the final output. For DNN and CTRNN, the previous states have to be saved to calculate the current output of the network. The RBF networks need a different unit to calculate the difference between points and calculate their radial basis function.

**Numpy** is the backbone behind this library. Neural Networks are just a systematic application of Matrix Operations. Numpy is a magic library. It's as if the writers of numpy thought of every possible case that could arise from the matrix operations. Any matrix operation a human can think of, numpy has it implemented already! This really simplified this week's task.

## Problems and their Solutions
Overall, the first week of coding was really enjoyable. Here is a list of problems and the solutions I came up for the code:

- **CTRNN**: Honestly, I didn't completely understand the concept behind CTRNN, before the start of the coding period and was really hesistant in coding it! However, my mentor helped me in understanding it better! Here is a [good link](https://neat-python.readthedocs.io/en/latest/ctrnn.html) Continuous Time Recurrent Neural Networks work using a differential equation and hence try simulating a continuous time behaviour. In order for the computer to understand it, we need to discretize the equation. One very common method of discretizing a differential equation is called the [First Order Euler Solution]. Using this solution, we can easily get a simple equation that works to generate the output of our network.

- **Visual Representation**: The visual representation of the networks was the most problematic. But following this [post](https://tgmstat.wordpress.com/2013/06/12/draw-neural-network-diagrams-graphviz/) made it really simple to code the representation using Graphviz library. All the networks except DNNs were easily represented. The problem with DNNs are their recurrent connections and delay systems. It is quite difficult to represent those. However, the current solution works for now!

- **Debugging and Testing**: This was a difficult task. Even after testing the networks, I am still not completely sure that they are error free. Only time will tell!

![Visual Representation of our DNN](/assets/img/DNN_repr.png)

*Visual Representation of our DNN*


