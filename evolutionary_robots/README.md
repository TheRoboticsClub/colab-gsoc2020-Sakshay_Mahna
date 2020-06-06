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

- Navigate to the directory `evolutionary_robots`. All the code developed and to be used is present in this directory.

## Neural Networks

### Expectations
The Neural Network Class will provide the student with a template of a Static or Dynamic Neural Network. Following are the features of the module:

- A template for variable number of layers, number of nodes in those layers and their activation function
- Load and retreive the weights of the Neural Network
- Load and Save the weights of the Neural Network to an external file
- A visual representation of the Neural Network in terms of the graph diagrams


The current directory consists of the unit tests. `neural_networks` directory consists of the Neural Network programs.

### Activation Functions

`activation_functions.py` consists of a collection of activation functions. The activation functions available are:

- Linear Activation
- Step Activation
- Sigmoid Activation
- Hyperbolic Tangent Activation
- ReLU Activation
- Maximum Value Activation

#### API Reference

