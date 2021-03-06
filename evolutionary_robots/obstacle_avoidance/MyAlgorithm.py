import sys
import math
import numpy as np
sys.path.append('./../libraries')
from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
from neural_networks.activation_functions import *

LOG_FOLDER = './log'

# Fill Parameters for Genetic Algorithm
POPULATION_SIZE = 
NUMBER_OF_GENERATIONS = 
MUTATION_PROBABILITY = 
EVALUATION_STEPS = 

def fitness_function(left_motor_speed, right_motor_speed, infrared):
	# Code the fitness function here
	
	fitness = 0
	
	return fitness
	
def define_neural_network():
	# Define the layers
	# Layer(name_of_layer, number_of_neurons, activation_function, sensor_inputs, list_of_output_layer_names)
	inputLayer = Layer("inputLayer", 8, IdentityActivation(), "INFRARED", ["outputLayer"])
	outputLayer = Layer("outputLayer", 2, TanhActivation(), "", ["MOTORS"])
	
	# Define the neural network
	neural_network = ArtificialNeuralNetwork([inputLayer, outputLayer], "STATIC")
	
	return neural_network
