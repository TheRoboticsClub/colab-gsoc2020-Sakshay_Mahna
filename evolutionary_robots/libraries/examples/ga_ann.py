# An exmample of a GA used to
# train Neural Network parameters

import sys
sys.path.append('./../')
import numpy as np

from genetic_algorithm.ga_nn import GeneticAlgorithmNN
from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
from neural_networks.activation_functions import IdentityActivation, SigmoidActivation, LinearActivation

# Define the Neural Network
# The Neural Network consists of 4 layers, input, hidden1, hidden2 and output
# The input layer consists of 1 neuron, takes input from INPUT and outputs to hidden1
# The hidden1 layer consists of 2 neurons, sigmoid activation and outputs to hidden2
# The hidden2 layer consists of 2 neurons, sigmoid activation and outputs to output
# The output layer has linear activation and outputs to OUTPUT

# The format followed is Layer(name_of_layer, number_of_neurons, activation_function, sensor_input, [list_of_output_connections])
inputLayer = Layer("inputLayer", 1, IdentityActivation(), "INPUT", ["hidden1Layer"])
hidden1Layer = Layer("hidden1Layer", 2, SigmoidActivation(), "", ["hidden2Layer"])
hidden2Layer = Layer("hidden2Layer", 2, SigmoidActivation(), "", ["outputLayer"])
outputLayer = Layer("outputLayer", 1, LinearActivation(), "", ["OUTPUT"])

# Generate the network
nn = ArtificialNeuralNetwork([
				inputLayer, 		# Layer 0 (Input Layer)
				hidden1Layer,		# Layer 1 (Hidden1 Layer)
				hidden2Layer, 		# Layer 2 (Hidden2 Layer)
				outputLayer		# Layer 3 (Output Layer)
			     ], "STATIC")
			     
			     
# Initialize the Genetic Algorithm
ga = GeneticAlgorithmNN(nn)

# Set the population size of the algorithm
ga.population_size = 50

# Set the mutation probability
ga.mutation_probability = 0.01

# Set the number of elites
ga.number_of_elites = 2

# Set the number of generations of the algorithm
ga.number_of_generations = 1000

# Pass the fitness function as an attribute
# The fitness function for the algorithm
def fitness_function(chromosome):
	input_vector = np.random.rand()
	actual_output_vector = input_vector * input_vector
	input_dictionary = {"INPUT": input_vector}
	
	output_dictionary = ga.calculate_output(input_dictionary, chromosome)
	output_vector = output_dictionary["OUTPUT"]
	
	fitness = -1 * abs(actual_output_vector - output_vector)
	return fitness[0]
	
ga.fitness_function = fitness_function

# Run the algorithm
print(ga.run())
	
