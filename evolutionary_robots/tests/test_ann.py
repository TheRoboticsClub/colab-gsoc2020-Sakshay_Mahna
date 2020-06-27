# This part is specific to each user, not recommended to be copied
import sys
sys.path.append('./../')

# Tests for ctrnn

from neural_networks.ann import ArtificialNeuralNetwork 
from neural_networks.interface import Layer
from neural_networks.activation_functions import LinearActivation, IdentityActivation
import unittest
import numpy as np

# Unit Test Artificial Neural Network class
class TestANN(unittest.TestCase):
	def setUp(self):
		self.activation_function = LinearActivation()
		self.identity = IdentityActivation()

	def test_outputs(self):
		# Simple ANN with 2 input 3 output and Linear Activation
		inputLayer = Layer("inputLayer", 2, self.identity, "SENSOR", ["outputLayer"])
		outputLayer = Layer("outputLayer", 3, self.activation_function, "", ["GRIPPER"])
		nn = ArtificialNeuralNetwork([inputLayer, outputLayer], "STATIC")
		
		parameter_vector = [[], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, -1, -1, -1]]
		nn.load_parameters_from_vector(parameter_vector)

		input_dict = {"SENSOR": np.array([1.0, 1.0])}
		output = nn.forward_propagate(input_dict)
		
		np.testing.assert_almost_equal(output["GRIPPER"], np.array([1.3, 1.7, 2.1]))
		
		nn = ArtificialNeuralNetwork([inputLayer, outputLayer], "DYNAMIC")
		
		parameter_vector = [[], [1, 1, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, -1, -1, -1]]
		nn.load_parameters_from_vector(parameter_vector)
		
		output = nn.forward_propagate(input_dict)
		np.testing.assert_almost_equal(output["GRIPPER"], np.array([1, 1, 1]))
	
	def test_parameters(self):
		inputLayer = Layer("inputLayer", 2, self.identity, "SENSOR", ["outputLayer"])
		outputLayer = Layer("outputLayer", 3, self.activation_function, "", ["GRIPPER"])
		nn = ArtificialNeuralNetwork([inputLayer, outputLayer], "STATIC")
		input_dict = {"SENSOR": np.array([1.0, 1.0])}
		
		parameter_vector = [[], [1, 1, 1]]
		with self.assertRaises(ValueError):
			nn.load_parameters_from_vector(parameter_vector)
			nn.forward_propagate(input_dict)
			
	def test_static_exception(self):
		inputLayer = Layer("inputLayer", 2, self.identity, "SENSOR", ["hiddenLayer1", "hiddenLayer2"])		# Input Layer
		hiddenLayer1 = Layer("hiddenLayer1", 1, self.activation_function, "", ["hiddenLayer2", "outputLayer"])		# Hidden Layer 1
		hiddenLayer2 = Layer("hiddenLayer2", 1, self.activation_function, "", ["hiddenLayer1", "outputLayer"])		# Hidden Layer 2
		outputLayer = Layer("outputLayer", 1, self.activation_function, "", ["MOTOR"])					# Output Layer

		with self.assertRaises(Exception) as context:
			nn = ArtificialNeuralNetwork([
							inputLayer, 		# Layer 0 (Input Layer)
							hiddenLayer1, 		# Layer 1 (Hidden Layer)
							hiddenLayer2, 		# Layer 2 (Hidden Layer)
							outputLayer		# Layer 3 (Output Layer)
							 ], "STATIC")

if __name__ == "__main__":
	unittest.main()


