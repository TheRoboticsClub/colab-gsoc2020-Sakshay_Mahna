# This part is specific to each user, not recommended to be copied
import sys
sys.path.append('./../')

# Tests for ctrnn

from neural_networks.ann import ArtificialNeuralNetwork 
from neural_networks.interface import Layer
from neural_networks.activation_functions import LinearActivation
import unittest
import numpy as np

# Unit Test Artificial Neural Network class
class TestANN(unittest.TestCase):
	def setUp(self):
		self.activation_function = LinearActivation()

	def test_outputs(self):
		# Simple ANN with 2 input 3 output and Linear Activation
		inputLayer = Layer(2, 0, None, [], [1])
		outputLayer = Layer(3, 1, self.activation_function, [0], [])
		nn = ArtificialNeuralNetwork([inputLayer, outputLayer])
		
		parameter_vector = [[], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, 1, 0]]
		nn.load_parameters_from_vector(parameter_vector)

		input_dict = {0: np.array([1.0, 1.0])}
		output = nn.forward_propagate(input_dict)
		
		np.testing.assert_almost_equal(output[1], np.array([1.3, 1.7, 2.1]))
		
		outputLayer.type_of_layer = 2
		nn = ArtificialNeuralNetwork([inputLayer, outputLayer])
		
		parameter_vector = [[], [1, 1, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, 1, 0]]
		nn.load_parameters_from_vector(parameter_vector)
		
		output = nn.forward_propagate(input_dict)
		np.testing.assert_almost_equal(output[1], np.array([0.013, 0.017, 0.021]))
	
	def test_parameters(self):
		inputLayer = Layer(2, 0, None, [], [1])
		outputLayer = Layer(3, 1, self.activation_function, [0], [])
		nn = ArtificialNeuralNetwork([inputLayer, outputLayer])
		
		parameter_vector = [[], [0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, 1, 0]]
		with self.assertRaises(ValueError):
			nn.load_parameters_from_vector(parameter_vector)
			
	def test_set_gains(self):
		inputLayer = Layer(2, 0, None, [], [1])
		outputLayer = Layer(3, 1, self.activation_function, [0], [])
		nn = ArtificialNeuralNetwork([inputLayer, outputLayer])
		
		parameter_vector = [[], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, 1, 0]]
		nn.load_parameters_from_vector(parameter_vector)
		
		input_dict = {0: np.array([1.0, 1.0]), 1: np.array([1.0, 1.0, 1.0])}
		output1 = nn.forward_propagate(input_dict)
		output2 = nn.forward_propagate(input_dict)
		
		outputLayer.gains = [2.0, 2.0, 2.0]
		nn = ArtificialNeuralNetwork([inputLayer, outputLayer])
		
		parameter_vector = [[], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1, 1, 1, 0]]
		nn.load_parameters_from_vector(parameter_vector)
		
		output3 = nn.forward_propagate(input_dict)
		
		np.testing.assert_array_equal(output1[1], output2[1])
		np.testing.assert_array_less(output1[1], output3[1])
		
	def test_order_execution(self):
		inputLayer = Layer(2, 0, None, [], [1, 2])
		outputLayer = Layer(2, 1, self.activation_function, [0, 2], [])
		hiddenLayer = Layer(2, 1, self.activation_function, [0], [1])
		nn = ArtificialNeuralNetwork([inputLayer, outputLayer, hiddenLayer])
		
		parameter_vector = [[], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 0]]
		nn.load_parameters_from_vector(parameter_vector)
		
		input_dict = {0: np.array([1.0, 1.0])}
		output1 = nn.forward_propagate(input_dict)
		
		nn.order_of_execution = [0, 2, 1]
		output2 = nn.forward_propagate(input_dict)
		
		np.testing.assert_array_less(output1[1], output2[1])

if __name__ == "__main__":
	unittest.main()


