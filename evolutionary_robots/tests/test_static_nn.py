# This part is specific to each user, not recommended to be copied
import sys
sys.path.append('./../')

# Tests for static neural networks

from neural_networks.static_nn import StaticNeuralNetwork 
from neural_networks.activation_functions import LinearActivation
import unittest
import numpy as np

# Unit Test Static Neural Network class
class TestStaticNeuralNetwork(unittest.TestCase):
	def setUp(self):
		self.activation_function = LinearActivation()
	
	def test_outputs(self):
		perceptron = StaticNeuralNetwork([[2, self.activation_function], [1]])
		perceptron.load_parameters_from_vector([0.5, 0.5, 1, 1, 0])		# weights, bias, activation
		output = perceptron.forward_propagate([1, 1])
		np.testing.assert_almost_equal(output, np.array([2]))
		
		nn = StaticNeuralNetwork([[2, self.activation_function], [3, self.activation_function], [3, self.activation_function], [1]])
		nn.load_parameters_from_vector([0, 0.25, 0.5, 0.5, 0.75, 1.0, 1, 0, 0, 1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0, 0, 1, 1, 0, 0.75, 0.5, 0.25, 0, 1, 0])
		output = nn.forward_propagate([1, 1])
		np.testing.assert_almost_equal(output, np.array([2.725]))
		
	def test_wrong_input(self):
		perceptron = StaticNeuralNetwork([[2, self.activation_function], [1]])
		perceptron.load_parameters_from_vector([0.5, 0.5, 1, 1, 0])		# weights, bias, activation
		with self.assertRaises(ValueError):
			output = perceptron.forward_propagate([1, 1, 1])
		
	def test_wrong_parameters(self):
		nn = StaticNeuralNetwork([[2, self.activation_function], [3, self.activation_function], [1]])
		with self.assertRaises(ValueError):
			nn.load_parameters_from_vector([0.5, 0, 0.5, 0, 0.5, 0, 1, 0, 0, 0.5, 0, 0, 1])


if __name__ == "__main__":
	unittest.main()
