# This part is specific to each user, not recommended to be copied
import sys
sys.path.append('./../')

from neural_networks.activation_functions import LinearActivation, SigmoidActivation
from neural_networks.dynamic_nn import DynamicNeuralNetwork
import unittest
import numpy as np

# Unit Test Dynamic Neural Network class
class TestDynamicNeuralNetwork(unittest.TestCase):
	def setUp(self):
		self.activation_function = LinearActivation()
	
	def test_outputs(self):
		nn = DynamicNeuralNetwork([[2, 2, [0], self.activation_function], [1]])
		nn.load_parameters_from_vector([1, 1, 1, 1, 0, 0, 1, 0])
		output = []
		output.append(nn.forward_propagate([1, 1]))
		output.append(nn.forward_propagate([1, 1]))
		output.append(nn.forward_propagate([1, 1]))
		output.append(nn.forward_propagate([1, 1]))
		np.testing.assert_almost_equal(np.array(output), np.array([[1], [3], [5], [7]]))
		
		nn = DynamicNeuralNetwork([[1, 2, [1, 2], self.activation_function], [2, 1, [0, 1], self.activation_function], [2, 1, [0], SigmoidActivation()], [1]])
		nn.load_parameters_from_vector([2, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.1, 0.2, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0.3, 0.4, 0.5, 0.6, 0, 1, 1, 0, 1, 0.7, 0.8, 0, 1, 0])
		output = []
		output.append(nn.forward_propagate([0.5]))
		output.append(nn.forward_propagate([0.5]))
		np.testing.assert_almost_equal(np.array(output), np.array([[0.8352085], [0.9980762]]))
		
	def test_wrong_input(self):
		nn = DynamicNeuralNetwork([[2, 2, [0], self.activation_function], [1]])
		nn.load_parameters_from_vector([1, 1, 1, 1, 0, 0, 1, 0])
		with self.assertRaises(ValueError):
			output = nn.forward_propagate([1, 1, 1])
		
	def test_wrong_parameters(self):
		nn = DynamicNeuralNetwork([[2, 1, [0], self.activation_function], [1]])
		with self.assertRaises(ValueError):
			nn.load_parameters_from_vector([1, 1, 1, 0, 0])
		
		


if __name__ == "__main__":
	unittest.main()

