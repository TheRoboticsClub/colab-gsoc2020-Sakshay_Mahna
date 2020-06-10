# This part is specific to each user, not recommended to be copied
import sys
sys.path.append('./../')

# Tests for ctrnn

from neural_networks.ctrnn import CTRNN 
from neural_networks.activation_functions import SigmoidActivation
import unittest
import numpy as np

# Unit Test CTRNN Neural Network class
class TestCTRNN(unittest.TestCase):
	def setUp(self):
		self.activation_function = SigmoidActivation()

	def test_outputs(self):
		# The CTRNN network has 2 input nodes, 3 hidden nodes and 1 output node
		# THe time constants for first activation are [2, 3, 5]
		# The time constant for second activation is [2]
		# The time interval chosen is 0.6
		nn = CTRNN([[2, [2, 3, 5], self.activation_function], [3, [2], self.activation_function], [1]], 0.6)

		# Weights given in the same format as Static Weights
		nn.load_parameters_from_vector(np.array([2, 3, 5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 1, 1, 0, 2, 0.7, 0.8, 0.9, 0, 1, 0]))
		output = nn.forward_propagate([1, 1])
		np.testing.assert_almost_equal(output, np.array([0.1740721]))
		
		output = nn.forward_propagate([1, 1])
		np.testing.assert_almost_equal(output, np.array([0.3140605]))
		
	def test_wrong_input(self):
		nn = CTRNN([[2, [2, 3, 5], self.activation_function], [3, [2], self.activation_function], [1]], 0.6)
		nn.load_parameters_from_vector(np.array([2, 3, 5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 1, 1, 0, 2, 0.7, 0.8, 0.9, 0, 1, 0]))
		with self.assertRaises(ValueError):
			output = nn.forward_propagate([1, 1, 1])
		
	def test_wrong_parameters(self):
		nn = CTRNN([[2, [2, 3, 5], self.activation_function], [3, [2], self.activation_function], [1]], 0.6)
		with self.assertRaises(ValueError):
			nn.load_parameters_from_vector(np.array([2, 3, 5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 1, 1, 0, 2, 0.7, 0.8, 0.9, 0, 1]))


if __name__ == "__main__":
	unittest.main()


