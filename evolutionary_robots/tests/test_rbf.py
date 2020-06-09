# This part is specific to each user, not recommended to be copied
import sys
sys.path.append('./../')

# Tests for rbf neural networks

from neural_networks.rbf_nn import GaussianRBFNetwork 
import unittest
import numpy as np

# Unit Test RBF Neural Network class
class TestRBFNeuralNetwork(unittest.TestCase):
	def test_outputs(self):
		nn = GaussianRBFNetwork(2, 2, 1)
		# The center of first node is (0.25, 0.75)
		# The center of second node is (0.0, 1.0)
		# The weights of second layer are [[0.5], [0.5]]
		nn.load_parameters_from_vector([0.25, 0.75, 0.0, 1.0, 0.5, 0.5])		# weights, bias, activation
		output = nn.forward_propagate([1, 1])
		np.testing.assert_almost_equal(output, np.array([0.4515704]))
		
	def test_wrong_input(self):
		nn = GaussianRBFNetwork(2, 2, 1)
		nn.load_parameters_from_vector([0.25, 0.75, 0.0, 1.0, 0.5, 0.5])		# weights, bias, activation
		with self.assertRaises(ValueError):
			output = nn.forward_propagate([1, 1, 1])
		
	def test_wrong_parameters(self):
		nn = GaussianRBFNetwork(2, 3, 1)
		with self.assertRaises(ValueError):
			nn.load_parameters_from_vector([0.5, 0, 0.5, 0, 0.5, 0, 1, 0, 0, 0.5, 0, 0, 1])


if __name__ == "__main__":
	unittest.main()

