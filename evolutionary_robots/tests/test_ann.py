# This part is specific to each user, not recommended to be copied
import sys
sys.path.append('./../')

# Tests for ctrnn

from neural_networks.ann import ArtificialNeuralNetwork 
from neural_networks.activation_functions import LinearActivation
import unittest
import numpy as np

# Unit Test Artificial Neural Network class
class TestANN(unittest.TestCase):
	def setUp(self):
		self.activation_function = LinearActivation()

	def test_outputs(self):
		nn = ArtificialNeuralNetwork(4, [0, 0, 0, 1], [1, 1, 1, 2], [None, None, None, LinearActivation()], [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
		
		parameters = [np.array([]), np.array([]), np.array([]), np.array([0.5, 0.5, 0.5, 1, 1, 1, 0, 0, 1, 0])]
		nn.load_parameters_from_vector(parameters)
		output = nn.forward_propagate({0: [1], 1: [1], 2: [1]})
		
		np.testing.assert_almost_equal(output[3], np.array([1.5, 3.0]))
		#np.testing.assert_almost_equal(output, np.array([0.3140605]))


if __name__ == "__main__":
	unittest.main()


