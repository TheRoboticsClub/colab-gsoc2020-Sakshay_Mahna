# Tests for activation functions

from neural_networks import activation_functions
import unittest
import numpy as np

# Unit Test Activation Class
class TestActivationFunctions(unittest.TestCase):
	def setUp(self):
		self.input_vector = np.array([1, 0, -0.1])
	
	def test_linear(self):
		Linear = activation_functions.LinearActivation(2, 1)
		np.testing.assert_almost_equal(Linear.calculate_activation(self.input_vector), np.array([0, -2, -2.2]))
		
	def test_step(self):
		Step = activation_functions.StepActivation(2, 1)
		np.testing.assert_almost_equal(Step.calculate_activation(self.input_vector), np.array([1, 0, 0]))
		
	def test_sigmoid(self):
		Sigmoid = activation_functions.SigmoidActivation(2, 1)
		np.testing.assert_almost_equal(Sigmoid.calculate_activation(self.input_vector), np.array([0.5, 0.11920292, 0.09975049]))
		
	def test_tanh(self):
		Tanh = activation_functions.TanhActivation(2, 1)
		np.testing.assert_almost_equal(Tanh.calculate_activation(self.input_vector), np.array([0, -0.96402758, -0.97574313]))
		
	def test_relu(self):
		Relu = activation_functions.ReluActivation(2, 1)
		np.testing.assert_almost_equal(Relu.calculate_activation(self.input_vector), np.array([0, 0, 0]))
		
	def test_maximum(self):
		Max = activation_functions.MaximumActivation()
		np.testing.assert_almost_equal(Max.calculate_activation(self.input_vector), np.array([1, 0, 0]))
		
if __name__ == "__main__":
	unittest.main()



