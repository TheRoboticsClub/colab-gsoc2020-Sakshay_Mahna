import sys
sys.path.append('./../')

from genetic_algorithm.ga import GeneticAlgorithm
import numpy as np
import unittest

class TestGA(unittest.TestCase):
	# FItness FUnction for test
	def fitness_function(self, chromosome):
		return np.sum(chromosome)
	
	# The algorithm should give an error
	# if we run without setting the fitness function
	def test_fitness(self):
		ga = GeneticAlgorithm()
		with self.assertRaises(AttributeError):
			ga.run()
	
	# Simple Test RUn
	def test_run(self):
		ga = GeneticAlgorithm()
		ga.fitness_function = self.fitness_function
		ga.run()
		ga.plot_fitness('plot')
		
if __name__ == "__main__":
	unittest.main()
