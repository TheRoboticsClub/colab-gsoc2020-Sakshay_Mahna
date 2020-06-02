# A parent template class of neural network containing basic functionality
# of saving and loading weights from the file.

import pickle
import numpy as np

class NeuralNetwork:
	# Function to save the layer weights
	def save_weights_to_file(self, file_name):
		# Use pickle to save the layer_vector
		# This even saves all the previous input we were working on!
		with open(filename, 'wb') as f:
			pickle.dump(self.layer_vector, f)
			
	# Function to load the layer weights
	def load_weights_from_file(self, file_name):
		# Use pickle to load the layer_vector
		with open(file_name, 'rb') as f:
			self.layer_vector = pickle.load(f)
