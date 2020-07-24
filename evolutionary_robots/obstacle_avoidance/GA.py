#!/usr/bin/python
import sys

import rospy
from std_srvs.srv import Empty
import glob
import os
import numpy as np
import shutil
import multiprocessing


class GA(object):
	"""
	Helper class to interface GeneticAlgorithmGazebo
	with MyAlgorithm
	
	...
	
	Parameters
	----------
	genetic_algorithm: GeneticAlgorithmGazebo object
	
	log_folder: string
		The destination of the folder to which logging should be done
		
	Methods
	-------
	initialize()
		This function is run when any button on is clicked
		
	return_stats()
		Function to return stats to the GUI
		
	calculate_output()
		Function to calculate output from neural network
		
	fitness_state()
		Operations to perform in FITNESS run_state
		
	print_state()
		Operations to perform in PRINT run_state
	
	selection_state()
		Operations to perform in SELECETION run_state
		
	crossover_state()
		Operations to perform in CROSSOVER run_state
		
	mutation_state()
		Operations to perform in MUTATION run_state
		
	next_state()
		Operations to perform in NEXT run_state
		
	end_state()
		Operations to perform in END run_state
	"""
	def __init__(self, genetic_algorithm, log_folder):
		"""
		Initialization function of the class
		"""
		self.run_state = "TRAIN"
		self.start = True
		self.log_folder = log_folder
		self.genetic_algorithm = genetic_algorithm
		self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		
	def initialize(self):
		"""
		This function is run when any button on 
		GUI is clicked
		"""
		if not os.path.exists(self.log_folder):
			os.makedirs(self.log_folder)
		elif(self.run_state == "TRAIN"):
			shutil.rmtree(self.log_folder)
			os.makedirs(self.log_folder)
			
		self.genetic_algorithm.generate_population()
		self.genetic_algorithm.generations.append(self.genetic_algorithm.population)
		
		# Initializing the Variables
		if(self.run_state == "CONTINUE"):
			files = glob.glob(self.log_folder + '/generation*[0-9].txt')
			self.genetic_algorithm.load_chromosome(files[-1])
			self.generation = self.genetic_algorithm.generation_start
			self.genetic_algorithm.test_network = self.genetic_algorithm.population[0]
			self.state = "FITNESS"
			
			try:
				best_individual = np.loadtxt(self.log_folder + '/current_best.txt', delimiter=' , ')
				self.genetic_algorithm.best_chromosome = best_individual
				best_fitness = np.loadtxt(self.log_folder + '/best_fitness.txt')
				self.genetic_algorithm.best_fitness = best_fitness
			except IOError:
				pass
			
		elif(self.run_state == "TRAIN"):
			self.generation = 1
			self.state = "SAVE"
			self.genetic_algorithm.test_network = self.genetic_algorithm.population[0]
			self.genetic_algorithm.save_chromosome(self.genetic_algorithm.population, self.log_folder + "/generation0%", header="Generation #0")
		
		elif(self.run_state == "TEST"):
			self.state = "TEST"
			self.generation = 0
			try:
				self.test_individual = np.loadtxt(self.log_folder + '/current_best.txt', delimiter=' , ')
			except IOError:
				print("File not found!")
				
			self.genetic_algorithm.test_network = self.test_individual
			
		elif(self.run_state[:6] == "RESUME"):
			percent = int(self.run_state[7:])
			
			try:
				self.genetic_algorithm.load_chromosome(self.log_folder + '/generation' + str(percent) + '%.txt')
			except IOError:
				print("File not found!")
				
			self.generation = self.genetic_algorithm.generation_start
			self.genetic_algorithm.test_network = self.genetic_algorithm.population[0]
			self.state = "FITNESS"
			
		# Print the legend
		legend = ["Generation", "Maximum Fitness", "Average Fitness", "Minimum Fitness"]
		print("{: <10} {: >20} {: >20} {: >20}".format(*legend))
		
		# Genetic Algorithm variables	
		self.fitness_iterations = 0
		self.fitness_vector = []
		self.individual_fitness = []
		self.individual_index = 0
		self.best_fitness = self.genetic_algorithm.best_fitness
		self.individual = self.genetic_algorithm.population[self.individual_index]
		self.evaluation_steps = self.genetic_algorithm.evaluation_steps
		self.genetic_algorithm.current_generation = self.generation - 1
		self.genetic_algorithm.generation_start = self.generation
		self.delete_process = None
		
		self.reset_simulation()
		
	def return_stats(self):
		"""
		Function to return stats to the GUI
		"""
		# Gather the stats and return them for
		# display in the GUI
		
		stats_array = []
		
		try:
			stats_array.append(self.generation - 1)
			stats_array.append(self.individual_index + 1)
			if(len(self.individual_fitness) == 0):
				stats_array.append(0)
			else:
				stats_array.append(self.individual_fitness[-1])
			stats_array.append(self.evaluation_steps - self.fitness_iterations)
			stats_array.append(self.best_fitness)
		
		except AttributeError:
			pass
			
		return stats_array
			
	def calculate_output(self, input_dictionary):
		"""
		Function to calculate output from neural network
		"""
		output = self.genetic_algorithm.test_output(input_dictionary)
		
		return output
		
	def save_state(self):
		"""
		Operations to perform in SAVE state
		"""
		self.genetic_algorithm.save_handler()
		
		# Delete the previous one
		# In a seperate process
		delete_process = multiprocessing.Process(target=self.genetic_algorithm.remove_chromosome, args=(self.log_folder + '/generation' + str(self.genetic_algorithm.current_generation - 1),))
						
		delete_process.start()
		
		self.state = "FITNESS"
		
	def fitness_state(self):
		"""
		Operations to perform in FITNESS state
		"""
		self.individual_fitness.append(self.genetic_algorithm.calculate_fitness(self.individual))
		self.fitness_iterations = self.fitness_iterations + 1
	
		if(self.fitness_iterations == self.evaluation_steps):
			self.individual_index = self.individual_index + 1
			if(self.individual_index < self.genetic_algorithm.population_size):
				self.individual = self.genetic_algorithm.population[self.individual_index]
				self.genetic_algorithm.test_network = self.individual
			self.fitness_vector.append(self.genetic_algorithm.determine_fitness(self.individual_fitness, self.individual))
			self.individual_fitness = []
			self.reset_simulation()
			self.fitness_iterations = 0
			
		if(self.individual_index == self.genetic_algorithm.population_size):
			self.state = "PRINT"
			self.genetic_algorithm.fitness_vector = np.array(self.fitness_vector)
			self.fitness_vector = []
			self.individual_index = 0
			
	def print_state(self):
		"""
		Operations to perform in PRINT state
		"""
		self.genetic_algorithm.generate_statistics()
		self.best_fitness = self.genetic_algorithm.best_fitness
		self.genetic_algorithm.fraction_save(self.generation)
		self.state = "SELECTION"
		
	def selection_state(self):
		"""
		Operations to perform in SELECTION state
		"""
		self.genetic_algorithm.selection()
		self.state = "CROSSOVER"
		
	def crossover_state(self):
		"""
		Operations to perform in CROSSOVER state
		"""
		self.genetic_algorithm.crossover()
		self.state = "MUTATION"
		
	def mutation_state(self):
		"""
		Operations to perform in MUTATION state
		"""
		self.genetic_algorithm.mutation()
		self.state = "NEXT"
		
	def next_state(self):
		"""
		Operations to perform in NEXT state
		"""
		self.genetic_algorithm.generations.append(self.genetic_algorithm.population)
		
		self.generation = self.generation + 1
		if(self.generation == self.genetic_algorithm.number_of_generations + 1):
			self.state = "END"
		else:
			self.individual = self.genetic_algorithm.population[self.individual_index]
			self.genetic_algorithm.current_generation = self.generation - 1
			self.state = "SAVE"
			
	def end_state(self):
		"""
		Operations to perform in END state
		"""
		self.genetic_algorithm.save_statistics(self.log_folder + '/stats')
		self.genetic_algorithm.save_chromosome(self.genetic_algorithm.best_chromosomes, self.log_folder + '/best_chromosomes')
		
		# Print the best fitness and return the chromosome
		print("The best fitness value acheived is: " + str(self.genetic_algorithm.best_fitness))
		print("Found in generation # " + str(self.genetic_algorithm.best_generation))
		
		
				
			
