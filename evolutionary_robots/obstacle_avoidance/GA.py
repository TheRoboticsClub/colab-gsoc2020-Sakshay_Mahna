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
		self.pause = False
		self.log_folder = log_folder
		self.genetic_algorithm = genetic_algorithm
		self.reset_simulation = (rospy.ServiceProxy('ga1/gazebo/reset_simulation', Empty),
								 rospy.ServiceProxy('ga2/gazebo/reset_simulation', Empty),
								 rospy.ServiceProxy('ga3/gazebo/reset_simulation', Empty),
								 rospy.ServiceProxy('ga4/gazebo/reset_simulation', Empty),
								 rospy.ServiceProxy('ga5/gazebo/reset_simulation', Empty))	
	
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
		if(self.run_state[:8] == "CONTINUE"):
			generation_number = int(self.run_state[8:])
			self.genetic_algorithm.load_generation(generation_number)
			self.generation = self.genetic_algorithm.generation_start
			self.genetic_algorithm.test_network = (0, self.genetic_algorithm.population[0])
			self.genetic_algorithm.test_network = (1, self.genetic_algorithm.population[1])
			self.genetic_algorithm.test_network = (2, self.genetic_algorithm.population[2])
			self.genetic_algorithm.test_network = (3, self.genetic_algorithm.population[3])
			self.genetic_algorithm.test_network = (4, self.genetic_algorithm.population[4])
			self.state = "FITNESS"
			
			try:
				best_individual = self.genetic_algorithm.load_chromosome(self.log_folder + '/current_best')
				self.genetic_algorithm.best_chromosome = best_individual
				best_fitness = self.genetic_algorithm.load_chromosome(self.log_folder + '/best_fitness')
				self.genetic_algorithm.best_fitness = best_fitness[0]
				self.genetic_algorithm.best_generation = int(best_fitness[1])
				best_chromosomes = self.genetic_algorithm.load_chromosome(self.log_folder + '/best_chromosomes')
				if(generation_number == 1):
					self.genetic_algorithm.best_chromosomes.append(best_chromosomes.tolist())
				else:
					self.genetic_algorithm.best_chromosomes = best_chromosomes.tolist()
			except IOError:
				pass
			
		elif(self.run_state == "TRAIN"):
			self.generation = 1
			self.state = "SAVE"
			self.genetic_algorithm.test_network = (0, self.genetic_algorithm.population[0])
			self.genetic_algorithm.test_network = (1, self.genetic_algorithm.population[1])
			self.genetic_algorithm.test_network = (2, self.genetic_algorithm.population[2])
			self.genetic_algorithm.test_network = (3, self.genetic_algorithm.population[3])
			self.genetic_algorithm.test_network = (4, self.genetic_algorithm.population[4])
			
			self.genetic_algorithm.save_chromosome(self.genetic_algorithm.population, 
								   self.log_folder + "/generation0", header="Generation #0")
		
		elif(self.run_state[:4] == "TEST"):
			self.state = "TEST"
			test_number = int(self.run_state[4:])
			self.generation = 0
			try:
				test_population = self.genetic_algorithm.load_chromosome(self.log_folder + "/best_chromosomes")
				self.test_individual = test_population[test_number]
			except IOError:
				print("File not found!")
				
			self.genetic_algorithm.test_network = (0, self.test_individual)
			
		# Print the legend
		legend = ["Generation", "Maximum Fitness", "Average Fitness", "Minimum Fitness"]
		print("{: <10} {: >20} {: >20} {: >20}".format(*legend))
		
		# Genetic Algorithm variables	
		self.fitness_iterations = 0
		self.fitness_vector = []
		self.individual_fitness = [[] for _ in range(5)] 	# Good lesson: This is different than [[]] * 5
		self.individual_index = 0
		self.best_fitness = self.genetic_algorithm.best_fitness
		self.individual = [self.genetic_algorithm.population[self.individual_index],
						   self.genetic_algorithm.population[self.individual_index + 1],
						   self.genetic_algorithm.population[self.individual_index + 2],
						   self.genetic_algorithm.population[self.individual_index + 3],
						   self.genetic_algorithm.population[self.individual_index + 4]]
		self.evaluation_steps = self.genetic_algorithm.evaluation_steps
		self.genetic_algorithm.current_generation = self.generation - 1
		self.genetic_algorithm.generation_start = self.generation
		self.delete_process = None
		
		if(self.state == "TEST"):
			self.reset_simulation[0]()
		else:
			self.reset_simulation[0]()
			self.reset_simulation[1]()
			self.reset_simulation[2]()
			self.reset_simulation[3]()
			self.reset_simulation[4]()
			
		
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
			stats_array.append(self.best_fitness)
		
		except AttributeError:
			pass
			
		return stats_array
			
	def calculate_output(self, input_dictionary, index):
		"""
		Function to calculate output from neural network
		"""
		
		output = self.genetic_algorithm.test_output(input_dictionary, index)
		
		return output
		
	def save_state(self):
		"""
		Operations to perform in SAVE state
		"""
		self.genetic_algorithm.save_handler()
		
		# Delete the previous one
		# In a seperate process
		if(self.generation % self.genetic_algorithm.replay_number != 2):
			delete_process = multiprocessing.Process(
							 target=self.genetic_algorithm.remove_chromosome, 
							 args=(self.log_folder + '/generation' + str(self.genetic_algorithm.current_generation - 1),))
							
			delete_process.start()
		
		self.state = "FITNESS"
		
	def fitness_state(self):
		"""
		Operations to perform in FITNESS state
		"""
		self.individual_fitness[0].append(self.genetic_algorithm.calculate_fitness(0))
		self.individual_fitness[1].append(self.genetic_algorithm.calculate_fitness(1))
		self.individual_fitness[2].append(self.genetic_algorithm.calculate_fitness(2))
		self.individual_fitness[3].append(self.genetic_algorithm.calculate_fitness(3))
		self.individual_fitness[4].append(self.genetic_algorithm.calculate_fitness(4))
		
		self.fitness_iterations = self.fitness_iterations + 1
	
		if(self.fitness_iterations == self.evaluation_steps):
			self.individual_index = self.individual_index + 5
			if(self.individual_index < self.genetic_algorithm.population_size):
				self.individual = [[]] * 5
				self.individual[0] = self.genetic_algorithm.population[self.individual_index]
				self.individual[1] = self.genetic_algorithm.population[self.individual_index + 1]
				self.individual[2] = self.genetic_algorithm.population[self.individual_index + 2]
				self.individual[3] = self.genetic_algorithm.population[self.individual_index + 3]
				self.individual[4] = self.genetic_algorithm.population[self.individual_index + 4]
				self.genetic_algorithm.test_network = (0, self.individual[0])
				self.genetic_algorithm.test_network = (1, self.individual[1])
				self.genetic_algorithm.test_network = (2, self.individual[2])
				self.genetic_algorithm.test_network = (3, self.individual[3])
				self.genetic_algorithm.test_network = (4, self.individual[4])
			self.fitness_vector.append(self.genetic_algorithm.determine_fitness(self.individual_fitness[0], self.individual[0]))
			self.fitness_vector.append(self.genetic_algorithm.determine_fitness(self.individual_fitness[1], self.individual[1]))
			self.fitness_vector.append(self.genetic_algorithm.determine_fitness(self.individual_fitness[2], self.individual[2]))
			self.fitness_vector.append(self.genetic_algorithm.determine_fitness(self.individual_fitness[3], self.individual[3]))
			self.fitness_vector.append(self.genetic_algorithm.determine_fitness(self.individual_fitness[4], self.individual[4]))
			self.individual_fitness = [[] for _ in range(5)]
			self.reset_simulation[0]()
			self.reset_simulation[1]()
			self.reset_simulation[2]()
			self.reset_simulation[3]()
			self.reset_simulation[4]()
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
		
		# Print the best fitness and return the chromosome
		print("The best fitness value acheived is: " + str(self.genetic_algorithm.best_fitness))
		print("Found in generation # " + str(self.genetic_algorithm.best_generation))
		
		
				
			
