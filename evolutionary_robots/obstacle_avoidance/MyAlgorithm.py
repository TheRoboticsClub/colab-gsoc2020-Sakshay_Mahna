#!/usr/bin/python
#-*- coding: utf-8 -*-
import threading
import time
import sys
import os
import rospy
import multiprocessing
from std_srvs.srv import Empty
from datetime import datetime

import math
import cv2
import numpy as np

sys.path.append('./../libraries/')
from genetic_algorithm.ga_simulation import GeneticAlgorithmGazebo
from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
from neural_networks.activation_functions import *

time_cycle = 80

class MyAlgorithm(threading.Thread):
    def __init__(self, sensor, motors):
        # Initializing the Algorithm object
        self.sensor = sensor
        self.motors = motors
        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        self.threshold_sensor_lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)
        
    def initialize(self):
        # Initializing the Neural Network
        self.neural_network = self.define_neural_network()
        self.ga = self.define_genetic_algorithm()
        
        # Genetic Algorithm variables
        self.state = "FITNESS"
        self.fitness_iterations = 0
        self.fitness_vector = []
        self.individual_fitness = []
        self.individual_index = 0
        self.individual = self.ga.population[self.individual_index]
        self.generation = 1
        self.ga.current_generation = self.generation - 1
        self.ga.generation_start = self.generation
        self.delete_process = None
        
    
    def fitness_function(self, chromosome):
    	# The fitness function
    	linear_speed = self.motors.getV()
    	rotation_speed = self.motors.getW()
    	infrared = np.max(self.getRange())
    	
    	fitness = linear_speed * (1 - math.sqrt(abs(rotation_speed))) * (2 - infrared)
    	
    	return fitness 
    	
    
    def define_neural_network(self):
    	# Define the layers
    	inputLayer = Layer("inputLayer", 8, IdentityActivation(), "INFRARED", ["outputLayer"])
    	outputLayer = Layer("outputLayer", 2, LinearActivation(), "", ["MOTORS"])
    	# Define the Neural Network
    	neural_network = ArtificialNeuralNetwork([inputLayer, outputLayer], "STATIC")
    	
    	return neural_network
    	
    def define_genetic_algorithm(self):
    	# Define the Genetic Algorithm
    	ga = GeneticAlgorithmGazebo(self.neural_network)
    
    	# Define the genetic algorithm
    	ga.population_size = 10
    	ga.number_of_generations = 100   
    	ga.mutation_probability = 0.01
    	ga.evaluation_time = 300
    	ga.number_of_elites = 0
    	ga.fitness_function = self.fitness_function
    	
    	# Generate the log directory
    	if not os.path.exists('./log'):
    		os.makedirs('./log')
    	
    	# Generate a random population
    	ga.generate_population()
    	ga.generations.append(ga.population)
    	ga.save_chromosome(ga.population, './log/generation0%', header="Generation #0")
    	
        # Print the legend
        legend = ["Generation", "Maximum Fitness", "Average Fitness", "Minimum Fitness"]
        print("{: <10} {: >20} {: >20} {: >20}".format(*legend))
    	
    	return ga
    	
    
    def getRange(self):
        self.lock.acquire()
        values = self.sensor.data.values
        self.lock.release()
        return values
        
    def reset_simulation(self):
        reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        reset()

    def run (self):
        self.initialize()
    	while(not self.kill_event.is_set()):
    		start_time = datetime.now()
    		
    		if(not self.stop_event.is_set()):
    			self.algorithm()
    			
    		finish_time = datetime.now()
    		
    		dt = finish_time - start_time
    		ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    		if(ms < time_cycle):
    			time.sleep((time_cycle - ms) / 1000.0)

    def stop (self):
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()

    def algorithm(self):
    	if(self.state == "FITNESS"):
    		output = self.ga.calculate_output({"INFRARED": self.getRange()}, self.individual)["MOTORS"]
    		self.motors.sendV(0.01 * (output[0] + output[1]))
    		self.motors.sendW(0.01 * (output[0] - output[1]))
    		self.individual_fitness.append(self.ga.calculate_fitness(self.individual))
    		self.fitness_iterations = self.fitness_iterations + 1
    		
    		if(self.fitness_iterations == self.ga.evaluation_steps):
    			self.individual_index = self.individual_index + 1
    			if(self.individual_index < self.ga.population_size):
    			    self.individual = self.ga.population[self.individual_index]
    			self.fitness_vector.append(self.ga.determine_fitness(self.individual_fitness, self.individual))
    			self.individual_fitness = []
    			self.reset_simulation()
    			self.fitness_iterations = 0
    			
    		if(self.individual_index == self.ga.population_size):
    			self.state = "PRINT"
    			self.ga.fitness_vector = np.array(self.fitness_vector)
    			self.fitness_vector = []
    			self.individual_index = 0
    			
    	elif(self.state == "PRINT"):
            self.motors.sendV(0)
            self.motors.sendW(0)
            self.ga.generate_statistics()
            self.ga.fraction_save(self.generation)
            self.state = "SELECTION"
    	
    	elif(self.state == "SELECTION"):
    		self.ga.selection()
    		self.state = "CROSSOVER"
    		
    	elif(self.state == "CROSSOVER"):
    		self.ga.crossover()
    		self.state = "MUTATION"
    		
    	elif(self.state == "MUTATION"):
    		self.ga.mutation()
    		self.state = "NEXT"
    		
    	elif(self.state == "NEXT"):
    	    self.ga.generations.append(self.ga.population)
            self.ga.save_handler()

            # Delete the previous one
            # In a sepearate process
            delete_process = multiprocessing.Process(target=self.ga.remove_chromosome,
									             args=('./log/generation' + str(self.ga.current_generation-1),))			
            delete_process.start()

            self.generation = self.generation + 1
            if(self.generation == self.ga.number_of_generations + 1):
	            self.state = "END"
            else:
	            self.individual = self.ga.population[self.individual_index]
	            self.ga.current_generation = self.generation - 1
	            self.state = "FITNESS"
				
        elif(self.state == "END"):
			self.ga.save_statistics('./log/stats')
			self.ga.save_chromosome(self.ga.best_chromosomes, './log/best_chromosomes')
			
			# Print the best fitness and return the chromosome
			print("The best fitness value acheived is: " + str(self.ga.best_fitness))
			print("Found in generation # " + str(self.ga.best_generation))
			
			# Done
			self.stop()
    		
        
