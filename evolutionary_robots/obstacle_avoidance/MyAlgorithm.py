#!/usr/bin/python
#-*- coding: utf-8 -*-
import threading
import time
import sys
import os
from datetime import datetime

import math
import cv2
import numpy as np

sys.path.append('./../libraries')
from neural_networks.ann import ArtificialNeuralNetwork
from neural_networks.interface import Layer
from neural_networks.activation_functions import *
from genetic_algorithm.ga_simulation import GeneticAlgorithmGazebo
from GA import GA

time_cycle = 80

class MyAlgorithm(threading.Thread):
    def __init__(self, sensor, motors):
        # Initializing the Algorithm object
        self.sensor = sensor
        self.motors = motors
        self.start_state = True
        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        self.threshold_sensor_lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)
    
    def fitness_function(self, chromosome):
    	# The fitness function
    	linear_speed = self.motors.getV()
    	rotation_speed = self.motors.getW()
    	infrared = np.min(self.getRange())
    	
    	# Individual motor speeds are not available in ROS
    	# Just a trick,
    	# Linear speed is proportional to sum of velocities
    	# Angular speed is proportional to difference of velocities
    	
    	fitness = linear_speed * (1 - math.sqrt(abs(rotation_speed))) * infrared
    	
    	return fitness 
    	
    
    def define_neural_network(self):
    	# Define the layers
    	# Layer(name_of_layer, number_of_neurons, activation_function, sensor_inputs, list_of_output_layer_names)
    	inputLayer = Layer("inputLayer", 8, IdentityActivation(), "INFRARED", ["outputLayer"])
    	outputLayer = Layer("outputLayer", 2, SigmoidActivation(), "", ["MOTORS"])
    	# Define the Neural Network
    	neural_network = ArtificialNeuralNetwork([inputLayer, outputLayer], "STATIC")
    	
    	return neural_network
    	
    def define_genetic_algorithm(self):
    	# Define the Genetic Algorithm
    	neural_network = self.define_neural_network()
    	ga = GeneticAlgorithmGazebo(neural_network)
    
    	# Define the genetic algorithm
    	log_folder = './log'
    	ga.population_size = 10
    	ga.number_of_generations = 100   
    	ga.mutation_probability = 0.01
    	ga.evaluation_steps = 100
    	ga.number_of_elites = 0
    	ga.fitness_function = self.fitness_function
    	
    	genetic_algorithm = GA(ga, log_folder)
    	
    	return genetic_algorithm
    
    def getRange(self):
        self.lock.acquire()
        values = self.sensor.data.values
        self.lock.release()
        return values

    def run (self):
    	if(self.start_state == True):
			self.GA = self.define_genetic_algorithm()
			self.start_state = False

        self.GA.run_state = self.run_state
        self.GA.initialize()
    		
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
    	self.motors.sendV(0)
    	self.motors.sendW(0)
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()

    def algorithm(self):
        if(self.GA.state == "SAVE"):
            self.GA.save_state()
              
    	elif(self.GA.state == "FITNESS"):
    		output = self.GA.calculate_output({"INFRARED": self.getRange()})["MOTORS"]
    		self.motors.sendV(4 * (output[0] + output[1]))
    		self.motors.sendW((output[0] - output[1]))
    		
    		self.GA.fitness_state()
    			
    	elif(self.GA.state == "PRINT"):
            self.motors.sendV(0)
            self.motors.sendW(0)
            self.GA.print_state()
    	
    	elif(self.GA.state == "SELECTION"):
    		self.GA.selection_state()
    		
    	elif(self.GA.state == "CROSSOVER"):
    		self.GA.crossover_state()
    		
    	elif(self.GA.state == "MUTATION"):
    		self.GA.mutation_state()
    		
    	elif(self.GA.state == "NEXT"):
    	    self.GA.next_state()
				
        elif(self.GA.state == "END"):
			self.GA.end_state()
			
			# Done
			self.stop()
			
        elif(self.GA.state == "TEST"):
            output = self.GA.calculate_output({"INFRARED": self.getRange()})["MOTORS"]
            self.motors.sendV(4 * (output[0] + output[1]))
            self.motors.sendW((output[0] - output[1]))
    		
        
