#!/usr/bin/python
#-*- coding: utf-8 -*-
import threading
import time
import sys
import os
import glob
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

time_cycle = 5

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
        
        self.log_folder = './log'
        self.get_latest_file()
        
    def get_latest_file(self):
        files = glob.glob(self.log_folder + '/generation*[0-9].txt')
        files.sort()
        
        try:
            self.latest_generation = int(files[-1][(len(self.log_folder) + 11):-4])
        except IndexError:
            self.latest_generation = 0
    
    def fitness_function(self, index):
    	# The fitness function
    	linear_speed = self.motors[index].getV()
    	rotation_speed = self.motors[index].getW()
    	infrared = np.min(self.getRange(index))
    	
    	# Individual motor speeds are not available in ROS
    	# Just a trick,
    	# Linear speed is proportional to sum of velocities
    	# Angular speed is proportional to difference of velocities
    	
    	fitness = linear_speed * (1 - abs(rotation_speed)) * infrared
    	
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
    	ga.population_size = 30
    	ga.number_of_generations = 100   
    	ga.mutation_probability = 0.01
    	ga.evaluation_steps = 1000
    	ga.number_of_elites = 4
    	ga.fitness_function = self.fitness_function
    	
    	genetic_algorithm = GA(ga, self.log_folder)
    	
    	return genetic_algorithm
    
    def getRange(self, index):
        self.lock.acquire()
        values = self.sensor[index].data.values
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
    	    #self.GA.synchronize()
    	    for index in range(5):
		        output = self.GA.calculate_output({"INFRARED": self.getRange(index)}, index)["MOTORS"]
		        self.motors[index].sendV(10 * (output[0] + output[1]))
		        self.motors[index].sendW(10 * (output[0] - output[1]))
		        #self.GA.synchronize()

            self.GA.fitness_state()
    			
    	elif(self.GA.state == "PRINT"):
    	    for index in range(5):
    	        self.motors[index].sendV(0)
    	        self.motors[index].sendW(0)
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
            output = self.GA.calculate_output({"INFRARED": self.getRange(0)}, 0)["MOTORS"]
            self.motors[0].sendV(10 * (output[0] + output[1]))
            self.motors[0].sendW(10 * (output[0] - output[1]))
    		
        
