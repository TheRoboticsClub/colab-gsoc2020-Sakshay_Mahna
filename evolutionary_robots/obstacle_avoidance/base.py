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

import MyAlgorithm as algorithm

sys.path.append('./../libraries')
from genetic_algorithm.ga_simulation import GeneticAlgorithmGazebo
from GA import GA

time_cycle = 20

# The basic machinery of the Robotics Academy Template
# One special thing in this: the user is not going to make changes in
# this part of the code. A state machine has already been implemented in 
# algorithm function whose parameters are to be coded by students in 
# MyALgorithm.py file
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
        
        self.log_folder = algorithm.LOG_FOLDER
        self.get_latest_file()
        self.define_neural_network = algorithm.define_neural_network
        
        self.WHEEL_RADIUS = self.motors[0].WHEEL_RADIUS
        self.WHEEL_DISTANCE = self.motors[0].WHEEL_DISTANCE
        
    def get_latest_file(self):
        files = glob.glob(self.log_folder + '/generation*[0-9].txt')
        files = sorted(files, key = lambda x: (len(x), x))
        
        try:
            self.latest_generation = int(files[-1][(len(self.log_folder) + 11):-4])
        except IndexError:
            self.latest_generation = 0
    
    def fitness_function(self, index):
    	# The fitness function
    	left_motor_speed = self.motors[index].left_motor_speed
    	right_motor_speed = self.motors[index].right_motor_speed
    	infrared = self.getRange(index)
    	
    	fitness = algorithm.fitness_function(left_motor_speed, right_motor_speed, infrared)
    	
    	if(fitness < 0):
    	    fitness = 0
    	
    	return fitness 
    	
    def define_genetic_algorithm(self):
    	# Define the Genetic Algorithm
    	neural_network = self.define_neural_network()
    	ga = GeneticAlgorithmGazebo(neural_network)
    
    	# Define the genetic algorithm
    	ga.population_size = algorithm.POPULATION_SIZE
    	ga.number_of_generations = algorithm.NUMBER_OF_GENERATIONS   
    	ga.mutation_probability = algorithm.MUTATION_PROBABILITY
    	ga.evaluation_steps = algorithm.EVALUATION_STEPS
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
    	        infrared = self.getRange(index)
    	        output = self.GA.calculate_output({"INFRARED": infrared}, index)["MOTORS"]
                output = output / 2
                if(max(infrared) > 0.80):
                    self.motors[index].sendV(0)
                else:
                    self.motors[index].sendV(4 * (output[0] + output[1]))
                self.motors[index].sendW(4 * (output[0] - output[1]))
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
			self.GA.save_state()
			
			# Done
			self.stop()
			
        elif(self.GA.state == "TEST"):
            output = self.GA.calculate_output({"INFRARED": self.getRange(0)}, 0)["MOTORS"]
            output = output / 2
            self.motors[0].sendV(4 * (output[0] + output[1]))
            self.motors[0].sendW(4 * (output[0] - output[1]))
    		
        
