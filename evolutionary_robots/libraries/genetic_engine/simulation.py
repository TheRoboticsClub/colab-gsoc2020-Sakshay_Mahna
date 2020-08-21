""" Docstring for the simulation.py module

This module contains the simulation aspect of our parallel physics engine
named genetic_engine. The simulation module will be executed in a Thread based
fashion. In this way, multiple threads of simulation can be run, making the
training faster. Also, we are going to have a time step control over the 
simulation, making it deterministic as well.

"""

import threading

class Simulation(threading.Thread):
	"""
	Simulation class. The various exercises can
	subclass this class to create their own environments
	
	...
	Parameters
	----------
	delta_time: float
		The time with which the simulation proceeds
	
	Attributes
	----------
	
	Methods
	-------
	"""
	def __init__(self, delta_time=0.0001):
		"""
		Initialization function of the class
		
		...
		Parameters
		----------
		delta_time: float
			The time with which the simulation proceeds
		
		Returns
		-------
		None
		
		Raises
		------
		None
		"""
		# Constructor of Thread parent class
		threading.Thread.__init__(self)
		
		# Initialize variables
		self.delta_time = delta_time
		self.bodies = []
		self.number_of_bodies = 0
		self.states = []
		
	def run(self):
		"""
		This function is supposed to be designed according to
		the exercise
		"""
		pass
		
	def collision_checker(self):
		"""
		Function to check the pairs between which collision has
		occured. Simply calls each and every object's collision
		checker function with another object as parameter.
		
		...
		Parameters
		----------
		None
		
		Returns
		-------
		collision_pairs: list of tuples
			The tuple pairs which have collided
			
		Raises
		------
		None
		"""
		
		# Collision set
		collison_set = set()
		
		# Check each body with another
		for body_i in range(self.number_of_bodies):
			for body_j in range(body_i, self.number_of_bodies):
				collided = self.bodies[body_i].collision_check(
						   self.bodies[body_j])
						   
				if(collided == True):
					pair_1 = (self.bodies[body_i], self.bodies[body_j])
					pair_2 = (self.bodies[body_j], self.bodies[body_j])
					
					if(pair_1 not in collision_set and\ 
					   pair_2 not in collision_set):
					   collision_set.add(pair_1)
					   
		collision_list = list(collision_set)
		
		return collision_list
		
	def collision_resolution(self, collision_list):
		"""
		Function to resolve the collisions of the objects
		
		...
		Parameters
		----------
		collision_list: list of tuples
			A list containing which objects have collided
			
		Returns
		-------
		None
		
		Raises
		------
		AssertionError
			Atleast one of the two colliding objects should be
			mobile.
		"""
		
		
		
