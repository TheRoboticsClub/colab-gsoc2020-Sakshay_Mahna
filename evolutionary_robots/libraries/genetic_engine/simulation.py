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
	delta_time(optional): float
		The time with which the simulation proceeds
	
	Attributes
	----------
	delta_time: float
		The time with which the simulation proceeds
		
	bodies: list of Body object
		The list of bodies in our simulated world
		
	number_of_bodies: integer
		Keeps a count of the number of bodies
	
	Methods
	-------
	run()
		This function is supposed to be designed according to
		the exercise
	
	collision_checker()
		Function to check the pairs between which collision has
		occured. Simply calls each and every object's collision
		checker function with another object as parameter.
	
	collision_resolver(collision_list)
		Function to resolve the collisions of the objects
		
	add_body(body)
		Function to append a new body to the world
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
		
	def collision_resolver(self, collision_list):
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
			
		Notes
		-----
		For now, the collision resolution is simple, the objects are only
		placed back at their position
		"""
		
		for tuples in collision_list:
			object1 = tuples[0]
			object2 = tuples[1]
			
			# Assertion check
			assert object1.category == 'mobile' or object2.category == 'mobile', \
				   "One of the two colliding objects should be mobile"
				   
			if(object1.category == 'mobile'):
				# We will reverse all the actions
				object1.velocity = -1 * object1.velocity
				object1.angular_velocity = -1 * object1.angular_velocity
				object1.move_timestep(self.delta_t)
				object1.velocity = -1 * object1.velocity
				object1.angular_velocity = -1 * object1.angular_velocity
			else:
				object2.velocity = -1 * object2.velocity
				object2.angular_velocity = -1 * object2.angular_velocity
				object2.move_timestep(self.delta_t)
				object2.velocity = -1 * object2.velocity
				object2.angular_velocity = -1 * object2.angular_velocity
				
	def add_body(self, body):
		"""
		Function to append a new body to the world
		
		...
		Parameters
		----------
		body: Body Object
			An instance of the Body class
			
		Returns
		-------
		None
		
		Raises
		------
		None
		"""
		self.bodies.append(body)
		self.number_of_bodies += 1
		
		
		
