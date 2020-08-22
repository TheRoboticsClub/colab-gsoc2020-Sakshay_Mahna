""" Docstring for Body module

This module contains the implementation of a super class Body. All the bodies
in our simulation are instances of this super class

"""

import math

class Body(object):
	"""
	The Body class
	
	...
	Parameters
	----------
	
	Attributes
	----------
	
	Methods
	-------
	
	"""
	
	def init(self, position, orientation, category, shape, dimensions):
		"""
		Initialization function of Body
		
		...
		Parameters
		----------
		position: 2 element list of floats
			Represents the initial position of the body
			
		orientation: float
			Represents the angle from 12 o'clock (in radians)
			
		category: 'mobile' or 'immobile'
			The category of the body
			
		shape: 'straight' or 'arc'
			The type of shape of the body
			
		dimensions: float or list of float depending upon shape
			The dimensions of the shape of the body.
			For rectangle, [length, breadth]
			For circle, radius
			
		Returns
		-------
		None
		
		Raises
		------
		None
		"""
		
		# Add the class variables
		self.initial_position = position
		self.initial_orientation = orientation
		self.position = position
		self.orientation = orientation
		self.category = category
		self.shape = shape
		self.dimensions = dimensions
		
	def collision_check(self, body):
		"""
		Performs a collision check of the body with another body
		Every body has it's own collision check
		
		...
		Parameters
		----------
		body: Instance of body class
			Body with which to check collision
			
		Returns
		-------
		collision: bool
			If the objects have collided
			
		Raises
		------
		None
		"""
		pass
		
	def reset_pose(self):
		"""
		Function to reset the pose of the body
		"""
		
		self.position = self.initial_position
		self.orientation = self.initial_orientation
		
	def set_equation(self):
		"""
		Set the equation of the body shape.
		
		Working:
		If the body is an arc: [radius, c_x, c_y]
		If the body is a rectangle: [[a1, b1, c1, x1_constraint, y1_constraint],
									 [a2, b2, c2, x2_constraint, y2_constraint],
									 [a3, b3, c3, x3_constraint, y3_constraint],
									 [a4, b4, c4, x4_constraint, y4_constraint]]
		"""
		if(self.shape == 'arc'):
			self.equation = [self.dimensions, self.position[0], self.position[1]]
		
		elif(self.shape == "straight"):
			a1 = math.cosine(self.orientation)
			b1 = -1 * math.sine(self.orientation)
			c1 = -1 * (self.position[1] + self.dimensions[1] / 2)
			
			a2 = math.sine(self.orientation)
			b2 = math.cosine(self.orientation)
			c2 = -1 * (self.position[0] + self.dimensions[0] / 2)
			
			a3 = math.cosine(self.orientation)
			b3 = -1 * math.sine(self.orientation)
			c3 = -1 * (self.position[1] - self.dimensions[1] / 2)
			
			a4 = math.sine(self.orientation)
			b4 = math.cosine(self.orientation)
			c4 = -1 * (self.position[0] - self.dimensions[0] / 2)
			
			x1_constraint = [math.cosine(self.orientation) * (-1) * (c4) - \
							 math.sine(self.orientation) * (-1) * (c1),
							 math.cosine(self.orientation) * (-1) * (c2) - \
							 math.sine(self.orientation) * (-1) * (c1)]
							 
			x2_constraint = [x1_constraint[1],
							 math.cosine(self.orientation) * (-1) * (c2) - \
							 math.sine(self.orientation) * (-1) * (c3)]
							 
			x3_constraint = [x2_constraint[1],
							 math.cosine(self.orientation) * (-1) * (c4) - \
							 math.sine(self.orientation) * (-1) * (c3)]
							 
			x4_constraint = [x3_constraint[1],
							 x1_constraint[0]]
							 
			y1_constraint = [math.sine(self.orientation) * (-1) * (c4) + \
							 math.cosine(self.orientation) * (-1) * (c1),
							 math.sine(self.orientation) * (-1) * (c2) + \
							 math.cosine(self.orientation) * (-1) * (c1)]
							 
			y2_constraint = [y1_constraint[1],
							 math.sine(self.orientation) * (-1) * (c2) + \
							 math.cosine(self.orientation) * (-1) * (c3)]
							 
			y3_constraint = [y2_constraint[1],
							 math.sine(self.orientation) * (-1) * (c4) + \
							 math.cosine(self.orientation) * (-1) * (c3)]
							 
			y4_constraint = [y3_constraint[1],
							 y1_constraint[0]]
							 
			self.equation = [[a1, b1, c1, x1_constraint, y1_constraint],
							 [a2, b2, c2, x2_constraint, y2_constraint],
							 [a3, b3, c3, x3_constraint, y3_constraint],
							 [a4, b4, c4, x4_constraint, y4_constraint]]
							 
							 
			
