import rospy
from sensor_msgs.msg import Range
import threading
from math import pi as PI
import numpy as np

# Number of sensors in the array
ARRAY_LENGTH = 8

class InfraredData:
    """
    Class to provide the infrared sensor data
    from the robot
    
    ...
    Parameters
    ----------
    None
    
    Attributes
    ----------
    values: array_like
        The array of sensor values
        
    Methods
    -------
    None
    """
    def __init__(self):
        """
        Initialization function of the class
        """
        self.values = [0] * ARRAY_LENGTH  # meters
        self.timeStamp = 0 # Time stamp [s] */

    def __str__(self):
        """
        Function to modify the print behaviour
        of the class
        """
        s = "InfraredArray: {\n   arrayLength: " + str(ARRAY_LENGTH)
        s = s + "\n   timeStamp: " + str(self.timeStamp) + "\n values: " + str(self.values) + "\n}"
        return s 

# Function that converts the ROS message
# to our desired class
def message2InfraredData(infra, infrared):
	# To distinguish between the different
	# sensors
	frame_id = infra.header.frame_id
	
	# Generate the index from frame_id
	index = int(frame_id[3:]) - 1
	
	# Modify the user class according to the
	# ROS message
	infrared.values[index] = infra.range
	infrared.timeStamp = infra.header.stamp.secs + (infra.header.stamp.nsecs * 1e-9)
	
	return infrared
	

class ListenerInfrared:
 
    def __init__(self, topic):
        
        self.topic = topic
        self.data = InfraredData()
        self.sub = None
        self.lock = threading.Lock()

        self.start()
 
    def __callback (self, message):

        sensor = message2InfraredData(message, self.data)

        self.lock.acquire()
        self.data = sensor
        self.lock.release()
        
    def stop(self):

        self.sub.unregister()

    def start (self):
 
        self.sub = rospy.Subscriber(self.topic, Range, self.__callback)
        
    def getSensorData(self):

        self.lock.acquire()
        sensor = self.data
        self.lock.release()
        
        return sensor



