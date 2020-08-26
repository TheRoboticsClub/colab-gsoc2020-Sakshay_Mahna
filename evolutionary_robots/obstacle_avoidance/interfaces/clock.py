import rospy
from rosgraph_msgs.msg import Clock
import threading
from math import pi as PI
import numpy as np

# Function that converts the ROS message
# to our desired class
def message2TimeData(message):
	# Extract secs and nsecs
	secs = float(message.clock.secs)
	nsecs = float(message.clock.nsecs)
	
	# Convert to milliseconds
	msecs = secs * 1000 + nsecs * 1e-6
	
	return msecs
	
# This class subscribes to the clock topic and
# interprets the milliseconds passed since the
# execution of the application
class ListenerClock:
 
    def __init__(self, topic):
        
        self.topic = topic
        self.msecs = 0
        self.sub = None
        self.lock = threading.Lock()

        self.start()
 
    def __callback (self, message):
        msecs = message2TimeData(message)

        self.lock.acquire()
        self.msecs = msecs
        self.lock.release()
        
    def stop(self):

        self.sub.unregister()

    def start (self):
        self.sub = rospy.Subscriber(self.topic, Clock, self.__callback)
        
    def getTimeData(self):

        self.lock.acquire()
        msecs = self.msecs
        self.lock.release()
        
        return msecs




