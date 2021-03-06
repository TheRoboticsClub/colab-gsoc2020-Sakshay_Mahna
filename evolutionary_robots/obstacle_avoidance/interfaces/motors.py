#
#  Copyright (C) 1997-2016 JDE Developers Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see http://www.gnu.org/licenses/.
#  Authors :
#       Alberto Martin Florido <almartinflorido@gmail.com>
#       Aitor Martinez Fernandez <aitor.martinez.fernandez@gmail.com>
#

import rospy
from geometry_msgs.msg import Twist
import threading
from math import pi as PI
from .threadPublisher import ThreadPublisher



def cmdvel2Twist(vel):

    tw = Twist()
    tw.linear.x = vel.vx
    tw.linear.y = vel.vy
    tw.linear.z = vel.vz
    tw.angular.x = vel.ax
    tw.angular.y = vel.ay
    tw.angular.z = vel.az

    return tw


class CMDVel ():

    def __init__(self):

        self.vx = 0 # vel in x[m/s] (use this for V in wheeled robots)
        self.vy = 0 # vel in y[m/s]
        self.vz = 0 # vel in z[m/s]
        self.ax = 0 # angular vel in X axis [rad/s]
        self.ay = 0 # angular vel in X axis [rad/s]
        self.az = 0 # angular vel in Z axis [rad/s] (use this for W in wheeled robots)
        self.timeStamp = 0 # Time stamp [s]


    def __str__(self):
        s = "CMDVel: {\n   vx: " + str(self.vx) + "\n   vy: " + str(self.vy)
        s = s + "\n   vz: " + str(self.vz) + "\n   ax: " + str(self.ax) 
        s = s + "\n   ay: " + str(self.ay) + "\n   az: " + str(self.az)
        s = s + "\n   timeStamp: " + str(self.timeStamp)  + "\n}"
        return s 

class PublisherMotors(object):
 
    def __init__(self, topic, maxV, maxW, clock):

        self.maxW = maxW
        self.maxV = maxV
        self.WHEEL_DISTANCE = 0.28
        self.WHEEL_RADIUS = 0.033

        self.topic = topic
        self.data = CMDVel()
        self.pub = rospy.Publisher(self.topic, Twist, queue_size=1)
        self.lock = threading.Lock()

        self.kill_event = threading.Event()
        self.thread = ThreadPublisher(self, self.kill_event, clock)

        self.thread.daemon = True
        self.start()
 
    def publish (self):

        self.lock.acquire()
        tw = cmdvel2Twist(self.data)
        self.lock.release()
        self.pub.publish(tw)
        
    def stop(self):
   
        self.pub.unregister()
        self.kill_event.set()

    def start (self):

        self.kill_event.clear()
        self.thread.start()

    def getMaxW(self):
        return self.maxW

    def getMaxV(self):
        return self.maxV
        
    def getV(self):
        self.lock.acquire()
        data = self.data.vx
        self.lock.release()
        
        return data
        
    def getW(self):
        self.lock.acquire()
        data = self.data.az
        self.lock.release()
        
        return data

    def sendVelocities(self, vel):

        self.lock.acquire()
        self.data = vel
        self.lock.release()

    def sendV(self, v):

        self.sendVX(v)

    def sendL(self, l):

        self.sendVY(l)

    def sendW(self, w):

        self.sendAZ(w)

    def sendVX(self, vx):

        self.lock.acquire()
        self.data.vx = vx
        self.lock.release()

    def sendVY(self, vy):

        self.lock.acquire()
        self.data.vy = vy
        self.lock.release()

    def sendAZ(self, az):
        self.lock.acquire()
        self.data.az = az
        self.lock.release()
        
    @property
    def left_motor_speed(self):
    	v = self.getV() / 4
    	w = self.getW() / 4
    	
    	left_motor_speed = (v - w) / 2 
    	return left_motor_speed
    	
    @property
    def right_motor_speed(self):
    	v = self.getV() / 4
    	w = self.getW() / 4
    	
    	right_motor_speed = (v + w) / 2
    	return right_motor_speed


