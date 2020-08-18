#!/usr/bin/python
# General imports
import sys
import rospy

# Practice imports
from gui.GUI import TestWindow
from gui.threadGUI import ThreadGUI
from PyQt5.QtWidgets import QApplication

from base import MyAlgorithm
from interfaces.infrared import ListenerInfrared
from interfaces.motors import PublisherMotors
from interfaces.clock import ListenerClock

if __name__ == "__main__":
    rospy.init_node("ObstacleAvoidanceER")

    clock = [ListenerClock("ga1_clock")]

    infrared = [ListenerInfrared("ga1/roombaIR/sensor/infrared")]
                
    motors = [PublisherMotors("ga1/roombaIR/cmd_vel", 10, 10, clock[0])]
    
    algorithm = MyAlgorithm(infrared, motors)

    app = QApplication(sys.argv)
    myGUI = TestWindow()
    myGUI.setAlgorithm(algorithm)
    myGUI.show()


    t2 = ThreadGUI(myGUI)
    t2.daemon=True
    t2.start()


    sys.exit(app.exec_())
