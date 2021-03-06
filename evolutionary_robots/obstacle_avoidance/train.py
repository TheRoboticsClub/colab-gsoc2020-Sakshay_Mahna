#!/usr/bin/python
# General imports
import sys
import rospy

# Practice imports
from gui.GUI import TrainWindow
from gui.threadGUI import ThreadGUI
from PyQt5.QtWidgets import QApplication

from base import MyAlgorithm
from interfaces.infrared import ListenerInfrared
from interfaces.motors import PublisherMotors
from interfaces.clock import ListenerClock

# Execute the application
if __name__ == "__main__":
    rospy.init_node("ObstacleAvoidanceER")

    clock = [ListenerClock("ga1_clock"),
             ListenerClock("ga2_clock"),
             ListenerClock("ga3_clock"),
             ListenerClock("ga4_clock"),
             ListenerClock("ga5_clock")]

    infrared = [ListenerInfrared("ga1/roombaIR/sensor/infrared"),
                ListenerInfrared("ga2/roombaIR/sensor/infrared"),
                ListenerInfrared("ga4/roombaIR/sensor/infrared"),
                ListenerInfrared("ga3/roombaIR/sensor/infrared"),
                ListenerInfrared("ga5/roombaIR/sensor/infrared")]
                
    motors = [PublisherMotors("ga1/roombaIR/cmd_vel", 10, 10, clock[0]),
              PublisherMotors("ga2/roombaIR/cmd_vel", 10, 10, clock[1]),
              PublisherMotors("ga3/roombaIR/cmd_vel", 10, 10, clock[2]),
              PublisherMotors("ga4/roombaIR/cmd_vel", 10, 10, clock[3]),
              PublisherMotors("ga5/roombaIR/cmd_vel", 10, 10, clock[4])]
    
    algorithm = MyAlgorithm(infrared, motors)

    app = QApplication(sys.argv)
    myGUI = TrainWindow()
    myGUI.setAlgorithm(algorithm)
    myGUI.show()


    t2 = ThreadGUI(myGUI)
    t2.daemon=True
    t2.start()


    sys.exit(app.exec_())
