#!/usr/bin/python
# General imports
import sys
import rospy

# Practice imports
from gui.GUI import MainWindow
from gui.threadGUI import ThreadGUI
from PyQt5.QtWidgets import QApplication

from MyAlgorithm import MyAlgorithm
from interfaces.infrared import ListenerInfrared
from interfaces.motors import PublisherMotors

if __name__ == "__main__":
    rospy.init_node("ObstacleAvoidanceER")

    infrared = [ListenerInfrared("ga1/roombaIR/sensor/infrared"),
                ListenerInfrared("ga2/roombaIR/sensor/infrared"),
                ListenerInfrared("ga4/roombaIR/sensor/infrared"),
                ListenerInfrared("ga3/roombaIR/sensor/infrared"),
                ListenerInfrared("ga5/roombaIR/sensor/infrared")]
    motors = [PublisherMotors("ga1/roombaIR/cmd_vel", 10, 5),
              PublisherMotors("ga2/roombaIR/cmd_vel", 10, 5),
              PublisherMotors("ga3/roombaIR/cmd_vel", 10, 5),
              PublisherMotors("ga4/roombaIR/cmd_vel", 10, 5),
              PublisherMotors("ga5/roombaIR/cmd_vel", 10, 5)]
    algorithm = MyAlgorithm(infrared, motors)

    app = QApplication(sys.argv)
    myGUI = MainWindow()
    myGUI.setAlgorithm(algorithm)
    myGUI.show()


    t2 = ThreadGUI(myGUI)
    t2.daemon=True
    t2.start()


    sys.exit(app.exec_())
