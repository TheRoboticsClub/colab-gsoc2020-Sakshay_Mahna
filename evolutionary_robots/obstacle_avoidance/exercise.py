#!/usr/bin/python
# General imports
import sys

# Practice imports
from gui.GUI import MainWindow
from gui.threadGUI import ThreadGUI
from PyQt5.QtWidgets import QApplication

from MyAlgorithm import MyAlgorithm
from interfaces.infrared import ListenerInfrared
from interfaces.motors import PublisherMotors

if __name__ == "__main__":

    infrared = ListenerInfrared("/roombaIR/sensor/infrared")
    motors = PublisherMotors("/roombaIR/cmd_vel", 4, 0.3)
    algorithm = MyAlgorithm(infrared, motors)

    app = QApplication(sys.argv)
    myGUI = MainWindow()
    myGUI.setAlgorithm(algorithm)
    myGUI.show()


    t2 = ThreadGUI(myGUI)
    t2.daemon=True
    t2.start()


    sys.exit(app.exec_())
