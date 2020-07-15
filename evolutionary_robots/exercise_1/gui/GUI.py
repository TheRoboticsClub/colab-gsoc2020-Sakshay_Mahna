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
#       Irene Lope Rodriguez<irene.lope236@gmail.com>
#       Vanessa Fernandez Martinez<vanessa_1895@msn.com>

from PyQt5.QtCore import pyqtSignal, Qt, QCoreApplication
from PyQt5.QtWidgets import QMainWindow
from gui.form import Ui_MainWindow
from gui.widgets.logoWidget import LogoWidget
from gui.widgets.graphWidget import GraphWidget
from gui.widgets.plotWidget import PlotWidget

class MainWindow(QMainWindow, Ui_MainWindow):

    updGUI=pyqtSignal()
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.logo = LogoWidget(self)
        self.logoLayout.addWidget(self.logo)
        self.logo.setVisible(True)
        
        self.graph = GraphWidget(self)
        self.graphLayout.addWidget(self.graph)
        self.graph.setVisible(True)
        
        self.plot = PlotWidget(self)
        self.plotLayout.addWidget(self.plot)
        self.plot.setVisible(True)

        self.testButton.clicked.connect(self.testClicked)
        self.trainButton.clicked.connect(self.trainClicked)
        
        self.updGUI.connect(self.updateGUI)

    def updateGUI(self):
        _translate = QCoreApplication.translate
        self.generation.setText(_translate("MainWindow", "CURRENT GENERATION:\t" + str(1)))

    def testClicked(self):
        pass
        
    def trainClicked(self):
    	self.algorithm.play()

    def setAlgorithm(self, algorithm):
        self.algorithm=algorithm

    def getAlgorithm(self):
        return self.algorithm

    def closeEvent(self, event):
        self.algorithm.kill()
        event.accept()
