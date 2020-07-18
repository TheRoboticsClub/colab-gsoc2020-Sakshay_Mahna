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
