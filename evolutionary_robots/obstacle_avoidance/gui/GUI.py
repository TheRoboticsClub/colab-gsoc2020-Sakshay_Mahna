from PyQt5.QtCore import pyqtSignal, Qt, QCoreApplication
from PyQt5.QtWidgets import QMainWindow
from gui.form import Ui_MainWindow
from gui.widgets.logoWidget import LogoWidget

class MainWindow(QMainWindow, Ui_MainWindow):

    updGUI=pyqtSignal()
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.logo = LogoWidget(self)
        self.logoLayout.addWidget(self.logo)
        self.logo.setVisible(True)
        self.display_stats = False
        
        self.clickedButton = False

        self.trainButton.clicked.connect(self.trainClicked)
        self.bestButton.clicked.connect(self.bestClicked)
        self.generationButton.clicked.connect(self.generationClicked)

        self.updGUI.connect(self.updateGUI)

    def updateGUI(self):
    	if(self.display_stats == True and self.algorithm.start_state == False):
        	self.update_stats()
        
    def trainClicked(self):
        self.display_stats = True
        self.algorithm.run_state = "TRAIN"
        if(self.clickedButton == False):
        	self.algorithm.play()
        	self.clickedButton = True
        else:
        	self.algorithm.GA.initialize()
            
    def bestClicked(self):
        generation = int(self.input_generation_2.value())
        self.algorithm.run_state = "TEST" + str(generation)
        self.display_stats = True
        if(self.clickedButton == False):
        	self.algorithm.play()
        	self.clickedButton = True
        else:
        	self.algorithm.GA.initialize()
            
    def generationClicked(self):
        generation = int(self.input_generation.value())
        self.algorithm.run_state = "CONTINUE" + str(generation)
        self.display_stats = True
        if(self.clickedButton == False):
        	self.algorithm.play()
        	self.clickedButton = True
        else:
        	self.algorithm.GA.initialize()
            
    def update_stats(self):
        stats_array = self.algorithm.GA.return_stats()
        
        if(len(stats_array) == 3):
        	_translate = QCoreApplication.translate
        	self.generation_value.setText(_translate("MainWindow", str(stats_array[0])))
        	self.individual_value.setText(_translate("MainWindow", str(stats_array[1])))
        	self.best_fitness_value.setText(_translate("MainWindow", str(stats_array[2])))
    	
    def update_plot(self):
    	self.plot.update_image()

    def setAlgorithm(self, algorithm):
        self.algorithm=algorithm
        _translate = QCoreApplication.translate
        self.input_generation.setMaximum(self.algorithm.latest_generation)
        self.input_generation_2.setMaximum(self.algorithm.latest_generation - 1)
        self.out_of_generation.setText(_translate("MainWindow", " / " + str(self.algorithm.latest_generation)))
        self.out_of_generation_2.setText(_translate("MainWindow", " / " + str(self.algorithm.latest_generation - 1)))
        self.last_generation.setText(_translate("MainWindow", str(self.algorithm.latest_generation)))

    def getAlgorithm(self):
        return self.algorithm

    def closeEvent(self, event):
        self.algorithm.kill()
        event.accept()
