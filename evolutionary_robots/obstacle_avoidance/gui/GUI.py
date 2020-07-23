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

        self.trainButton.clicked.connect(self.trainClicked)
        self.trainButton.setCheckable(True)
		
        self.continueButton.clicked.connect(self.continueClicked)
        self.continueButton.setCheckable(True)

        self.bestButton.clicked.connect(self.bestClicked)
        self.bestButton.setCheckable(True)

        self.generationButton.clicked.connect(self.generationClicked)
        self.generationButton.setCheckable(True)

        self.updGUI.connect(self.updateGUI)

    def updateGUI(self):
    	if(self.display_stats == True and self.algorithm.start_state == False):
        	self.update_stats()
        
    def trainClicked(self):
        if self.trainButton.isChecked():
            self.trainButton.setText('Stop Training')
            self.trainButton.setStyleSheet("background-color: #7dcea0")
            self.algorithm.run_state = "TRAIN"
            self.algorithm.play()
            self.display_stats = True
        else:
            self.trainButton.setText('Start Training')
            self.trainButton.setStyleSheet("background-color: #ec7063")
            self.algorithm.stop()
            self.display_stats = False
            
    def continueClicked(self):
    	if self.continueButton.isChecked():
            self.continueButton.setText('Stop Training')
            self.continueButton.setStyleSheet("background-color: #7dcea0")
            self.algorithm.run_state="CONTINUE"
            self.algorithm.play()
            self.display_stats = True
    	else:
            self.continueButton.setText('Continue Training')
            self.continueButton.setStyleSheet("background-color: #ec7063")
            self.algorithm.stop()
            self.display_stats = False
            
    def bestClicked(self):
    	if self.bestButton.isChecked():
            self.bestButton.setText('Stop Testing')
            self.bestButton.setStyleSheet("background-color: #7dcea0")
            self.algorithm.run_state = "TEST"
            self.algorithm.play()
            self.display_stats = True
    	else:
            self.bestButton.setText('Test Best Generation')
            self.bestButton.setStyleSheet("background-color: #ec7063")
            self.algorithm.stop()
            self.display_stats = False
            
    def generationClicked(self):
    	if self.generationButton.isChecked():
            self.generationButton.setText('Stop Testing')
            self.generationButton.setStyleSheet("background-color: #7dcea0")
            if(self.button_0.isChecked()):
            	self.algorithm.run_state = "RESUME_0"
            elif(self.button_25.isChecked()):
            	self.algorithm.run_state = "RESUME_25"
            elif(self.button_50.isChecked()):
            	self.algorithm.run_state = "RESUME_50"
            elif(self.button_75.isChecked()):
            	self.algorithm.run_state = "RESUME_75"
            elif(self.button_100.isChecked()):
            	self.algorithm.run_state = "RESUME_100"
            else:
            	self.algorithm.run_state = "RESUME_0"
            self.algorithm.play()
            self.display_stats = True
    	else:
            self.generationButton.setText('Test Generation')
            self.generationButton.setStyleSheet("background-color: #ec7063")
            self.algorithm.stop()
            self.display_stats = False
            
    def update_stats(self):
        stats_array = self.algorithm.GA.return_stats()
        
        if(len(stats_array) == 5):
        	_translate = QCoreApplication.translate
        	self.generation_value.setText(_translate("MainWindow", str(stats_array[0])))
        	self.individual_value.setText(_translate("MainWindow", str(stats_array[1])))
        	self.fitness_value.setText(_translate("MainWindow", str(stats_array[2])))
        	self.timer_value.setText(_translate("MainWindow", str(stats_array[3])))
        	self.best_fitness_value.setText(_translate("MainWindow", str(stats_array[4])))
    	
    def update_plot(self):
    	self.plot.update_image()

    def setAlgorithm(self, algorithm):
        self.algorithm=algorithm

    def getAlgorithm(self):
        return self.algorithm

    def closeEvent(self, event):
        self.algorithm.kill()
        event.accept()
