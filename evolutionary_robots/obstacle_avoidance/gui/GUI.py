from PyQt5.QtCore import pyqtSignal, Qt, QCoreApplication
from PyQt5.QtWidgets import QMainWindow
from gui.form import Ui_TrainWindow, Ui_TestWindow
from gui.widgets.logoWidget import LogoWidget

# Test Window
class TestWindow(QMainWindow, Ui_TestWindow):

    updGUI=pyqtSignal()
    def __init__(self, parent=None):
        super(TestWindow, self).__init__(parent)
        self.setupUi(self)
        self.logo = LogoWidget(self)
        self.logoLayout.addWidget(self.logo)
        self.logo.setVisible(True)
        self.display_stats = False
        
        self.clickedButton = False
        
        # Attach event handler to bestButton
        self.bestButton.clicked.connect(self.bestClicked)
        
    def updateGUI(self):
        pass
     
    # Event handler function of bestButton       
    def bestClicked(self):
        # Pass the generation number as a string
        generation = int(self.input_generation_2.value())
        self.algorithm.run_state = "TEST" + str(generation)
        self.display_stats = True
        if(self.clickedButton == False):
        	self.algorithm.play()
        	self.clickedButton = True
        else:
        	self.algorithm.GA.initialize()
    	
    def update_plot(self):
    	self.plot.update_image()

    def setAlgorithm(self, algorithm):
        self.algorithm=algorithm
        _translate = QCoreApplication.translate
        self.input_generation_2.setMaximum(self.algorithm.latest_generation - 1)
        self.out_of_generation_2.setText(_translate("MainWindow", " / " + str(self.algorithm.latest_generation - 1)))

    def getAlgorithm(self):
        return self.algorithm

    def closeEvent(self, event):
        self.algorithm.kill()
        event.accept()
   
# Train window     
class TrainWindow(QMainWindow, Ui_TrainWindow):

    updGUI=pyqtSignal()
    def __init__(self, parent=None):
        super(TrainWindow, self).__init__(parent)
        self.setupUi(self)
        self.logo = LogoWidget(self)
        self.logoLayout.addWidget(self.logo)
        self.logo.setVisible(True)
        self.display_stats = False

        # Attach event handler to trainButton and generationButton
        self.trainButton.clicked.connect(self.trainClicked)
        self.trainButton.setCheckable(True)
        self.generationButton.clicked.connect(self.generationClicked)
        self.generationButton.setCheckable(True)

        self.updGUI.connect(self.updateGUI)

    def updateGUI(self):
    	if(self.display_stats == True and self.algorithm.start_state == False):
        	self.update_stats()
        
    # Event handler function of trainButton
    def trainClicked(self):
        self.display_stats = True
        self.algorithm.run_state = "TRAIN"
        if self.trainButton.isChecked():
            self.trainButton.setText('Stop Training')
            self.algorithm.play()
        else:
            self.trainButton.setText('Start Training')
            self.algorithm.stop()
    
    # Event Handler function of TestButton       
    def generationClicked(self):
        generation = int(self.input_generation.value())
        self.algorithm.run_state = "CONTINUE" + str(generation)
        self.display_stats = True
        if self.generationButton.isChecked():
            self.generationButton.setText('Stop Training')
            self.algorithm.play()
        else:
	        self.generationButton.setText('Continue Training')
	        self.algorithm.stop()
    
    # Function to update the statistics       
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
        self.out_of_generation.setText(_translate("MainWindow", " / " + str(self.algorithm.latest_generation)))
        self.last_generation.setText(_translate("MainWindow", str(self.algorithm.latest_generation)))

    def getAlgorithm(self):
        return self.algorithm

    def closeEvent(self, event):
        self.algorithm.kill()
        event.accept()
