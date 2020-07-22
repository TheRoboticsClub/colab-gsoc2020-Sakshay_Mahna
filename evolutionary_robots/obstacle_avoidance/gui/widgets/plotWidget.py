import resources_rc
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal, QPointF, Qt, QPoint
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel
import cv2

class PlotWidget(QWidget):
    def __init__(self,winParent):    
        super(PlotWidget, self).__init__()
        self.winParent=winParent
        
    def show_plot(self):
        self.plot = cv2.imread("./log/fitness_plot.png", cv2.IMREAD_COLOR)
        if(self.plot is not None):
            self.plot = cv2.resize(self.plot, (400, 391))
            self.plot = cv2.cvtColor(self.plot, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(self.plot.data, self.plot.shape[1], self.plot.shape[0], QtGui.QImage.Format_RGB888)
            self.pixmap = QtGui.QPixmap.fromImage(image)
            
            self.height = self.pixmap.height()
            self.width = self.pixmap.width()
            self.plotWidget = QLabel(self)
            self.plotWidget.setPixmap(self.pixmap)
        
        


