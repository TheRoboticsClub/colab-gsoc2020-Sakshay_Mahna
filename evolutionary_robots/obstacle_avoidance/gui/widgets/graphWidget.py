import resources_rc
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal, QPointF, Qt, QPoint
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel
import cv2

class GraphWidget(QWidget):
    def __init__(self,winParent):    
        super(GraphWidget, self).__init__()
        self.winParent=winParent
        
        self.graph = cv2.imread("resources/graph.png", cv2.IMREAD_GRAYSCALE)
        self.graph = cv2.resize(self.graph, (691, 331))
        image = QtGui.QImage(self.graph.data, self.graph.shape[1], self.graph.shape[0], self.graph.shape[1], QtGui.QImage.Format_Indexed8)
        self.pixmap = QtGui.QPixmap.fromImage(image)
        
        self.height = self.pixmap.height()
        self.width = self.pixmap.width()
        self.graphWidget = QLabel(self)
        self.graphWidget.setPixmap(self.pixmap)
        


