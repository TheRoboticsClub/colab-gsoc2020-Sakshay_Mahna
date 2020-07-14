#
#  Copyright (C) 1997-2015 JDE Developers Team
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
#       Alberto Martin Florido <almartinflorido@gmail.com>
#
import resources_rc
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal, QPointF, Qt, QPoint
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel
import cv2

class PlotWidget(QWidget):
    def __init__(self,winParent):    
        super(PlotWidget, self).__init__()
        self.winParent=winParent
        
        self.plot = cv2.imread("resources/fitness_plot.png", cv2.IMREAD_COLOR)
        self.plot = cv2.resize(self.plot, (500, 230))
        self.plot = cv2.cvtColor(self.plot, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(self.plot.data, self.plot.shape[1], self.plot.shape[0], QtGui.QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(image)
        
        self.height = self.pixmap.height()
        self.width = self.pixmap.width()
        self.plotWidget = QLabel(self)
        self.plotWidget.setPixmap(self.pixmap)
        


