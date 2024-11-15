from PyQt6.QtWidgets import QWidget, QDockWidget, QApplication, QLayout, QMainWindow, QGridLayout
from PyQt6.QtWidgets import QDialog, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSlot, Qt
import numpy as np
import cv2


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Window")
        grid_layout = QGridLayout()

        docked = QDockWidget("Dockable", self)
        docked.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        dockedWidget = QWidget(self)
        docked.setWidget(dockedWidget)
        dockedWidget.setLayout(QVBoxLayout())
        dockedWidget.layout().addWidget(QPushButton("1"))

        docked_2 = QDockWidget("Dockable_2", self)
        docked_2.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        dockedWidget_2 = QWidget(self)
        docked_2.setWidget(dockedWidget_2)
        dockedWidget_2.setLayout(QVBoxLayout())
        dockedWidget_2.layout().addWidget(QPushButton("2"))

        docked_3 = QDockWidget("Dockable_3", self)
        docked_3.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        dockedWidget_3 = QWidget(self)
        docked_3.setWidget(dockedWidget_3)
        dockedWidget_3.setLayout(QVBoxLayout())
        dockedWidget_3.layout().addWidget(QPushButton("3"))

        docked_4 = QDockWidget("Dockable_4", self)
        docked_4.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        dockedWidget_4 = QWidget(self)
        docked_4.setWidget(dockedWidget_4)
        dockedWidget_4.setLayout(QVBoxLayout())
        dockedWidget_4.layout().addWidget(QPushButton("4"))

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, docked)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, docked_2)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, docked_3)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, docked_4)
        widget = QWidget()
        widget.setLayout(grid_layout)
        self.setCentralWidget(widget)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())