from PyQt6.QtWidgets import QWidget, QDockWidget, QApplication, QLayout
from PyQt6.QtWidgets import QDialog, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSlot, Qt
import numpy as np
import cv2


class PopOutMenu(QDockWidget, QWidget):
    def __init__(self, enabled_channels: list):
        super().__init__()
        self.enabled_channels = enabled_channels
        self.setWidget(QWidget(self))
        self.main_layout = QVBoxLayout()
        self.widget().setLayout(self.main_layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Позволяет растягиваться

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.menu_label = QLabel('Recognition Settings')




    def add_widgets_to_layout(self):
        self.main_layout.addWidget(self.menu_label)


if __name__ == '__main__':
    app = QApplication([])
    window = PopOutMenu()
    window.show()
    app.exec()