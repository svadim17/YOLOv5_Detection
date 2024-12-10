from PyQt6.QtWidgets import (QWidget, QDockWidget, QApplication, QSpinBox, QDoubleSpinBox,
                             QAbstractSpinBox, QDialog, QCheckBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QSpacerItem, QSizePolicy, QGroupBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSlot, Qt, pyqtSignal
import numpy as np
import cv2
import qdarktheme
import time


class ProcessOptions(QWidget):
    signal_process_name = pyqtSignal(str)
    signal_restart_process_name = pyqtSignal(str)

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle(name)
        self.setFixedSize(350, 120)

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.main_layout.setSpacing(20)
        self.setLayout(self.main_layout)

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.btn_get_process_status = QPushButton('Get process status')
        self.btn_get_process_status.setMinimumWidth(130)
        self.btn_get_process_status.clicked.connect(self.send_process_name)

        self.l_process_status = QLabel('Status:')
        self.l_process_status.setStyleSheet("font-size: 14px")

        self.btn_restart_process = QPushButton('Restart process')
        self.btn_restart_process.setMinimumWidth(130)
        self.btn_restart_process.clicked.connect(self.send_restart_process_name)

    def add_widgets_to_layout(self):
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.btn_get_process_status)
        status_layout.addWidget(self.l_process_status)

        self.main_layout.addLayout(status_layout)
        self.main_layout.addWidget(self.btn_restart_process, alignment=Qt.AlignmentFlag.AlignLeft)

    def send_process_name(self):
        self.l_process_status.setText('Status: ')
        self.signal_process_name.emit(self.name)

    def send_restart_process_name(self):
        self.l_process_status.setText('Status:')
        self.signal_restart_process_name.emit(self.name)

    def update_process_status(self, status: bool):
        self.l_process_status.setText(f'Status: {str(status)}  |  {time.strftime("%H:%M:%S", time.localtime())}')


if __name__ == '__main__':
    app = QApplication([])
    qdarktheme.setup_theme()
    window = ProcessOptions(name='test')
    window.show()
    app.exec()
