from PyQt6.QtWidgets import (QWidget, QListWidget, QApplication, QSpacerItem, QSizePolicy,
                             QStackedWidget, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QTabWidget, QComboBox, QGroupBox, QMainWindow)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtCore import pyqtSlot, Qt, pyqtSignal
import qdarktheme
from os import walk


class MainTab(QMainWindow):

    def __init__(self):
        super().__init__()
        self.create_widgets()

    def create_widgets(self):
        self.chb_show_frequencies = QCheckBox('Show frequencies')
        self.chb_show_frequencies.setChecked(True)
        state = self.chb_show_frequencies.isChecked()
        print(state)


if __name__ == '__main__':
    a = MainTab()