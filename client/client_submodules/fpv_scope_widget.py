from PySide6.QtWidgets import (QWidget, QListWidget, QApplication, QSpacerItem, QSizePolicy,
                             QStackedWidget, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QTabWidget, QComboBox, QGroupBox, QSpinBox, QDockWidget)
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPainter, QPen
from PySide6.QtCore import Qt, QRect
import qdarktheme
from os import walk
import yaml


class FPVScopeWidget(QDockWidget):
    def __init__(self, theme_type: str):
        super().__init__()
        self.theme_type = theme_type
