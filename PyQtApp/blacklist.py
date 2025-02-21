import loguru
from PyQt6.QtWidgets import (QWidget, QListWidget, QApplication, QSpacerItem, QSizePolicy,
                             QStackedWidget, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QTabWidget, QComboBox, QGroupBox)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtCore import pyqtSlot, Qt, pyqtSignal
import qdarktheme
from os import walk


class SaveTab(QWidget):

    def __init__(self, enabled_channels: list, map_list: list):
        super().__init__()
        self.enabled_channels = enabled_channels
        self.map_list = map_list

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.channels_stacks = {}

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.chb_save_detected = QCheckBox('Save detected')

        self.box_channels = QGroupBox('Channels')

        self.channels_list = QListWidget()
        self.channels_list.setFixedHeight(90)
        self.channels_list.setMinimumWidth(80)
        self.stack = QStackedWidget()

        i = 0
        for channel in self.enabled_channels:
            self.channels_list.insertItem(i, channel)
            stack_widget = SaveStack(channel_name=channel, map_list=self.map_list)
            self.channels_stacks[channel] = stack_widget
            self.stack.addWidget(stack_widget)
            i += 1
        self.channels_list.currentRowChanged.connect(self.show_current_stack)
        # self.channels_list.setMinimumWidth(100)

        self.btn_save = QPushButton('Start saving')
        self.btn_save.setCheckable(True)
        self.btn_save.setStyleSheet("font-size: 14px;")
        self.btn_save.clicked.connect(self.btn_save_clicked)

    def add_widgets_to_layout(self):
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        save_detected_layout = QHBoxLayout()
        save_detected_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        save_detected_layout.addWidget(self.chb_save_detected)

        box_layout = QVBoxLayout()
        box_layout.addWidget(self.channels_list, alignment=Qt.AlignmentFlag.AlignTop)
        self.box_channels.setLayout(box_layout)

        central_layout = QHBoxLayout()
        central_layout.addWidget(self.box_channels, alignment=Qt.AlignmentFlag.AlignTop)
        central_layout.addWidget(self.stack, alignment=Qt.AlignmentFlag.AlignTop)

        self.main_layout.addLayout(save_detected_layout)
        self.main_layout.addSpacing(20)
        self.main_layout.addLayout(central_layout)
        self.main_layout.addWidget(self.btn_save)

    def show_current_stack(self, i):
        self.stack.setCurrentIndex(i)

    def btn_save_clicked(self):
        if not self.btn_save.isChecked():
            self.btn_save.setText('Start saving')
        else:
            self.btn_save.setText('Stop saving')


class SaveStack(QWidget):
    def __init__(self, channel_name: str, map_list: list):
        super().__init__()
        self.name = channel_name
        self.map_list = map_list

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.checkboxes = {}

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.box_classes = QGroupBox(f'Classes to save ({self.name})')
        self.box_classes.setMinimumWidth(170)
        checkbox_all_classes = QCheckBox('All images')
        checkbox_all_classes.stateChanged.connect(self.chb_all_classes_clicked)
        self.checkboxes['All'] = checkbox_all_classes
        for clas_name in self.map_list:
            checkbox = QCheckBox(clas_name)
            self.checkboxes[clas_name] = checkbox
        checkbox_clear = QCheckBox('Clear')
        self.checkboxes['Clear'] = checkbox_clear

    def add_widgets_to_layout(self):
        classes_layout = QVBoxLayout()
        classes_layout.setSpacing(0)
        for chb in self.checkboxes.values():
            chb_layout = QHBoxLayout()
            chb_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            chb_layout.addWidget(chb)
            classes_layout.addLayout(chb_layout)

        central_layout = QHBoxLayout()
        central_layout.addLayout(classes_layout)
        self.box_classes.setLayout(central_layout)
        self.main_layout.addWidget(self.box_classes, alignment=Qt.AlignmentFlag.AlignLeft)
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addSpacerItem(spacer)

    def chb_all_classes_clicked(self, state: int):
        if bool(state):
            for chb in self.checkboxes.values():
                chb.setChecked(True)
        else:
            for chb in self.checkboxes.values():
                chb.setChecked(False)