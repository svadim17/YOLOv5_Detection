from PyQt6.QtWidgets import (QWidget, QListWidget, QApplication, QSpacerItem, QSizePolicy,
                             QStackedWidget, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QTabWidget, QComboBox)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import pyqtSlot, Qt, pyqtSignal
import qdarktheme
import numpy as np
import cv2


class SettingsWidget(QWidget):
    def __init__(self, enabled_channels: list, map_list: list):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle('Settings')
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)

        self.mainTab = MainTab()
        self.saveTab = SaveTab(enabled_channels=enabled_channels, map_list=map_list)

        self.tab = QTabWidget()
        self.tab.addTab(self.mainTab, 'Main')
        self.tab.addTab(self.saveTab, 'Images saving')

        self.main_layout.addWidget(self.tab)


class MainTab(QWidget):

    def __init__(self):
        super().__init__()
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)
        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.l_spectrogram_resolution = QLabel('Spectrogram resolution')
        self.cb_spectrogram_resolution = QComboBox()
        self.cb_spectrogram_resolution.setMinimumWidth(100)
        self.cb_spectrogram_resolution.addItem('240x240', (240, 240))
        self.cb_spectrogram_resolution.addItem('320x320', (320, 320))
        self.cb_spectrogram_resolution.addItem('400x400', (400, 400))
        self.cb_spectrogram_resolution.addItem('480x480', (480, 480))
        self.cb_spectrogram_resolution.addItem('560x560', (560, 560))
        self.cb_spectrogram_resolution.addItem('640x640', (640, 640))
        self.cb_spectrogram_resolution.setCurrentIndex(1)

        self.l_chb_show_zscale = QLabel('Show ZScale settings')
        self.chb_show_zscale = QCheckBox()
        self.chb_show_zscale.setChecked(True)

        self.l_show_frequencies = QLabel('Show frequencies')
        self.chb_show_frequencies = QCheckBox()
        self.chb_show_frequencies.setChecked(True)

        self.l_accummulation = QLabel('Enable accumulation')
        self.chb_accumulation = QCheckBox()
        self.chb_accumulation.setChecked(True)

        self.btn_save_client_config = QPushButton('Save client config')

        self.btn_save_server_config = QPushButton('Save server config')

    def add_widgets_to_layout(self):
        spectr_layout = QVBoxLayout()
        spectr_layout.setSpacing(5)
        spectr_layout.addWidget(self.l_spectrogram_resolution, alignment=Qt.AlignmentFlag.AlignLeft)
        spectr_layout.addWidget(self.cb_spectrogram_resolution, alignment=Qt.AlignmentFlag.AlignLeft)

        zscale_layout = QHBoxLayout()
        zscale_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        zscale_layout.addWidget(self.chb_show_zscale)
        zscale_layout.addWidget(self.l_chb_show_zscale)

        frequencies_layout = QHBoxLayout()
        frequencies_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        frequencies_layout.addWidget(self.chb_show_frequencies)
        frequencies_layout.addWidget(self.l_show_frequencies)

        accum_layout = QHBoxLayout()
        accum_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        accum_layout.addWidget(self.chb_accumulation)
        accum_layout.addWidget(self.l_accummulation)

        btns_layout = QHBoxLayout()
        btns_layout.addWidget(self.btn_save_client_config)
        btns_layout.addWidget(self.btn_save_server_config)

        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.main_layout.addLayout(spectr_layout)
        self.main_layout.addLayout(zscale_layout)
        self.main_layout.addLayout(frequencies_layout)
        self.main_layout.addLayout(accum_layout)
        self.main_layout.addSpacerItem(spacer)
        self.main_layout.addLayout(btns_layout)


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
        self.chb_save_detected = QCheckBox()
        self.l_save_detected = QLabel('Save detected')
        self.l_save_detected.setStyleSheet("font-size: 14px")

        self.channels_list = QListWidget()
        self.channels_list.setMinimumWidth(200)
        self.stack = QStackedWidget()

        i = 0
        for channel in self.enabled_channels:
            self.channels_list.insertItem(i, channel)
            stack_widget = SaveStack(channel_name=channel, map_list=self.map_list)
            self.channels_stacks[channel] = stack_widget
            self.stack.addWidget(stack_widget)
            i += 1
        self.channels_list.currentRowChanged.connect(self.show_current_stack)

        self.btn_save = QPushButton('Start saving')
        self.btn_save.setCheckable(True)
        self.btn_save.setStyleSheet("font-size: 14px;")
        self.btn_save.clicked.connect(self.btn_save_clicked)

    def add_widgets_to_layout(self):
        save_detected_layout = QHBoxLayout()
        save_detected_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        save_detected_layout.addWidget(self.chb_save_detected)
        save_detected_layout.addWidget(self.l_save_detected)

        central_layout = QHBoxLayout()
        central_layout.addWidget(self.channels_list)
        central_layout.addWidget(self.stack)

        self.main_layout.addLayout(save_detected_layout)
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
        self.checkboxes_labels = {}

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.l_stack = QLabel(self.name)
        self.l_stack.setStyleSheet("font-size: 16px;")

        self.l_classes = QLabel('Classes to save')
        self.l_classes.setStyleSheet("font-size: 14px")

        checkbox_all_classes = QCheckBox()
        checkbox_all_classes.stateChanged.connect(self.chb_all_classes_clicked)
        label = QLabel('All images')
        self.checkboxes['All'] = checkbox_all_classes
        self.checkboxes_labels['All'] = label
        for clas_name in self.map_list:
            checkbox = QCheckBox()
            label = QLabel(clas_name)
            self.checkboxes[clas_name] = checkbox
            self.checkboxes_labels[clas_name] = label
        checkbox_clear = QCheckBox()
        label = QLabel('Clear')
        self.checkboxes['Clear'] = checkbox_clear
        self.checkboxes_labels['Clear'] = label

    def add_widgets_to_layout(self):
        classes_layout = QVBoxLayout()
        classes_layout.setSpacing(0)
        for chb, label in zip(self.checkboxes.values(), self.checkboxes_labels.values()):
            chb_layout = QHBoxLayout()
            chb_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            chb_layout.addWidget(chb)
            chb_layout.addWidget(label)
            classes_layout.addLayout(chb_layout)

        central_layout = QHBoxLayout()
        central_layout.addWidget(self.l_classes, alignment=Qt.AlignmentFlag.AlignTop)
        central_layout.addSpacing(50)
        central_layout.addLayout(classes_layout)

        self.main_layout.addWidget(self.l_stack, alignment=Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
        self.main_layout.addSpacing(20)
        self.main_layout.addLayout(central_layout)
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addSpacerItem(spacer)

    def chb_all_classes_clicked(self, state: int):
        if bool(state):
            for chb in self.checkboxes.values():
                chb.setChecked(True)
        else:
            for chb in self.checkboxes.values():
                chb.setChecked(False)


if __name__ == '__main__':
    app = QApplication([])
    qdarktheme.setup_theme()
    window = SettingsWidget(enabled_channels=['2G4', '1G2', '5G8'], map_list=['Autel', 'Fpv', 'Dji', 'WiFi'])
    window.show()
    app.exec()
