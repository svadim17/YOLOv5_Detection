from PyQt6.QtWidgets import QDialog, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QApplication
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import Qt
import qdarktheme
from collections import namedtuple


ChannelItem = namedtuple('Channel', ['name', 'checkbox', 'label'])


class ConnectWindow(QDialog):

    def __init__(self, ip: str, available_channels: list, channels_info: list):
        super().__init__()
        self.ip = ip
        self.available_channels = available_channels
        self.channels_info = channels_info

        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

        self.setWindowTitle('Connection parameters')
        self.setMinimumWidth(220)
        self.main_layout = QVBoxLayout()

        title = QGroupBox('Avaliable channels')
        title.setStyleSheet('font-size: 14px;')
        title.setLayout(self.main_layout)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(title)

        self.channels = []
        self.enabled_channels = []
        self.checkboxes = {}

        self.create_connect_controls()

    def create_connect_controls(self):
        self.btn_connect = QPushButton('Connect to channels')
        self.btn_connect.clicked.connect(self.btn_connect_clicked)
        if self.channels_info:
            for ch in self.channels_info:
                checkbox = QCheckBox(ch.name)
                checkbox.setChecked(True)
                checkbox.setStyleSheet('font-size: 12px;')
                self.checkboxes[ch.name] = checkbox
                l_hardware = QLabel(ch.hardware_type)
                channel = ChannelItem(name=ch.name, checkbox=checkbox, label=l_hardware)
                self.channels.append(channel)
            self.add_widgets_to_layout()
        else:
            self.main_layout.addWidget(QLabel('No available channels!'))

    def add_widgets_to_layout(self):
        layout_chb = QVBoxLayout()
        for channel in self.channels:
            chb_horiz_layout = QHBoxLayout()
            chb_horiz_layout.addWidget(channel.checkbox)
            chb_horiz_layout.addWidget(channel.label, alignment=Qt.AlignmentFlag.AlignRight)
            layout_chb.addLayout(chb_horiz_layout)
        self.main_layout.addLayout(layout_chb)
        self.layout().addWidget(self.btn_connect)

    def btn_connect_clicked(self):
        for chb in self.checkboxes.values():
            if chb.isChecked():
                self.enabled_channels.append(chb.text())
        self.close()


if __name__ == '__main__':
    app = QApplication([])
    qdarktheme.setup_theme()
    window = ConnectWindow(ip='127.0.0.1', available_channels=['2G4', '5G8'], hardware='USRP')
    window.show()
    app.exec()


