from PyQt6.QtWidgets import QDialog, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QApplication
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import Qt
import qdarktheme


class ConnectWindow(QDialog):

    def __init__(self, ip: str, available_channels: list):
        super().__init__()
        self.ip = ip
        self.available_channels = available_channels

        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

        self.setWindowTitle('Connection parameters')
        self.setMinimumWidth(220)
        self.main_layout = QVBoxLayout()

        title = QGroupBox('Avaliable channels')
        title.setStyleSheet('font-size: 14px;')
        title.setLayout(self.main_layout)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(title)

        self.checkboxes = {}
        self.enabled_channels = []

        self.create_connect_controls()

    def create_connect_controls(self):
        self.btn_connect = QPushButton('Connect to channels')
        self.btn_connect.clicked.connect(self.btn_connect_clicked)
        if self.available_channels:
            for name in self.available_channels:
                checkbox = QCheckBox(name)
                checkbox.setChecked(True)
                checkbox.setStyleSheet('font-size: 12px;')
                self.checkboxes[name] = checkbox
            self.add_widgets_to_layout()
        else:
            self.main_layout.addWidget(QLabel('No available channels!'))

    def add_widgets_to_layout(self):
        layout_chb = QVBoxLayout()
        for chb in self.checkboxes.values():
            layout_chb.addWidget(chb)
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
    window = ConnectWindow(ip='127.0.0.1', available_channels=['2G4', '5G8'])
    window.show()
    app.exec()


