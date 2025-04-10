import time
from PyQt6.QtWidgets import (QDialog, QApplication, QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QGroupBox, QComboBox)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtCore import Qt
import qdarktheme


class WelcomeWindow(QDialog):
    signal_connect_to_server = pyqtSignal(str, str)

    def __init__(self, server_addr: list, server_port: str):
        super().__init__()
        self.server_addr = server_addr
        self.server_port = server_port
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle('Welcome window')
        self.setMinimumWidth(350)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.group_box = QGroupBox('Welcome to NN Application!')
        self.group_box.setStyleSheet('font-size: 14px;')

        # self.le_server_addr = QLineEdit()
        # self.le_server_addr.setPlaceholderText("Enter server address")
        # self.le_server_addr.setText(self.server_addr)

        self.l_server_addr = QLabel('Server address')
        self.cb_server_addr = QComboBox()
        self.cb_server_addr.setPlaceholderText('Enter server address')      # appears when field is empty
        self.cb_server_addr.setEditable(True)
        self.cb_server_addr.addItems(self.server_addr)
        self.cb_server_addr.setCurrentIndex(0)

        self.l_server_port = QLabel('Server port')
        self.le_server_port = QLineEdit()
        self.le_server_port.setPlaceholderText("Enter server port")
        self.le_server_port.setText(self.server_port)

        self.btn_connect = QPushButton('Connect to server')
        self.btn_connect.clicked.connect(self.btn_connect_clicked)

    def add_widgets_to_layout(self):
        addr_layout = QVBoxLayout()
        addr_layout.addWidget(self.l_server_addr)
        addr_layout.addWidget(self.cb_server_addr)

        port_layout = QVBoxLayout()
        port_layout.addWidget(self.l_server_port)
        port_layout.addWidget(self.le_server_port)

        controls_layout = QHBoxLayout()
        controls_layout.addLayout(addr_layout)
        controls_layout.addLayout(port_layout)

        group_box_layout = QVBoxLayout()
        group_box_layout.addSpacing(15)
        group_box_layout.addLayout(controls_layout)
        self.group_box.setLayout(group_box_layout)

        self.main_layout.addWidget(self.group_box)
        self.main_layout.addWidget(self.btn_connect)

    def btn_connect_clicked(self):
        self.signal_connect_to_server.emit(self.cb_server_addr.currentText(), self.le_server_port.text())


if __name__ == '__main__':
    app = QApplication([])
    qdarktheme.setup_theme()
    window = WelcomeWindow(server_addr='127.0.0.1', server_port='51234')
    window.show()
    app.exec()
