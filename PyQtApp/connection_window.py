from PyQt6.QtWidgets import QDialog, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal


class ConnectWindow(QDialog):
    signal_init_connections = pyqtSignal(list)

    def __init__(self, ip: str, grpc_port: str, grpc_thread):
        super().__init__()
        self.ip = ip
        self.gRPC_port = grpc_port
        self.gRPCThread = grpc_thread

        self.setWindowTitle('Connection parameters')
        self.setFixedSize(400, 300)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.available_channels = None
        self.checkboxes = {}
        self.enabled_channels = []

        self.gRPCThread.connect_to_gRPC_server(ip=self.ip, port=self.gRPC_port)
        self.available_channels = self.gRPCThread.getAvailableChannelsRequest()

        self.create_connect_controls()

    def create_connect_controls(self):
        self.btn_connect = QPushButton('Connect to channels')
        self.btn_connect.clicked.connect(self.btn_connect_clicked)

        if self.available_channels:
            for name in self.available_channels:
                checkbox = QCheckBox(name)
                checkbox.setChecked(True)
                self.checkboxes[name] = checkbox
            self.add_widgets_to_layout()
        else:
            self.main_layout.addWidget(QLabel('No available channels!'))

    def add_widgets_to_layout(self):
        layout_chb = QVBoxLayout()
        for chb in self.checkboxes.values():
            layout_chb.addWidget(chb)
        self.main_layout.addLayout(layout_chb)
        self.main_layout.addWidget(self.btn_connect)

    def btn_connect_clicked(self):
        for chb in self.checkboxes.values():
            if chb.isChecked():
                self.enabled_channels.append(chb.text())
        self.signal_init_connections.emit(self.enabled_channels)
        self.close()




