from PySide6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QScrollArea, QMessageBox, QGroupBox)
from PySide6.QtCore import Qt, QDateTime
from PySide6.QtGui import QIcon
from collections import namedtuple
from datetime import datetime
import random


RemoteIdObject = namedtuple('RemoteId', ['mac_address', 'distance', 'azimuth'])


class RemoteIdWidget(QDockWidget, QWidget):
    def __init__(self, theme_type: str, history_size: int = 10):
        super().__init__()
        self.setWindowTitle('Remote ID')
        self.theme_type = theme_type
        self.history_size = history_size

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.main_layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout()
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.list_widget.setLayout(self.list_layout)
        self.scroll_area.setWidget(self.list_widget)

        self.clear_button = QPushButton("Clear all")
        self.clear_button.clicked.connect(self.clear_all_items)

        self.main_layout.addWidget(self.scroll_area)
        self.main_layout.addWidget(self.clear_button, alignment=Qt.AlignmentFlag.AlignBottom)

        self.mac_counter = 0

    def add_remote_id(self, new_object: RemoteIdObject):
        box_object = QGroupBox(f'MAC: {new_object.mac_address}')

        icon_label = QLabel()
        icon_label.setPixmap(QIcon(f"../assets/icons/{self.theme_type}/remote_id.png").pixmap(24, 24))

        l_time = QLabel(f'Updated: {datetime.now().strftime("%H:%M:%S")}')
        l_distance = QLabel(f'Distance: {RemoteIdObject.distance}')
        l_azimith = QLabel(f'Azimuth: {RemoteIdObject.distance}')

        remove_button = QPushButton("Remove item")
        remove_button.setStyleSheet("color: red; font-weight: bold;")
        remove_button.clicked.connect(lambda _, w=box_object: self.remove_item(w))

        box_layout = QHBoxLayout()
        box_layout.setContentsMargins(0, 0, 0, 0)
        left_layout = QVBoxLayout()
        left_layout.addWidget(icon_label)
        right_layout = QVBoxLayout()
        right_layout.addWidget(l_time)
        right_layout.addWidget(l_distance)
        right_layout.addWidget(l_azimith)
        box_layout.addLayout(left_layout)
        box_layout.addLayout(right_layout)
        box_layout.addWidget(remove_button)
        box_object.setLayout(box_layout)
        self.list_layout.addWidget(box_object)

        # Ограничение количества отображаемых элементов
        if self.list_layout.count() > self.history_size:
            old_widget = self.list_layout.itemAt(0).widget()
            self.list_layout.removeWidget(old_widget)
            old_widget.deleteLater()

    def remove_item(self, item: QWidget):
        self.list_layout.removeWidget(item)
        item.deleteLater()

    def clear_all_items(self):
        confirm = QMessageBox.question(self, "Clear all", "Delete all elements Remote ID?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            while self.list_layout.count():
                widget = self.list_layout.itemAt(0).widget()
                self.list_layout.removeWidget(widget)
                widget.deleteLater()

    def emulate(self):
        obj = RemoteIdObject(mac_address=f"DE:AD:BE:EF:{self.mac_counter:02X}:{random.randint(0, 255):02X}",
                             distance=round(random.uniform(10, 3000), 1),
                             azimuth=round(random.uniform(0, 360), 1))

        self.add_remote_id(new_object=obj)
        self.mac_counter += 1
