from PySide6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QScrollArea, QMessageBox, QGroupBox)
from PySide6.QtCore import Qt, QDateTime
from PySide6.QtGui import QIcon, QPixmap
from collections import namedtuple
from datetime import datetime
import random


AeroscopeObject = namedtuple('Aeroscope', ['name', 'serial_number', 'distance', 'altitude'])


class AeroscopeWidget(QDockWidget):
    def __init__(self, theme_type: str, logger_, history_size: int = 20):
        super().__init__()
        # self.setMaximumWidth(400)
        self.setMinimumWidth(300)

        self.setTitleBarWidget(QWidget())
        # self.setWindowTitle('Aeroscope')
        self.theme_type = theme_type
        self.logger = logger_
        self.history_size = history_size
        self.pixmap = QPixmap(f"./assets/icons/drones/dji_drone.png").scaled(
            100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.container = QWidget()
        self.setWidget(self.container)

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.container.setLayout(self.main_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout()
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.list_widget.setLayout(self.list_layout)
        self.scroll_area.setWidget(self.list_widget)

        self.clear_button = QPushButton("Clear all")
        self.clear_button.setMinimumWidth(200)
        self.clear_button.clicked.connect(self.clear_all_items)

        self.main_layout.addWidget(self.scroll_area)
        self.main_layout.addWidget(self.clear_button, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter)

        self.mac_counter = 0

    def add_drone(self, new_object: AeroscopeObject):
        self.logger.debug(f'Adding drone object..')

        box_object = QGroupBox(f'{new_object.name}')

        icon_label = QLabel()
        icon_label.setPixmap(self.pixmap)

        l_time = QLabel(f'Updated: {datetime.now().strftime("%H:%M:%S")}')
        l_distance = QLabel(f'Distance: {new_object.distance}')
        l_altitude = QLabel(f'Altitude: {new_object.altitude}')
        l_serial_numb = QLabel(f'S/N: {new_object.serial_number}')

        remove_button = QPushButton("Remove item")
        remove_button.setStyleSheet("color: red; font-weight: bold;")
        remove_button.setMinimumWidth(150)
        remove_button.clicked.connect(lambda _, w=box_object: self.remove_item(w))

        box_layout = QVBoxLayout()
        box_layout.setContentsMargins(5, 10, 5, 10)
        left_layout = QVBoxLayout()
        left_layout.addWidget(icon_label)
        right_layout = QVBoxLayout()
        right_layout.addWidget(l_time)
        right_layout.addWidget(l_distance)
        right_layout.addWidget(l_altitude)
        right_layout.addWidget(l_serial_numb)
        box_main_layout = QHBoxLayout()
        box_main_layout.addLayout(left_layout)
        box_main_layout.addLayout(right_layout)

        box_layout.addLayout(box_main_layout)
        box_layout.addWidget(remove_button, alignment=Qt.AlignmentFlag.AlignCenter)
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
        self.logger.debug(f'Emulating remote id object..')

        obj = AeroscopeObject(name='DJI Mavic 3',
                              distance=round(random.uniform(10, 3000), 1),
                              altitude=round(random.uniform(0, 300), 1),
                              serial_number='HFDJH4HJ8934FDF3')

        self.add_drone(new_object=obj)
        self.mac_counter += 1

    def theme_changed(self, type: str):
        self.theme_type = type
        self.pixmap = QPixmap(f"./assets/icons/{self.theme_type}/remote_id.png")
