from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QApplication, QScrollArea
)
from PySide6.QtCore import Qt
import sys


class ChannelsWidget(QDockWidget):
    def __init__(self, widgets: list):
        super().__init__()
        self.setWindowTitle('Channels')
        self.setMinimumSize(500, 450)
        self.widgets = widgets

        container = QWidget()
        self.main_layout = QVBoxLayout(container)

        # scroll_area = QScrollArea()
        # scroll_area.setWidgetResizable(True)
        # scroll_area.setWidget(container)
        # scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # self.setWidget(scroll_area)

        self.setWidget(container)
        self.add_widgets_to_layout()

    def add_widgets_to_layout(self):
        first_line = QHBoxLayout()
        second_line = QHBoxLayout()

        for widget in self.widgets:
            first_line.addWidget(widget)
        self.main_layout.addLayout(first_line)

        # if len(self.widgets) <= 2:
        #     for widget in self.widgets:
        #         first_line.addWidget(widget)
        #     self.main_layout.addLayout(first_line)
        # elif len(self.widgets) > 2:
        #     counter = 0
        #     for widget in self.widgets:
        #         if counter % 2 == 0:
        #             first_line.addWidget(widget)
        #         else:
        #             second_line.addWidget(widget)
        #         counter += 1
        self.main_layout.addLayout(first_line)
        # self.main_layout.addLayout(second_line)
