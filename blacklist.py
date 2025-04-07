from PyQt6.QtWidgets import (QWidget, QDockWidget, QApplication, QTabWidget, QSlider, QSpinBox, QCheckBox,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QSizePolicy, QMenu,
                             QTreeWidget, QTreeWidgetItem, QMainWindow)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QFont, QCursor, QAction
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
import numpy as np
import cv2
import qdarktheme
from collections import deque
import time
import pyqtgraph
from client.client_submodules.recognition_options import RecognitionOptions
from client.client_submodules.process_options import ProcessOptions
from client.client_submodules.gRPC_thread import ChannelInfo
import sys

class DroneTreeWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Frequency Tree")
        self.setGeometry(100, 100, 800, 600)

        # Пример данных
        self.channel_info = type('ChannelInfo', (), {'central_freq': [2400000000, 4900000000]})()
        self.map_list = ["Autel", "Fp"]
        self.drons_btns = {}
        self.drons_freq = {}
        self.freq_labels = {}
        self.drone_btns_all_freq = {}

        # Основной layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(self.main_widget)

        # Создаем кнопки
        self.create_buttons()

        # Подключаем сигнал изменения размера окна
        self.resizeEvent = self.on_resize

    def create_buttons(self):
        self.freq_tree = QTreeWidget()
        self.freq_tree.setColumnCount(len(self.map_list) + 1)
        self.freq_tree.setHeaderHidden(True)

        # Отключаем полосы прокрутки
        self.freq_tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.freq_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Устанавливаем политику размеров
        self.freq_tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Устанавливаем фиксированную ширину колонок
        self.freq_tree.setColumnWidth(0, 100)  # Фиксированная ширина для колонки с частотами
        for col in range(1, len(self.map_list) + 1):
            self.freq_tree.setColumnWidth(col, 80)  # Фиксированная ширина для колонок с дронами

        all_freq_tree_item = QTreeWidgetItem(self.freq_tree)
        self.btn_all_freq = QPushButton('ALL freq')
        self.btn_all_freq.clicked.connect(lambda checked, item=all_freq_tree_item: self.toggle_frequencies(item))
        self.freq_tree.setItemWidget(all_freq_tree_item, 0, self.btn_all_freq)

        self.drone_btns_all_freq = {}
        for col, drone in enumerate(self.map_list, 1):
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            drone_btn = QPushButton(drone)
            drone_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            drone_freq = QLabel('None')
            drone_freq.setAlignment(Qt.AlignmentFlag.AlignCenter)
            drone_freq.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layout.addWidget(drone_btn)
            layout.addWidget(drone_freq)

            self.drone_btns_all_freq[drone] = drone_btn
            self.freq_tree.setItemWidget(all_freq_tree_item, col, container)

        self.main_layout.addWidget(self.freq_tree)
        self.adjust_tree_size()

    def toggle_frequencies(self, item):
        if item.childCount() == 0:
            for freq in self.channel_info.central_freq:
                freq_item = QTreeWidgetItem(item)

                freq_label = QLabel(f'{freq/1000_000_000:.2f}GHz')
                freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.freq_tree.setItemWidget(freq_item, 0, freq_label)

                drones_btns = {}
                drones_freqs = {}
                for col, drone in enumerate(self.map_list, 1):
                    container = QWidget()
                    drone_layout = QVBoxLayout(container)
                    drone_layout.setContentsMargins(0, 0, 0, 0)
                    drone_layout.setSpacing(2)

                    drone_btn = QPushButton(drone)
                    drone_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                    drone_freq = QLabel('None')
                    drone_freq.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    drone_freq.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

                    drone_layout.addWidget(drone_btn)
                    drone_layout.addWidget(drone_freq)
                    self.freq_tree.setItemWidget(freq_item, col, container)

                    drones_btns[drone] = drone_btn
                    drones_freqs[drone] = drone_freq

                self.drons_btns[freq] = drones_btns
                self.drons_freq[freq] = drones_freqs
                self.freq_labels[freq] = freq_label

            item.setExpanded(True)
        else:
            is_expanded = item.isExpanded()
            item.setExpanded(not is_expanded)
        self.adjust_tree_size()

    def adjust_tree_size(self):
        """Подгоняем размер дерева под содержимое"""
        # Подсчитываем количество видимых строк
        visible_rows = 1  # Начальная строка "ALL freq"
        root = self.freq_tree.topLevelItem(0)
        if root and root.isExpanded():
            visible_rows += root.childCount()

        # Рассчитываем высоту на основе видимых строк
        row_height = self.freq_tree.sizeHintForRow(0)  # Высота одной строки
        total_height = row_height * visible_rows

        # Рассчитываем общую ширину на основе фиксированных колонок
        total_width = 100 + 80 * len(self.map_list)  # 100 для первой колонки + 80 для каждой колонки дронов

        # Устанавливаем размеры дерева
        self.freq_tree.setMinimumHeight(total_height)
        self.freq_tree.setMaximumHeight(total_height)
        self.freq_tree.setMinimumWidth(total_width)

    def on_resize(self, event):
        """Обработчик изменения размера окна"""
        self.adjust_tree_size()
        super().resizeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = DroneTreeWidget()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()