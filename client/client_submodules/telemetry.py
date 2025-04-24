import loguru
from PyQt6.QtWidgets import (QWidget, QListWidget, QApplication, QSpacerItem, QSizePolicy,
                             QStackedWidget, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QTabWidget, QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPainter, QPen
from PyQt6.QtCore import pyqtSlot, Qt, pyqtSignal, QRect
import qdarktheme
from os import walk
import yaml


class TelemetryWidget(QWidget):
    def __init__(self, theme_type: str):
        super().__init__()
        self.theme_type = theme_type

        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle('Server`s Telemetry')
        self.setWindowIcon(QIcon(f'assets/icons/light/telemetry.png'))
        self.setFixedSize(550, 480)

        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)
        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.box_cpu = QGroupBox('CPU')
        self.cpu_load_circle = CircularProgress(label='LOAD')
        self.cpu_temp_label = QLabel('TEMP: --°C')
        self.cpu_temp_label.setFont(QFont("Arial", 14, QFont.Weight.Normal))

        self.box_gpu = QGroupBox('GPU')
        self.gpu_load_circle = CircularProgress(label='LOAD')
        self.gpu_memory_circle = CircularProgress(label='MEM')
        self.gpu_progress_label = ProgressLabel(postfix='GB')
        self.gpu_temp_label = QLabel('TEMP: --°C')
        self.gpu_temp_label.setFont(QFont("Arial", 14, QFont.Weight.Medium))


        self.box_ram = QGroupBox('RAM')
        self.ram_memory_circle = CircularProgress(label='MEM')
        self.ram_progress_label = ProgressLabel(postfix='GB')

        self.box_network = QGroupBox('Network')
        self.network_up_icon = QLabel()
        pixmap = QPixmap(f'./assets/icons/{self.theme_type}/upload.png')
        self.network_up_icon.setPixmap(pixmap.scaled(25, 25))
        self.network_up_label = QLabel('Up: -- Mb/s')
        self.network_up_label.setFont(QFont("Arial", 14, QFont.Weight.Medium))
        self.network_down_icon = QLabel()
        pixmap = QPixmap(f'./assets/icons/{self.theme_type}/download.png')
        self.network_down_icon.setPixmap(pixmap.scaled(25, 25))
        self.network_down_label = QLabel('Down: -- Mb/s')
        self.network_down_label.setFont(QFont("Arial", 14, QFont.Weight.Medium))

        self.cpu_load_circle.setFixedSize(120, 120)
        self.gpu_load_circle.setFixedSize(120, 120)
        self.gpu_memory_circle.setFixedSize(120, 120)
        self.ram_memory_circle.setFixedSize(120, 120)

    def add_widgets_to_layout(self):
        cpu_layout = QHBoxLayout()
        cpu_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        cpu_layout.addWidget(self.cpu_load_circle)
        cpu_layout.addWidget(self.cpu_temp_label)
        self.box_cpu.setLayout(cpu_layout)

        gpu_layout = QVBoxLayout()
        gpu_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        first_line_layout = QHBoxLayout()
        first_line_layout.addWidget(self.gpu_load_circle)
        first_line_layout.addWidget(self.gpu_temp_label)
        second_line_layout = QHBoxLayout()
        second_line_layout.addWidget(self.gpu_memory_circle)
        second_line_layout.addWidget(self.gpu_progress_label)
        gpu_layout.addLayout(first_line_layout)
        gpu_layout.addLayout(second_line_layout)
        self.box_gpu.setLayout(gpu_layout)

        ram_layout = QHBoxLayout()
        ram_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        ram_layout.addWidget(self.ram_memory_circle)
        ram_layout.addWidget(self.ram_progress_label)
        self.box_ram.setLayout(ram_layout)

        network_layout = QVBoxLayout()
        network_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        network_up_layout = QHBoxLayout()
        network_up_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        network_up_layout.addWidget(self.network_up_icon)
        network_up_layout.addWidget(self.network_up_label)
        network_down_layout = QHBoxLayout()
        #network_down_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        network_down_layout.addWidget(self.network_down_icon)
        network_down_layout.addWidget(self.network_down_label)
        network_layout.addLayout(network_up_layout)
        network_layout.addLayout(network_down_layout)
        self.box_network.setLayout(network_layout)

        first_line_main_layout = QHBoxLayout()
        first_line_main_layout.addWidget(self.box_cpu)
        first_line_main_layout.addWidget(self.box_gpu)

        second_line_main_layout = QHBoxLayout()
        second_line_main_layout.addWidget(self.box_ram)
        second_line_main_layout.addWidget(self.box_network)

        self.main_layout.addLayout(first_line_main_layout)
        self.main_layout.addLayout(second_line_main_layout)

    def udpate_widgets_states(self, info: dict):
        if 'CPU load' in info:
            self.cpu_load_circle.setValue(val=info['CPU load'])
        if 'CPU temp' in info:
            temp = info['CPU temp']
            self.cpu_temp_label.setText(f'TEMP: {temp}°C')
        if 'GPU load' in info:
            self.gpu_load_circle.setValue(val=info['GPU load'])
        if 'GPU mem_used' in info and 'GPU mem_total' in info:
            self.gpu_memory_circle.setValue(val=info['GPU mem_used'] / info['GPU mem_total'] * 100)
            self.gpu_progress_label.update_value(info['GPU mem_used'] / 1024, info['GPU mem_total'] / 1024)
        if 'GPU temp' in info:
            temp = info['GPU temp']
            self.gpu_temp_label.setText(f'TEMP: {temp}°C')
        if 'RAM used' in info and 'RAM total' in info:
            self.ram_memory_circle.setValue(val=info['RAM used'] / info['RAM total'] * 100)
            self.ram_progress_label.update_value(info['RAM used'] / 1024, info['RAM total'] / 1024)
        if 'NET up' in info and 'NET down' in info:
            up, down = info['NET up'], info['NET down']
            self.network_up_label.setText(f'Up: {(up / 128):.1f} Mb/s')       # приходит в КБ, а перевожу в Мб (делю на 128)
            self.network_down_label.setText(f'Down: {(down / 128):.1f} Mb/s')       # приходит в КБ, а перевожу в Мб (делю на 128)


class CircularProgress(QWidget):
    def __init__(self, label, color=QColor("#00ffff")):
        super().__init__()
        self.value = 0
        self.label = label
        self.color = color
        # self.setMinimumSize(50, 50)

    def setValue(self, val):
        self.value = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        pen_width = 8
        radius = int(min(rect.width(), rect.height()) / 2 - pen_width)

        center = rect.center()
        painter.translate(center)
        painter.rotate(-90)

        arc_rect = QRect(-radius, -radius, radius * 2, radius * 2)

        # Draw background circle
        pen = QPen(QColor("#003333"), pen_width)
        painter.setPen(pen)
        painter.drawArc(arc_rect, 0, 360 * 16)

        # Draw value arc
        pen.setColor(self.color)
        painter.setPen(pen)
        angle = int(360 * self.value / 100)
        painter.drawArc(arc_rect, 0, -angle * 16)

        # Draw text
        painter.resetTransform()
        painter.setPen(QColor("white"))
        font = QFont("Arial", 14, QFont.Weight.Medium)
        painter.setFont(font)
        text = f"{self.label}\n{self.value:.0f}%"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class ProgressLabel(QLabel):
    def __init__(self, postfix='GB'):
        super().__init__()
        self.postfix = postfix
        self.setFont(QFont("Arial", 14, QFont.Weight.Normal))
        # self.setStyleSheet("color: #00ff99; background-color: #222; padding: 6px; border-radius: 6px;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText(f'-- / --  {self.postfix}')

    def update_value(self, min: float, max: float):
        self.setText(f'{min:.1f} / {max:.1f}  {self.postfix}')


# class MemoryProgress2(QWidget):
#     def __init__(self, label, color=QColor("#00ffff"), value_postfix='GB'):
#         super().__init__()
#         self.postfix = value_postfix
#         self.value = 0
#         self.label = label
#         self.value_label = ''
#         self.color = color
#         self.setMinimumSize(50, 50)
#
#     def setValue(self, used: float, total: float):
#         self.value = used / total * 100
#         self.value_label = f'{used:.1f} / {total:.1f} {self.postfix}'
#         self.update()
#
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.RenderHint.Antialiasing)
#
#         rect = self.rect()
#         pen_width = 8
#         radius = int(min(rect.width(), rect.height()) / 2 - pen_width)
#
#         center = rect.center()
#         painter.translate(center)
#         painter.rotate(-90)
#
#         arc_rect = QRect(-radius, -radius, radius * 2, radius * 2)
#
#         # Draw background circle
#         pen = QPen(QColor("#222"), pen_width)
#         painter.setPen(pen)
#         painter.drawArc(arc_rect, 0, 360 * 16)
#
#         # Draw value arc
#         pen.setColor(self.color)
#         painter.setPen(pen)
#         angle = int(360 * self.value / 100)
#         painter.drawArc(arc_rect, 0, -angle * 16)
#
#         # Draw text
#         painter.resetTransform()
#         painter.setPen(QColor("white"))
#         font = QFont("Arial", 16, QFont.Weight.Bold)
#         painter.setFont(font)
#         text = f"{self.label}\n{self.value:.0f}%"
#         painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
#
#         # Draw label below circle
#         label_rect = QRect(rect.left(), rect.bottom() - 30, rect.width(), 30)
#         font = QFont("Arial", 12)
#         painter.setFont(font)
#         painter.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.value_label)


# class MemoryProgress(QWidget):
#     def __init__(self, label, value_postfix='GB'):
#         super().__init__()
#         self.l_circular_progress = label
#         self.postfix = value_postfix
#         self.main_layout = QVBoxLayout()
#         self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.main_layout.setSpacing(10)
#         self.setLayout(self.main_layout)
#         self.create_widgets()
#         self.add_widgets_to_layout()
#
#     def create_widgets(self):
#         self.circularProgress = CircularProgress(label=self.l_circular_progress)
#         self.l_memory = QLabel()
#         self.l_memory.setText(f'{0} / {0}  {self.postfix}')
#
#     def add_widgets_to_layout(self):
#         self.main_layout.addWidget(self.circularProgress)
#         self.main_layout.addWidget(self.l_memory)
#
#     def setValue(self, used: float, total: float):
#         self.circularProgress.setValue(val=used / total * 100)
#         self.l_memory.setText(f'{used:.1f} / {total:.1f}  {self.postfix}')


if __name__ == '__main__':
    app = QApplication([])
    qdarktheme.setup_theme()
    window = TelemetryWidget(theme_type='dark')
    window.show()
    app.exec()
