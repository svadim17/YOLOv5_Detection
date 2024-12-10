from PyQt6.QtWidgets import (QWidget, QDockWidget, QApplication, QTabWidget, QSlider, QSpinBox, QCheckBox,
                             QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy, QMenu, QSpacerItem)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QFont, QCursor, QAction
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
import numpy as np
import cv2
import qdarktheme
from collections import deque
import time
import pyqtgraph
from recognition_options import RecognitionOptions
from process_options import ProcessOptions


class RecognitionWidget(QDockWidget, QWidget):

    def __init__(self, window_name,
                 map_list: list,
                 zscale_settings: list,
                 recogn_options: dict,
                 **widgets_statuses):
        super().__init__()
        self.name = window_name
        self.map_list = map_list
        self.zscale_settings = zscale_settings
        self.recogn_options_dict = recogn_options

        self.show_img_status = widgets_statuses.get('show_images', False)
        self.show_histogram_status = widgets_statuses.get('show_histogram', False)
        self.show_spectrum_status = widgets_statuses.get('show_spectrum', False)

        self.setWindowTitle(self.name)
        self.setWidget(QWidget(self))
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.widget().setLayout(self.main_layout)
        # self.setFixedSize(600, 600)
        # self.setMinimumSize(200, 200)  # Минимальный размер окна
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Позволяет растягиваться

        self.img_width, self.img_height = 320, 320
        self.drons_btns = {}
        self.drons_freq = {}
        self.last_fps = deque(maxlen=5)
        self.last_fps_2 = deque(maxlen=5)
        self.last_time = 0
        self.last_time_2 = 0

        self.recognOptions = RecognitionOptions(name=self.name,
                                                zscale_settings=zscale_settings,
                                                current_recogn_settings=self.recogn_options_dict)
        self.accum_size = self.recognOptions.spb_accum_size.value()
        self.recognOptions.spb_accum_size.valueChanged.connect(self.change_accum_size)

        self.processOptions = ProcessOptions(name=self.name)

        self.hist_deques = {name: deque(maxlen=self.accum_size) for name in self.map_list}
        self.create_widgets()

    @pyqtSlot(int)
    def change_accum_size(self, val: int):
        self.accum_size = val
        self.hist_deques = {name: deque(maxlen=self.accum_size) for name in self.map_list}
        label_style = {"color": "#EEE", "font-size": "10pt"}
        self.histogram_plot.setLabel("left", f"% of {self.accum_size} accumulation", **label_style)

    def context_menu(self):
        self.menu = QMenu(self)

        self.act_options = QAction('Recognition options')
        self.act_options.triggered.connect(self.recognOptions.show)

        self.act_process_status = QAction('Process options')
        self.act_process_status.triggered.connect(self.processOptions.show)

        self.menu.addAction(self.act_options)
        self.menu.addAction(self.act_process_status)
        self.menu.exec(QCursor.pos())

    def create_widgets(self):
        self.tab = QTabWidget()
        self.main_layout.addWidget(self.tab)
        if not self.show_img_status and not self.show_histogram_status and not self.show_spectrum_status:
            self.tab.hide()
        else:
            self.tab.show()

        self.create_buttons()
        self.create_spectrogram_tab()
        self.create_histogram_tab()
        self.create_spectrum_tab()

    def create_buttons(self):
        for name in self.map_list:
            drone_btn = QPushButton(name)
            self.drons_btns[name] = drone_btn
            drone_freq = QLabel('None')
            drone_freq.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.drons_freq[name] = drone_freq

        drons_btns_layout = QHBoxLayout()
        for drone_name in self.drons_btns.keys():
            drone_layout = QVBoxLayout()
            drone_layout.addWidget(self.drons_btns[drone_name])
            drone_layout.addWidget(self.drons_freq[drone_name])
            drons_btns_layout.addLayout(drone_layout)
        self.main_layout.addLayout(drons_btns_layout)

    def create_spectrogram_tab(self):
        self.img_frame = QLabel()
        pixmap = QPixmap('assets/icons/spectrum_img.png')
        self.img_frame.setPixmap(pixmap.scaled(self.img_width, self.img_height))
        self.img_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Центрирование изображения
        # self.img_frame.setScaledContents(True)  # масштабирование изображения по размеру виджета
        self.img_frame.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.img_frame.customContextMenuRequested.connect(self.context_menu)

        self.fps_painter = QPainter()
        self.fps_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.fps_painter.drawPixmap(0, 0, pixmap)
        self.fps_painter.drawText(10, 30, 'FPS = ∞')
        self.fps_painter.end()

        self.l_fps = QLabel()
        self.l_fps.setStyleSheet("font-size: 15px;")

        img_frame_layout = QHBoxLayout()
        img_frame_layout.addWidget(self.img_frame)

        self.spectrogram_widget = QWidget()
        self.spectrogram_widget.setLayout(img_frame_layout)

        if self.show_img_status:
            self.tab.addTab(self.spectrogram_widget, 'Spectrogram')
        else:
            self.spectrogram_widget.hide()

    def create_histogram_tab(self):
        self.histogram_plot = pyqtgraph.plot()
        self.histogram_plot.setYRange(0, 101, 0)
        self.histogram_plot.showAxis('left', True)
        label_style = {"color": "#EEE", "font-size": "10pt"}
        self.histogram_plot.setLabel("left", f"% of {self.accum_size} accumulation", **label_style)

        # Set names om X axis
        ticks = []
        for i in range(len(self.map_list)):
            ticks.append((i, self.map_list[i]))
        ticks = [ticks]
        ax = self.histogram_plot.getAxis('bottom')
        ax.setTicks(ticks)
        # self.histogram_widget.addWidget(self.plot, row=0, colspan=len(self.map_list))

        if self.show_histogram_status:
            self.tab.addTab(self.histogram_plot, 'Histogram')
        else:
            self.histogram_plot.hide()

    def create_spectrum_tab(self):
        self.spectrum_plot = pyqtgraph.PlotWidget()
        self.spectrum_curve = self.spectrum_plot.plot(pen=(229, 165, 10))

        if self.show_spectrum_status:
            self.tab.addTab(self.spectrum_plot, 'Spectrum')
        else:
            self.spectrum_plot.hide()

    @pyqtSlot(int)
    def show_frequencies(self, state: int):
        if bool(state):
            for dron_freq in self.drons_freq.values():
                dron_freq.show()
        else:
            for dron_freq in self.drons_freq.values():
                dron_freq.hide()

    def enable_spectrogram(self):
        self.show_img_status = True
        self.tab.addTab(self.spectrogram_widget, 'Spectrogram')
        if not self.tab.isVisible():
            self.tab.show()

    def disable_spectrogram(self):
        self.show_img_status = False
        self.spectrogram_widget.hide()
        self.remove_tab_by_name(name='Spectrogram')

    def enable_histogram(self):
        self.show_histogram_status = True
        self.tab.addTab(self.histogram_plot, 'Histogram')
        if not self.tab.isVisible():
            self.tab.show()

    def disable_histogram(self):
        self.show_histogram_status = False
        self.histogram_plot.hide()
        self.remove_tab_by_name(name='Histogram')

    def enable_spectrum(self):
        self.show_spectrum_status = True
        self.tab.addTab(self.spectrum_plot, 'Spectrum')
        if not self.tab.isVisible():
            self.tab.show()

    def disable_spectrum(self):
        self.show_spectrum_status = True
        self.spectrum_plot.hide()
        self.remove_tab_by_name(name='Spectrum')

    def remove_tab_by_name(self, name: str):
        for i in range(self.tab.count()):
            if self.tab.tabText(i) == name:
                self.tab.removeTab(i)
                break
        if self.tab.count() == 0:
            self.tab.hide()

    def resize_img(self, new_resolution: tuple[int, int]):
        self.img_width, self.img_height = new_resolution
        # self.img_frame.resize(self.img_width, self.img_height)
        self.img_frame.setFixedSize(self.img_width, self.img_height)

        self.setMaximumWidth(self.img_width + 40)
        self.setMaximumHeight(self.img_height + 150)

        self.parent().adjustSize()

    def convert_cv_qt(self, cv_img, fps: str):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # rgb_image = cv2.resize(rgb_image, (self.img_width, self.img_height))
        # rotate = self.rotate_cb.currentData()
        # if rotate is not None:
        # rgb_image = cv2.rotate(rgb_image, rotate)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        picture = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        picture = picture.scaled(self.img_width, self.img_height)

        painter = QPainter(picture)
        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont('Courier', 12, QFont.Weight.Light))
        painter.drawText(5, 20, fps)
        painter.end()
        return QPixmap.fromImage(picture)

    @pyqtSlot(dict)
    def update_state(self, info: dict):
        self.last_fps_2.append(1 / (time.time() - self.last_time_2))
        current_fps = f'FPS: {sum(self.last_fps_2) / len(self.last_fps_2):.1f}'
        self.last_time_2 = time.time()
        self.setWindowTitle(self.name + f' {current_fps}')

        band_name = info['band_name']
        for drone_dict in info['drones']:
            if drone_dict['state']:
                self.drons_btns[drone_dict['name']].setStyleSheet("background-color: #F0483C; "
                                                                  "font: bold; "
                                                                  "color: #FFFFFF")
                self.drons_freq[drone_dict['name']].setText(str(drone_dict['freq']))
            else:
                self.drons_btns[drone_dict['name']].setStyleSheet("background-color: ")
                self.drons_freq[drone_dict['name']].setText('None')
            self.hist_deques[drone_dict['name']].appendleft(int(drone_dict['state']))

        if self.histogram_plot:
            self.update_histogram_plot()

        if 'detected_img' in info:
            self.last_fps.append(1 / (time.time() - self.last_time))
            current_fps = f'FPS: {sum(self.last_fps) / len(self.last_fps):.1f}'
            self.last_time = time.time()

            img_arr = np.frombuffer(info['detected_img'], dtype=np.uint8).reshape((640, 640, 3))
            img_pixmap = self.convert_cv_qt(img_arr, fps=current_fps)
            self.img_frame.setPixmap(img_pixmap)

    def update_histogram_plot(self):
        accumed_val = [sum(deq) / self.accum_size * 100 for deq in self.hist_deques.values()]
        # colors = [sum(deq) for deq in self.hist_deques.values()]
        # Clear plot
        for item in self.histogram_plot.items():
            self.histogram_plot.removeItem(item)
        # Draw bars
        bar_item = pyqtgraph.BarGraphItem(x=np.arange(len(self.map_list)),
                                          height=accumed_val,
                                          width=0.7,
                                          brush=(255, 163, 72))
        self.histogram_plot.addItem(bar_item)

    def update_spectrum_plot(self, data):
        self.spectrum_curve.setData(y=data)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    main_window = RecognitionWidget(window_name='afasd',
                                    map_list=['1', '2', '3', '4'],
                                    img_show_status=True,
                                    zscale_settings=[-20, 20],
                                    recogn_options={'accum_size': 10, 'threshold': 0.5, 'exceedance': 0.7})
    main_window.show()
    sys.exit(app.exec())
