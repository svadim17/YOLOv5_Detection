from PyQt6.QtWidgets import QWidget, QDockWidget, QApplication, QTabWidget, QSlider, QSpinBox
from PyQt6.QtWidgets import QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy, QMenu
from PyQt6.QtGui import QPixmap, QImage, QPainter, QFont, QCursor, QAction
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
import numpy as np
import cv2
import qdarktheme
from collections import deque
import time
import pyqtgraph
from recognition_options import RecognitionOptions


class RecognitionWidget(QDockWidget, QWidget):

    def __init__(self, window_name, map_list: list, img_show_status: bool, zscale_settings: list, recogn_options: dict):
        super().__init__()
        self.name = window_name
        self.map_list = map_list
        self.img_show_status = img_show_status
        self.zscale_settings = zscale_settings
        self.recogn_options_dict = recogn_options

        self.setWindowTitle(self.name)
        self.setWidget(QWidget(self))
        self.main_layout = QVBoxLayout()
        self.widget().setLayout(self.main_layout)
        # self.setFixedSize(600, 600)
        # self.setMinimumSize(200, 200)  # Минимальный размер окна
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Позволяет растягиваться

        self.img_width, self.img_height = 320, 320
        self.drons_btns = {}
        self.drons_freq = {}
        self.last_fps = deque(maxlen=10)
        self.last_time = 0

        self.recogn_options = RecognitionOptions(name=self.name,
                                                 zscale_settings=zscale_settings,
                                                 current_recogn_settings=self.recogn_options_dict)
        self.accum_size = self.recogn_options.spb_accum_size.value()
        self.recogn_options.spb_accum_size.valueChanged.connect(self.change_accum_size)
        self.hist_deques = {name: deque(maxlen=self.accum_size) for name in self.map_list}
        self.create_widgets()
        self.add_widgets_to_layout()

    @pyqtSlot(int)
    def change_accum_size(self, val: int):
        self.accum_size = val
        self.hist_deques = {name: deque(maxlen=self.accum_size) for name in self.map_list}
        label_style = {"color": "#EEE", "font-size": "10pt"}
        self.histogram_plot.setLabel("left", f"% of {self.accum_size} accumulation", **label_style)

    def context_menu(self):
        self.menu = QMenu(self)
        self.act_options = QAction('Options')
        self.act_options.triggered.connect(self.recogn_options.show)
        self.menu.addAction(self.act_options)
        self.menu.exec(QCursor.pos())

    def create_widgets(self):
        if self.img_show_status:
            self.tab = QTabWidget()

            self.create_spectrogram_tab()
            self.create_histogram_tab()

        for name in self.map_list:
            drone_btn = QPushButton(name)
            self.drons_btns[name] = drone_btn
            drone_freq = QLabel('None')
            drone_freq.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.drons_freq[name] = drone_freq

    def create_spectrogram_tab(self):
        self.img_frame = QLabel()
        pixmap = QPixmap('assets/spectrum_img.png')
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

        spectrogram_widget = QWidget()
        spectrogram_widget.setLayout(img_frame_layout)
        self.tab.addTab(spectrogram_widget, 'Spectrogram')

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
        self.tab.addTab(self.histogram_plot, 'Histogram')

    def add_widgets_to_layout(self):
        if self.img_show_status:
            self.main_layout.addWidget(self.tab)

        drons_btns_layout = QHBoxLayout()
        for drone_name in self.drons_btns.keys():
            drone_layout = QVBoxLayout()
            drone_layout.addWidget(self.drons_btns[drone_name])
            drone_layout.addWidget(self.drons_freq[drone_name])
            drons_btns_layout.addLayout(drone_layout)
        self.main_layout.addLayout(drons_btns_layout)

    @pyqtSlot(int)
    def show_frequencies(self, state: int):
        if bool(state):
            for dron_freq in self.drons_freq.values():
                dron_freq.show()
        else:
            for dron_freq in self.drons_freq.values():
                dron_freq.hide()

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

        picture = picture.scaled(self.img_width,  self.img_height)

        painter = QPainter(picture)
        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont('Courier', 12, QFont.Weight.Light))
        painter.drawText(5, 20, fps)
        painter.end()
        return QPixmap.fromImage(picture)

    @pyqtSlot(dict)
    def update_state(self, info: dict):
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
        self.update_histogram_plot()

        if 'detected_img' in info:
            self.last_fps.append(1 / (time.time() - self.last_time))
            current_fps = f'FPS: {sum(self.last_fps)/ len(self.last_fps):.1f}'
            self.last_time = time.time()

            img_arr = np.frombuffer(info['detected_img'], dtype=np.uint8).reshape((640, 640, 3))
            img_pixmap = self.convert_cv_qt(img_arr, fps=current_fps)
            self.img_frame.setPixmap(img_pixmap)

    def update_histogram_plot(self):
        accumed_val = [sum(deq)/self.accum_size*100 for deq in self.hist_deques.values()]
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
