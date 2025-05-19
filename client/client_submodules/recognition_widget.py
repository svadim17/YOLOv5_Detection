from PySide6.QtWidgets import (QWidget, QDockWidget, QApplication, QTabWidget, QSlider, QSpinBox, QCheckBox,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QSizePolicy, QMenu,
                             QTreeWidget, QTreeWidgetItem)
from PySide6.QtGui import QPixmap, QImage, QPainter, QFont, QCursor, QAction, QIcon
from PySide6.QtCore import Slot, Qt
import numpy as np
import os
import cv2
import qdarktheme
from collections import deque
import time
import pyqtgraph
from .recognition_options import RecognitionOptions
from .process_options import ProcessOptions
from .gRPC_thread import ChannelInfo


class RecognitionWidget(QDockWidget, QWidget):

    def __init__(self, window_name,
                 map_list: list,
                 zscale_settings: list,
                 recogn_options: dict,
                 signal_settings: dict,
                 show_recogn_options: bool,
                 show_freq: bool,
                 channel_info: ChannelInfo,
                 theme_type: str,
                 **widgets_statuses):
        super().__init__()
        self.name = window_name
        self.map_list = map_list
        self.zscale_settings = zscale_settings
        self.recogn_options_dict = recogn_options
        self.signal_settings = signal_settings
        self.show_recogn_options = show_recogn_options
        self.channel_info = channel_info
        self.theme_type = theme_type

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
        self.freq_labels = {}
        self.drone_btns_all_freq = {}

        self.last_fps = deque(maxlen=5)
        self.last_fps_2 = deque(maxlen=5)
        self.last_time = 0
        self.last_time_2 = 0

        self.recognOptions = RecognitionOptions(name=self.name,
                                                zscale_settings=zscale_settings,
                                                current_recogn_settings=self.recogn_options_dict,
                                                channel_info=self.channel_info)
        self.accum_size = self.recognOptions.spb_accum_size.value()
        self.recognOptions.spb_accum_size.valueChanged.connect(self.change_accum_size)

        self.processOptions = ProcessOptions(name=self.name)

        self.hist_deques = {name: deque(maxlen=self.accum_size) for name in self.map_list}
        self.create_widgets()

        self.show_frequencies(state=int(show_freq))

    @Slot(int)
    def change_accum_size(self, val: int):
        self.accum_size = val
        self.hist_deques = {name: deque(maxlen=self.accum_size) for name in self.map_list}
        label_style = {"color": "#EEE", "font-size": "10pt"}
        self.histogram_plot.setLabel("left", f"% of {self.accum_size} accumulation", **label_style)

    def create_context_menu(self):
        self.menu = QMenu(self)

        self.act_options = QAction('Recognition options')
        self.act_options.triggered.connect(self.recognOptions.show)

        self.act_process_status = QAction('Process options')
        self.act_process_status.triggered.connect(self.processOptions.show)

        self.menu.addAction(self.act_process_status)
        self.add_remove_recogn_options(status=int(self.show_recogn_options))

    def context_menu_requested(self):
        self.menu.exec(QCursor.pos())

    def add_remove_recogn_options(self, status):
        if status:
            self.menu.addAction(self.act_options)
        else:
            self.menu.removeAction(self.act_options)

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
        self.create_context_menu()

    def create_buttons_tree(self):
        self.freq_tree = QTreeWidget()

        self.freq_tree.setColumnCount(len(self.map_list) + 1)      # +1 for freq column
        # self.freq_tree.setHeaderLabels(['Freq'] + self.map_list)
        self.freq_tree.setHeaderHidden(True)
        # self.freq_tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.freq_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.freq_tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.freq_tree.setColumnWidth(0, 200)  # Фиксированная ширина для колонки с частотами
        for col in range(1, len(self.map_list) + 1):
            self.freq_tree.setColumnWidth(col, 80)

        all_freq_tree_item = QTreeWidgetItem(self.freq_tree)
        self.btn_all_freq = QPushButton('All freq')
        self.btn_all_freq.clicked.connect(lambda checked, item=all_freq_tree_item: self.toggle_frequencies(item))
        self.freq_tree.setItemWidget(all_freq_tree_item, 0, self.btn_all_freq)

        for col, drone in enumerate(self.map_list, 1):
            drone_btn = QPushButton(drone)
            drone_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.drone_btns_all_freq[drone] = drone_btn
            self.freq_tree.setItemWidget(all_freq_tree_item, col, drone_btn)

        tree_layout = QHBoxLayout()
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.addWidget(self.freq_tree, alignment=Qt.AlignmentFlag.AlignCenter)

        self.main_layout.addLayout(tree_layout)
        # self.adjust_tree_size()

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
        # self.adjust_tree_size()

    def adjust_tree_size(self):
        """Подгоняем размер дерева под содержимое"""
        # Подсчитываем количество видимых строк
        visible_rows = 1  # Начальная строка "All freq"
        root = self.freq_tree.topLevelItem(0)
        if root and root.isExpanded():
            visible_rows += root.childCount()

        # Рассчитываем высоту на основе видимых строк
        row_height = self.freq_tree.sizeHintForRow(0)  # Высота одной строки
        total_height = row_height * visible_rows

        # Рассчитываем общую ширину на основе фиксированных колонок
        total_width = 200 + 80 * len(self.map_list)  # 200 для первой колонки + 80 для каждой колонки дронов

        # Устанавливаем размеры дерева
        self.freq_tree.setMinimumHeight(total_height)
        self.freq_tree.setMaximumHeight(total_height)
        self.freq_tree.setMinimumWidth(total_width)
        self.freq_tree.setMaximumWidth(total_width)

        # Устанавливаем минимальный размер виджета
        self.setMinimumWidth(total_width)
        self.setMinimumHeight(total_height + self.tab.height() if self.tab.isVisible() else total_height)

    def create_buttons(self):
        self.btn_all_freq = QPushButton(icon=QIcon(f'./assets/icons/{self.theme_type}/arrow_right.png'))
        self.btn_all_freq.setCheckable(True)
        self.btn_all_freq.clicked.connect(self.btn_all_freq_clicked)

        for name in self.map_list:
            drone_btn = QPushButton(name)
            self.drone_btns_all_freq[name] = drone_btn

        self.all_freq_layout = QGridLayout()
        self.all_freq_layout.addWidget(self.btn_all_freq)
        colomn = 1
        for btn in self.drone_btns_all_freq.values():
            self.all_freq_layout.addWidget(btn, 0, colomn)
            colomn += 1
        self.main_layout.addLayout(self.all_freq_layout)

        self.all_drons_btns_widget = QWidget()
        all_drons_btns_layout = QVBoxLayout()
        self.all_drons_btns_widget.setLayout(all_drons_btns_layout)
        for freq in self.channel_info.central_freq:
            drons_btns = {}
            drons_freq = {}
            for name in self.map_list:
                drone_btn = QPushButton(name)
                drons_btns[name] = drone_btn
                drone_freq = QLabel('None')
                drone_freq.setAlignment(Qt.AlignmentFlag.AlignCenter)
                drons_freq[name] = drone_freq

            self.drons_btns_layout = QGridLayout()
            freq_label = QLabel(f'{freq/1000_000_000:.2f}GHz')
            freq_label.setStyleSheet('transform: rotate(90deg);')
            self.drons_btns_layout.addWidget(freq_label)
            colomn = 1
            for drone_name in drons_btns.keys():
                drone_layout = QVBoxLayout()
                drone_layout.addWidget(drons_btns[drone_name])
                drone_layout.addWidget(drons_freq[drone_name])
                self.drons_btns_layout.addLayout(drone_layout, 0, colomn)
                colomn += 1

            self.drons_btns[freq] = drons_btns
            self.drons_freq[freq] = drons_freq
            self.freq_labels[freq] = freq_label
            all_drons_btns_layout.addLayout(self.drons_btns_layout)

        self.main_layout.addWidget(self.all_drons_btns_widget)
        self.all_drons_btns_widget.hide()

    def btn_all_freq_clicked(self):
        if not self.btn_all_freq.isChecked():
            self.btn_all_freq.setIcon(QIcon(f'./assets/icons/{self.theme_type}/arrow_right.png'))
            self.all_drons_btns_widget.hide()
        else:
            self.btn_all_freq.setIcon(QIcon(f'./assets/icons/{self.theme_type}/arrow_down.png'))
            self.all_drons_btns_widget.show()


    def create_buttons_old(self):
        for freq in self.channel_info.central_freq:
            drons_btns = {}
            drons_freq = {}
            for name in self.map_list:
                drone_btn = QPushButton(name)
                drons_btns[name] = drone_btn
                drone_freq = QLabel('None')
                drone_freq.setAlignment(Qt.AlignmentFlag.AlignCenter)
                drons_freq[name] = drone_freq

            drons_btns_layout = QGridLayout()
            freq_label = QLabel(f'{freq/1000_000_000:.2f}GHz')
            freq_label.setStyleSheet('transform: rotate(90deg);')
            drons_btns_layout.addWidget(freq_label)
            colomn = 1
            for drone_name in drons_btns.keys():
                drone_layout = QVBoxLayout()
                drone_layout.addWidget(drons_btns[drone_name])
                drone_layout.addWidget(drons_freq[drone_name])
                drons_btns_layout.addLayout(drone_layout, 0, colomn)
                colomn += 1

            self.drons_btns[freq] = drons_btns
            self.drons_freq[freq] = drons_freq
            self.freq_labels[freq] = freq_label

            self.main_layout.addLayout(drons_btns_layout)

    def create_spectrogram_tab(self):
        self.img_frame = QLabel()
        pixmap = QPixmap('../assets/icons/spectrum_img.png')
        self.img_frame.setPixmap(pixmap.scaled(self.img_width, self.img_height))
        self.img_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Центрирование изображения
        # self.img_frame.setScaledContents(True)  # масштабирование изображения по размеру виджета
        self.img_frame.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.img_frame.customContextMenuRequested.connect(self.context_menu_requested)

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
        self.spectrum_plot.setXRange(0, self.signal_settings['width'], padding=0.03)
        if self.show_spectrum_status:
            self.tab.addTab(self.spectrum_plot, 'Spectrum')
        else:
            self.spectrum_plot.hide()

    @Slot(int)
    def show_frequencies(self, state: int):
        for dron_freq in self.drons_freq.values():
            for widget_ in dron_freq.values():
                if bool(state):
                    widget_.show()
                else:
                    widget_.hide()

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

        # self.setMaximumWidth(self.img_width + 40)
        # self.setMaximumHeight(self.img_height + 150)

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

    @Slot(dict)
    def update_state(self, info: dict):
        band_name = info['band_name']
        channel_freq = info['channel_freq']
        freq_ghz = channel_freq / 1_000_000_000

        for freq, label in self.freq_labels.items():
            if label is not None:
                if channel_freq == freq:
                    label.setStyleSheet("color: red")
                else:
                    label.setStyleSheet("")

        # self.freq_labels[channel_freq].setStyleSheet("color: red")
        self.last_fps_2.append(1 / (time.time() - self.last_time_2))
        current_fps = f'FPS: {sum(self.last_fps_2) / len(self.last_fps_2):.1f}'
        self.last_time_2 = time.time()
        self.setWindowTitle(f'{self.name}  |  Fc = {freq_ghz:.4f} GHz  |  {current_fps}')

        for drone_dict in info['drones']:
            if (channel_freq in self.drons_btns and
                    channel_freq in self.drons_freq and
                    drone_dict['name'] in self.drons_btns[channel_freq] and
                    drone_dict['name'] in self.drons_freq[channel_freq]):
                if drone_dict['state']:
                    self.drons_btns[channel_freq][drone_dict['name']].setStyleSheet("background-color: #F0483C; "
                                                                      "font: bold; "
                                                                      "color: #FFFFFF")
                    self.drons_freq[channel_freq][drone_dict['name']].setText(str(drone_dict['freq']))
                else:
                    self.drons_btns[channel_freq][drone_dict['name']].setStyleSheet("background-color: ")
                    self.drons_freq[channel_freq][drone_dict['name']].setText('None')
                self.hist_deques[drone_dict['name']].appendleft(int(drone_dict['state']))

            if drone_dict['state']:
                self.drone_btns_all_freq[drone_dict['name']].setStyleSheet("background-color: #F0483C; "
                                                                           "font: bold; "
                                                                           "color: #FFFFFF")
            else:
                self.drone_btns_all_freq[drone_dict['name']].setStyleSheet("background-color: ")

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

    def update_channel_freq(self, new_freq):
        freq_ghz = new_freq / 1_000_000_000
        self.setWindowTitle(f'{self.name} | Fc = {freq_ghz:.4f} GHz')

    def theme_changed(self, theme: str):
        self.theme_type = theme
        self.btn_all_freq.setIcon(QIcon(f'./assets/icons/{self.theme_type}/arrow_right.png'))
        self.all_drons_btns_widget.hide()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    main_window = RecognitionWidget(window_name='Detectors',
                                    map_list=['1A', '2B', '3C', '4D'],
                                    img_show_status=True,
                                    zscale_settings=[-20, 20],
                                    recogn_options={'accum_size': 10, 'threshold': 0.5, 'exceedance': 0.7},
                                    signal_settings={'width': 1024, 'height': 2048, 'fs': 12880000},
                                    show_recogn_options=True,
                                    channel_info=ChannelInfo(name='afasd',
                                                             hardware_type='Alinx',
                                                             central_freq=[5_000_000_000, 6_000_000_000]),
                                    show_images=False,
                                    show_histogram=False,
                                    show_spectrum=False
                                    )

    main_window.show()
    sys.exit(app.exec())
