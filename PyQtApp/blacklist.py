import sys
from PyQt6.QtWidgets import QMainWindow, QApplication, QTabWidget, QToolBar, QPushButton, QMenu, QScrollArea, QWidget
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt
import qdarktheme
from gRPC_thread import gRPCThread, connect_to_gRPC_server, gRPCServerErrorThread
from connection_window import ConnectWindow
from recognition_widget import RecognitionWidget
from settings import SettingsWidget
from processing import Processor
import yaml
from loguru import logger

try:
    logger.remove(0)
except:
    logger.remove(1)

log_level = "TRACE"
log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |"
              " <level>{level: <8}</level> |"
              " <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
logger.add(sys.stderr, format=log_format, colorize=True, backtrace=True, diagnose=True, enqueue=True)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.logger_ = logger

        # central_widget = QWidget(self)
        # self.setCentralWidget(central_widget)

        self.setWindowTitle('NN Recognition')
        self.config = self.load_config()
        self.map_list = list(self.config['map_list'])
        self.detetcted_img_status = bool(self.config['show_images'])
        self.clear_img_status = False

        self.recogn_widgets = {}
        self.recogn_settings_widgets = {}
        gRPC_channel = connect_to_gRPC_server(ip=self.config['server_addr'], port=self.config['server_port'])
        self.gRPCThread = gRPCThread(channel=gRPC_channel,
                                     map_list=self.map_list,
                                     detected_img_status=self.detetcted_img_status,
                                     clear_img_status=self.clear_img_status,
                                     logger_=self.logger_)
        self.gRPCErrorTread = gRPCServerErrorThread(gRPC_channel, self.logger_)
        self.available_channels = self.gRPCThread.getAvailableChannelsRequest()

        self.connectWindow = ConnectWindow(ip=self.config['server_addr'], available_channels=self.available_channels)
        self.connectWindow.finished.connect(self.connect_window_closed)

        self.create_actions()

        self.processor = Processor(logger_=self.logger_)

        self.link_events()

        self.adjustSize()


        # self.scroll = QScrollArea()
        # self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # self.scroll.setWidgetResizable(True)
        # self.scroll.setWidget(self)

    def connect_window_closed(self):
        self.enabled_channels = self.connectWindow.enabled_channels
        # self.init_recogn_settings()
        self.create_menu()
        self.create_toolbar()
        self.settingsWidget = SettingsWidget(enabled_channels=self.enabled_channels, map_list=self.map_list)
        self.settingsWidget.mainTab.chb_accumulation.stateChanged.connect(self.gRPCThread.onOffAccumulationRequest)
        self.settingsWidget.mainTab.btn_save_client_config.clicked.connect(self.save_config)
        self.settingsWidget.mainTab.btn_save_server_config.clicked.connect(lambda: self.gRPCThread.saveConfigRequest('kgbradar'))
        self.settingsWidget.saveTab.btn_save.clicked.connect(self.change_save_flag)
        self.init_recognition_widgets()

        self.settingsWidget.mainTab.cb_spectrogram_resolution.currentTextChanged.connect(lambda a:
                     self.set_spectrogram_resolution(self.settingsWidget.mainTab.cb_spectrogram_resolution.currentData()))

        self.show()
        self.move_window_to_center()

    def set_spectrogram_resolution(self, new_resolution: tuple[int, int]):
        for recogn_widget in self.recogn_widgets.values():
            recogn_widget.resize_img(new_resolution)

    def create_menu(self):
        self.act_settings = QAction('Settings', self)
        self.act_settings.setIcon(QIcon('assets/icons/btn_settings.png'))
        self.act_settings.triggered.connect(self.open_settings)

    def create_actions(self):
        self.act_start = QAction()
        self.act_start.setIcon(QIcon('assets/icons/btn_start.png'))
        self.act_start.setText('Start')
        self.act_start.setCheckable(True)
        self.act_start.triggered.connect(self.change_connection_state)

    def create_toolbar(self):
        self.toolBar = QToolBar()
        self.toolBar.addAction(self.act_start)
        self.toolBar.addAction(self.act_settings)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolBar)

    def init_recognition_widgets(self):
        current_zscale_settings_dict = self.gRPCThread.getCurrentZScaleRequest()
        recogn_settings = self.gRPCThread.gerCurrentRecognitionSettings()
        for channel in self.enabled_channels:
            recogn_widget = RecognitionWidget(window_name=channel,
                                              map_list=self.map_list,
                                              img_show_status=self.detetcted_img_status,
                                              zscale_settings=current_zscale_settings_dict[channel],
                                              recogn_options=recogn_settings[channel],
                                              show_images=self.config['show_images'],
                                              show_histogram=self.config['show_histogram'],
                                              show_spectrum=self.config['show_spectrum'])
            recogn_widget.recogn_options.signal_zscale_changed.connect(self.gRPCThread.changeZScaleRequest)
            recogn_widget.recogn_options.signal_recogn_settings.connect(self.gRPCThread.sendRecognitionSettings)
            self.settingsWidget.mainTab.chb_show_zscale.stateChanged.connect(recogn_widget.recogn_options.show_zscale_settings)
            self.settingsWidget.mainTab.chb_show_frequencies.stateChanged.connect(recogn_widget.show_frequencies)
            self.recogn_widgets[channel] = recogn_widget
        self.add_recogn_widgets(type_of_adding='default')
        self.processor.init_recogn_widgets(recogn_widgets=self.recogn_widgets)

    def add_recogn_widgets(self, type_of_adding: str):
        if type_of_adding == 'default':
            i = 1
            for widget in self.recogn_widgets.values():
                if i % 2 == 1:
                    self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, widget)
                else:
                    self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, widget)
                i += 1
                widget.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        elif type_of_adding == 'tabs':
            i = 0
            for widget in self.recogn_widgets.values():
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, widget)

            # табуляция между соседними виджетами
            while i < len(self.recogn_widgets) - 1:
                try:
                    self.tabifyDockWidget(self.recogn_widgets[list(self.recogn_widgets.keys())[i]],
                                          self.recogn_widgets[list(self.recogn_widgets.keys())[i + 1]])
                    i += 2
                except: pass

    def change_connection_state(self, status: bool):
        if status:
            self.act_start.setIcon(QIcon('assets/icons/btn_stop.png'))
            for channel in self.enabled_channels:
                try:
                    self.gRPCThread.startChannelRequest(channel_name=channel)
                except:
                    self.logger_.warning(f'Error with starting gRPC channel {channel} or channel is already started.')
                self.gRPCThread.start()
        else:
            self.act_start.setIcon(QIcon('assets/icons/btn_start.png'))
            if self.gRPCThread.isRunning():
                self.gRPCThread.requestInterruption()
                self.logger_.info('gRPCThread is requested to stop.')
            else:
                self.logger_.warning('gRPCThread is not running.')

    def open_recognition_settings(self, channel):
        self.recogn_settings_widgets[channel].show()

    def open_settings(self):
        self.settingsWidget.show()

    def link_events(self):
        self.gRPCThread.signal_dataStream_response.connect(
            lambda info: self.processor.parsing_data(info,
                                                     self.settingsWidget.saveTab.chb_save_detected.isChecked(),
                                                     ))

    def change_save_flag(self):
        if self.clear_img_status:
            self.clear_img_status = False
            self.gRPCThread.clear_img_status = False
            self.change_connection_state(status=False)      # restart gRPC stream
            self.gRPCThread.msleep(500)
            self.change_connection_state(status=True)
            self.logger_.info('Saving images stopped!')
        else:
            self.clear_img_status = True
            self.gRPCThread.clear_img_status = True
            self.change_connection_state(status=False)      # restart gRPC stream
            self.gRPCThread.msleep(500)
            self.change_connection_state(status=True)
            self.logger_.info('Saving images started!')

    def save_config(self):
        pass

    def load_config(self):
        try:
            with open('client_conf.yaml', encoding='utf-8') as f:
                config = dict(yaml.load(f, Loader=yaml.SafeLoader))
                self.logger_.success(f'Config loaded successfully!')
                return config
        except Exception as e:
            self.logger_.error(f'Error with loading config: {e}')

    def move_window_to_center(self):
        # Получаем размеры экрана
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Получаем размеры окна
        window_width = self.width()
        window_height = self.height()

        # Вычисляем координаты для центра экрана
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        print(screen_width, screen_height, window_width, window_height, x, y)

        # Перемещаем окно в центр экрана
        self.move(x, y)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    main_window = MainWindow()
    main_window.connectWindow.show()
    sys.exit(app.exec())

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

            self.img_show_status = widgets_statuses.get('show_images', False)
            self.histogram_show_status = widgets_statuses.get('show_histogram', False)
            self.spectrum_show_status = widgets_statuses.get('show_spectrum', False)

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
            self.tab = QTabWidget()
            self.tab.hide()
            self.create_spectrogram_tab()
            self.create_histogram_tab()
            self.create_spectrum_tab()

            for name in self.map_list:
                drone_btn = QPushButton(name)
                self.drons_btns[name] = drone_btn
                drone_freq = QLabel('None')
                drone_freq.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.drons_freq[name] = drone_freq

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

            spectrogram_widget = QWidget()
            spectrogram_widget.setLayout(img_frame_layout)

            if self.img_show_status:
                self.tab.addTab(spectrogram_widget, 'Spectrogram')
            if not self.tab.isVisible():
                self.tab.show()

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

            if self.histogram_show_status:
                self.tab.addTab(self.histogram_plot, 'Histogram')
            if not self.tab.isVisible():
                self.tab.show()

        def create_spectrum_tab(self):
            self.spectrum_plot = pyqtgraph.PlotWidget()
            self.spectrum_curve = self.spectrum_plot.plot(pen=(229, 165, 10))

            if self.spectrum_show_status:
                self.tab.addTab(self.spectrum_plot, 'Spectrum')
            if not self.tab.isVisible():
                self.tab.show()

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

            picture = picture.scaled(self.img_width, self.img_height)

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




