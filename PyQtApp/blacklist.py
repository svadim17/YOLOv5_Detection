import sys
from PyQt6.QtWidgets import (QMainWindow, QApplication, QTabWidget, QToolBar, QPushButton, QMenu, QScrollArea, QWidget,
                             QLabel, QVBoxLayout, QHBoxLayout, QToolButton)
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt, QSize, QEvent
import qdarktheme
from gRPC_thread import gRPCThread, connect_to_gRPC_server, gRPCServerErrorThread
from connection_window import ConnectWindow
from recognition_widget import RecognitionWidget
from recognition_options import RecognitionOptions
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

class CustomTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.initial_pos = None
        title_bar_layout = QHBoxLayout(self)
        title_bar_layout.setContentsMargins(1, 1, 1, 1)
        title_bar_layout.setSpacing(2)
        self.title = QLabel(f"{self.__class__.__name__}", self)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet(
            """
        QLabel { text-transform: uppercase; font-size: 10pt; margin-left: 48px; }
        """
        )

        if title := parent.windowTitle():
            self.title.setText(title)
        title_bar_layout.addWidget(self.title)
        # Min button
        self.min_button = QToolButton(self)
        min_icon = QIcon()
        min_icon.addFile("min.svg")
        self.min_button.setIcon(min_icon)
        self.min_button.clicked.connect(self.window().showMinimized)

        # Max button
        self.max_button = QToolButton(self)
        max_icon = QIcon()
        max_icon.addFile("max.svg")
        self.max_button.setIcon(max_icon)
        self.max_button.clicked.connect(self.window().showMaximized)

        # Close button
        self.close_button = QToolButton(self)
        close_icon = QIcon()
        close_icon.addFile("close.svg")  # Close has only a single state.
        self.close_button.setIcon(close_icon)
        self.close_button.clicked.connect(self.window().close)

        # Normal button
        self.normal_button = QToolButton(self)
        normal_icon = QIcon()
        normal_icon.addFile("normal.svg")
        self.normal_button.setIcon(normal_icon)
        self.normal_button.clicked.connect(self.window().showNormal)
        self.normal_button.setVisible(False)
        # Add buttons
        buttons = [
            self.min_button,
            self.normal_button,
            self.max_button,
            self.close_button,
        ]
        for button in buttons:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedSize(QSize(16, 16))
            button.setStyleSheet(
                """QToolButton {
                    border: none;
                    padding: 2px;
                }
                """
            )
            title_bar_layout.addWidget(button)

    def window_state_changed(self, state):
        if state == Qt.WindowState.WindowMaximized:
            self.normal_button.setVisible(True)
            self.max_button.setVisible(False)
        else:
            self.normal_button.setVisible(False)
            self.max_button.setVisible(True)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.logger_ = logger

        self.title_bar = CustomTitleBar(self)
        central_widget = QWidget(self)
        self.dock_area_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.centralWidget().setLayout(QVBoxLayout())
        self.centralWidget().layout().addWidget(self.title_bar)
        self.centralWidget().layout().addWidget(QLabel('12316565163546+58'))
        self.title_bar.setFixedHeight(30)

        self.setWindowTitle('NN Recognition')
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)


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

        # self.scroll = QScrollArea()
        # self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # self.scroll.setWidgetResizable(True)
        # self.scroll.setWidget(self)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            self.title_bar.window_state_changed(self.windowState())
        super().changeEvent(event)
        event.accept()

    def window_state_changed(self, state):
        self.normal_button.setVisible(state == Qt.WindowState.WindowMaximized)
        self.max_button.setVisible(state != Qt.WindowState.WindowMaximized)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.initial_pos = event.position().toPoint()
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.initial_pos is not None:
            delta = event.position().toPoint() - self.initial_pos
            self.window().move(
                self.window().x() + delta.x(),
                self.window().y() + delta.y(),
            )
        super().mouseMoveEvent(event)
        event.accept()

    def mouseReleaseEvent(self, event):
        self.initial_pos = None
        super().mouseReleaseEvent(event)
        event.accept()

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

    def set_spectrogram_resolution(self, new_resolution: tuple[int, int]):
        for recogn_widget in self.recogn_widgets.values():
            recogn_widget.resize_img(new_resolution)

    def create_menu(self):
        self.menu_bar = self.menuBar()

        self.recogn_menu = self.menu_bar.addMenu('Recognition')
        for channel in self.enabled_channels:
            act_channel = QAction(channel, self)
            act_channel.triggered.connect(lambda checked, name=channel: self.open_recognition_settings(name))
            self.recogn_menu.addAction(act_channel)

        self.act_settings = QAction('Settings', self)
        self.act_settings.triggered.connect(self.open_settings)
        self.menu_bar.addAction(self.act_settings)

    def create_actions(self):
        self.act_start = QAction()
        self.act_start.setIcon(QIcon('assets/btn_start.png'))
        self.act_start.setText('Start')
        self.act_start.setCheckable(True)
        self.act_start.triggered.connect(self.change_connection_state)

    def create_toolbar(self):
        self.toolBar = QToolBar()
        self.toolBar.addAction(self.act_start)
        self.toolBar.addWidget(self.menu_bar)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolBar)

    # def init_recogn_settings(self):
    #     # recogn_settings = self.gRPCThread.gerCurrentRecognitionSettings()
    #     for channel in self.enabled_channels:
    #         self.recogn_settings_widgets[channel] = RecognitionSettingsPage(channel, recogn_settings[channel])
    #         self.recogn_settings_widgets[channel].signal_recogn_settings.connect(self.gRPCThread.sendRecognitionSettings)

    def init_recognition_widgets(self):
        current_zscale_settings_dict = self.gRPCThread.getCurrentZScaleRequest()
        recogn_settings = self.gRPCThread.gerCurrentRecognitionSettings()
        for channel in self.enabled_channels:
            recogn_widget = RecognitionWidget(window_name=channel,
                                              map_list=self.map_list,
                                              img_show_status=self.detetcted_img_status,
                                              zscale_settings=current_zscale_settings_dict[channel],
                                              recogn_options=recogn_settings[channel])
            recogn_widget.signal_zscale_changed.connect(self.gRPCThread.changeZScaleRequest)
            recogn_widget.recogn_options.signal_recogn_settings.connect(self.gRPCThread.sendRecognitionSettings)
            self.settingsWidget.mainTab.chb_show_zscale.stateChanged.connect(recogn_widget.show_zscale_settings)
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
            self.act_start.setIcon(QIcon('assets/btn_stop.png'))
            for channel in self.enabled_channels:
                try:
                    self.gRPCThread.startChannelRequest(channel_name=channel)
                except:
                    self.logger_.warning(f'Error with starting gRPC channel {channel} or channel is already started.')
                self.gRPCThread.start()
        else:
            self.act_start.setIcon(QIcon('assets/btn_start.png'))
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    main_window = MainWindow()
    main_window.connectWindow.show()
    sys.exit(app.exec())



