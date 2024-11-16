import sys
from PyQt6.QtWidgets import QMainWindow, QApplication, QTabWidget, QToolBar, QPushButton, QMenu
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt
import qdarktheme
from gRPC_thread import gRPCThread
from connection_window import ConnectWindow
from recognition_widget import RecognitionWidget
from popout_menu import PopOutMenu
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
              " {extra} |"
              " <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
logger.add(sys.stderr, format=log_format, colorize=True, backtrace=True, diagnose=True, enqueue=True)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('NN Recognition')
        self.config = self.load_conf()
        self.map_list = list(self.config['map_list'])
        self.create_actions()
        self.create_toolbar()

        self.gRPCThread = gRPCThread(map_list=self.map_list, img_status=True)

        self.connectWindow = ConnectWindow(ip=self.config['server_addr'],
                                           grpc_port=self.config['server_port'],
                                           grpc_thread=self.gRPCThread)
        self.connectWindow.finished.connect(self.show)      # открытие основного окна после закрытия ConnectWindow

        self.processor = Processor()

        self.popoutMenu = PopOutMenu(enabled_channels=self.connectWindow.enabled_channels)
        self.popoutMenu.setFloating(True)
        self.popoutMenu.hide()

        self.recogn_widgets = {}
        self.link_events()

    def create_actions(self):
        self.act_start = QAction()
        self.act_start.setIcon(QIcon('assets/btn_start.png'))
        self.act_start.setText('Start')
        self.act_start.setCheckable(True)
        self.act_start.triggered.connect(self.change_connection_state)

        self.act_popout_menu = QAction()
        self.act_popout_menu.setText('Menu')
        self.act_popout_menu.setCheckable(True)
        self.act_popout_menu.triggered.connect(self.toggle_popout_menu)

    def create_toolbar(self):
        self.toolBar = QToolBar()
        self.toolBar.addAction(self.act_start)
        self.toolBar.addAction(self.act_popout_menu)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolBar)

    def create_recognition_widgets(self, enabled_channels: list):
        current_zscale_settings_dict = self.gRPCThread.getCurrentZScaleRequest()
        for channel in enabled_channels:
            recogn_widget = RecognitionWidget(window_name=channel,
                                              map_list=self.map_list,
                                              zscale_settings=current_zscale_settings_dict[channel])
            recogn_widget.signal_zscale_changed.connect(self.gRPCThread.changeZScaleRequest)
            self.recogn_widgets[channel] = recogn_widget
        self.add_recogn_widgets(type_of_adding='default')
        self.processor.init_recogn_widgets(recogn_widgets=self.recogn_widgets)

    def add_recogn_widgets(self, type_of_adding: str):
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.popoutMenu)

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

    def toggle_popout_menu(self):
        if self.popoutMenu.isVisible():
            self.popoutMenu.hide()
            self.add_recogn_widgets(type_of_adding='default')
        else:
            self.popoutMenu.show()
            self.add_recogn_widgets(type_of_adding='tabs')

    def change_connection_state(self, status: bool):
        if status:
            self.act_start.setIcon(QIcon('assets/btn_stop.png'))
            for channel in self.connectWindow.enabled_channels:
                try:
                    self.connectWindow.gRPCThread.startChannelRequest(channel_name=channel)
                except:
                    logger.warning(f'Error with starting gRPC channel {channel} or channel is already started.')
                self.connectWindow.gRPCThread.start()
        else:
            self.act_start.setIcon(QIcon('assets/btn_start.png'))
            if self.connectWindow.gRPCThread.isRunning():
                self.connectWindow.gRPCThread.requestInterruption()
                logger.info('gRPCThread is requested to stop.')
            else:
                logger.warning('gRPCThread is not running.')

    def link_events(self):
        self.connectWindow.signal_init_connections.connect(lambda channels: self.create_recognition_widgets(channels))
        self.connectWindow.gRPCThread.signal_dataStream_response.connect(self.processor.parsing_data)

    def load_conf(self):
        try:
            with open('config.yaml', encoding='utf-8') as f:
                config = dict(yaml.load(f, Loader=yaml.SafeLoader))
                logger.success(f'Config loaded successfully!')
                return config
        except Exception as e:
            logger.error(f'Error with loading config: {e}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    main_window = MainWindow()
    main_window.connectWindow.show()
    sys.exit(app.exec())



