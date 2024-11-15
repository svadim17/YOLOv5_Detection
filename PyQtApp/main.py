import sys
from PyQt6.QtWidgets import QMainWindow, QApplication, QTabWidget, QToolBar, QPushButton
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt
import qdarktheme
from connection_window import ConnectWindow
from recognition_widget import RecognitionWidget
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
        # self.setDockOptions(QMainWindow.DockOption.AllowTabbedDocks | QMainWindow.DockOption.AnimatedDocks)
        self.config = self.load_conf()
        self.create_actions()
        self.create_toolbar()

        self.connectWindow = ConnectWindow(ip=self.config['server_addr'],
                                           grpc_port=self.config['server_port'],
                                           map_list=list(self.config['map_list']))
        self.connectWindow.finished.connect(self.show)

        self.processor = Processor()

        self.recogn_widgets = {}
        self.link_events()

    def create_actions(self):
        self.act_start = QAction()
        self.act_start.setIcon(QIcon('assets/btn_start.png'))
        self.act_start.setText('Start')
        self.act_start.setCheckable(True)
        self.act_start.triggered.connect(self.change_connection_state)

    def create_toolbar(self):
        self.toolBar = QToolBar()
        self.toolBar.addAction(self.act_start)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.toolBar)

    def create_recognition_widgets(self, enabled_channels: list):
        for channel in enabled_channels:
            recogn_widget = RecognitionWidget(window_name=channel, map_list=['Autel', 'Fpv', 'Dji', 'WiFi'])
            self.recogn_widgets[channel] = recogn_widget
            # self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, recogn_widget)
            # self.tab.addTab(recogn_widget, channel)
        self.add_recogn_widgets()
        self.processor.init_recogn_widgets(recogn_widgets=self.recogn_widgets)

    def add_recogn_widgets(self):
        i = 1
        for widget in self.recogn_widgets.values():
            if i % 2 == 1:
                self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, widget)
            else:
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, widget)
            i += 1

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



