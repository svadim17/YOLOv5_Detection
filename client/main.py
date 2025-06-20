import os
import sys
from PyQt6.QtWidgets import QMainWindow, QApplication, QToolBar
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt
import qdarktheme
from client_submodules.gRPC_thread import gRPCThread, connect_to_gRPC_server, gRPCServerErrorThread
from client_submodules.welcome_window import WelcomeWindow
from client_submodules.connection_window import ConnectWindow
from client_submodules.recognition_widget import RecognitionWidget
from client_submodules.settings import SettingsWidget
from client_submodules.processing import Processor
from client_submodules.sound_thread import SoundThread
from client_submodules.telemetry import TelemetryWidget


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

        self.setWindowTitle('NN Recognition v25.18')
        self.setWindowIcon(QIcon('./assets/icons/nn1.ico'))
        self.config = self.load_config()
        self.server_ip = list(self.config['server_addr'])
        self.grpc_port = self.config['server_port']
        self.map_list = list(self.config['map_list'])
        self.show_img_status = bool(self.config['settings_main']['show_spectrogram'])
        self.show_histogram_status = bool(self.config['settings_main']['show_histogram'])
        self.show_spectrum_status = bool(self.config['settings_main']['show_spectrum'])
        self.watchdog = bool(self.config['settings_main']['watchdog'])
        self.welcome_window_state = bool(self.config['show_welcome_window'])
        self.clear_img_status = False

        self.theme_type = self.config['settings_main']['theme']['type']
        with open('app_themes.yaml', 'r', encoding='utf-8') as f:           # load available themes
            self.themes = yaml.safe_load(f)
        qdarktheme.setup_theme(theme=self.theme_type,
                               custom_colors=self.themes[self.config['settings_main']['theme']['name']],
                               additional_qss="QToolTip { "
                                              "background-color: #ffff99;"
                                              "color: #000000;"
                                              "border: 1px solid #000000;"
                                              "padding: 2px;}")

        self.recogn_widgets = {}
        self.recogn_settings_widgets = {}
        self.sound_states = {}
        self.sound_classes_states = {}
        for name in self.map_list:
            self.sound_classes_states[name] = self.config['settings_sound']['classes_sound'][name]

        if self.welcome_window_state:
            self.welcomeWindow = WelcomeWindow(server_addr=self.server_ip, server_port=self.grpc_port)
            self.welcomeWindow.signal_connect_to_server.connect(self.connect_to_server)
            self.welcomeWindow.finished.connect(self.welcome_window_closed)
            self.welcomeWindow.show()
        else:
            self.connect_to_server(server_ip=self.server_ip[0], grpc_port=self.grpc_port)

    def connect_to_server(self, server_ip: str, grpc_port: str):
        self.server_ip = server_ip
        try:
            self.gRPC_channel = connect_to_gRPC_server(ip=server_ip, port=grpc_port)
            self.logger_.success(f'Successfully connected to {server_ip}:{grpc_port}!')
            if self.welcome_window_state:
                self.welcomeWindow.close()
            else:
                self.welcome_window_closed()
        except Exception as e:
            self.logger_.critical(f'Error with connecting to {server_ip}:{grpc_port}! \n{e}')

    def welcome_window_closed(self):
        self.gRPCThread = gRPCThread(channel=self.gRPC_channel,
                                     map_list=self.map_list,
                                     detected_img_status=self.show_img_status,
                                     clear_img_status=self.clear_img_status,
                                     spectrum_status=self.show_spectrum_status,
                                     watchdog=self.watchdog,
                                     logger_=self.logger_)
        self.gRPCErrorTread = gRPCServerErrorThread(self.gRPC_channel, self.logger_)
        self.available_channels, self.channels_info = self.gRPCThread.getAvailableChannelsRequest()
        self.connectWindow = ConnectWindow(ip=self.config['server_addr'],
                                           available_channels=self.available_channels,
                                           channels_info=self.channels_info)
        self.connectWindow.show()
        self.connectWindow.finished.connect(self.connect_window_closed)

        self.create_actions()

        self.processor = Processor(logger_=self.logger_)

        self.link_events()
        self.adjustSize()

    def connect_window_closed(self):
        self.enabled_channels = self.connectWindow.enabled_channels
        self.enabled_channels_info = self.connectWindow.enabled_channels_info

        self.gRPCThread.init_enabled_channels(enabled_channels=self.enabled_channels)
        self.signal_settings = self.gRPCThread.signalSettings(enabled_channels=self.enabled_channels)
        for channel in self.enabled_channels:
            self.sound_states[channel] = self.config['settings_sound']['sound_status']

        self.create_menu()
        self.create_toolbar()
        self.settingsWidget = SettingsWidget(enabled_channels=self.enabled_channels,
                                             config=self.config,
                                             enabled_channels_info=self.enabled_channels_info,
                                             logger_=self.logger_)
        self.settingsWidget.mainTab.chb_accumulation.stateChanged.connect(self.gRPCThread.onOffAccumulationRequest)
        self.settingsWidget.mainTab.chb_watchdog.stateChanged.connect(self.gRPCThread.change_watchdog_status)
        self.settingsWidget.mainTab.chb_spectrogram_show.stateChanged.connect(self.change_img_status)
        self.settingsWidget.mainTab.chb_histogram_show.stateChanged.connect(self.change_histogram_status)
        self.settingsWidget.mainTab.chb_spectrum_show.stateChanged.connect(self.change_spectrum_status)
        self.settingsWidget.mainTab.cb_themes.currentIndexChanged.connect(self.apply_theme)
        self.settingsWidget.mainTab.cb_theme_type.currentIndexChanged.connect(self.apply_theme)
        self.settingsWidget.btn_save_client_config.clicked.connect(self.save_config)
        self.settingsWidget.btn_save_server_config.clicked.connect(lambda: self.gRPCThread.saveConfigRequest('kgbradar'))
        self.settingsWidget.saveTab.btn_save.clicked.connect(self.change_save_status)
        self.soundThread = SoundThread(sound_name=self.settingsWidget.soundTab.cb_sound.currentText(), logger_=self.logger_)
        self.settingsWidget.soundTab.cb_sound.currentTextChanged.connect(self.soundThread.sound_file_changed)
        self.settingsWidget.soundTab.btn_play_sound.clicked.connect(self.soundThread.play_sound)
        self.settingsWidget.soundTab.signal_sound_states.connect(self.processor.init_sound_states)
        self.settingsWidget.soundTab.signal_sound_classes_states.connect(self.processor.init_sound_classes_states)
        self.settingsWidget.alinxTab.btn_get_soft_ver.clicked.connect(self.gRPCThread.getAlinxSoftVer)
        self.settingsWidget.alinxTab.btn_get_load_detect.clicked.connect(self.gRPCThread.getLoadDetectState)
        self.settingsWidget.alinxTab.spb_gain_24.valueChanged.connect(
            lambda: self.gRPCThread.setGain(channel_name='2G4', gain=self.settingsWidget.alinxTab.spb_gain_24.value()))
        self.settingsWidget.alinxTab.spb_gain_58.valueChanged.connect(
            lambda: self.gRPCThread.setGain(channel_name='5G8', gain=self.settingsWidget.alinxTab.spb_gain_58.value()))
        self.settingsWidget.alinxTab.signal_autoscan_state.connect(self.gRPCThread.setAutoscanState)
        self.settingsWidget.alinxTab.signal_set_central_freq.connect(self.gRPCThread.setFrequency)
        self.settingsWidget.nnTab.btn_get_nn_info.clicked.connect(lambda: self.gRPCThread.nnInfo(self.enabled_channels))
        self.settingsWidget.usrpTab.signal_central_freq_changed.connect(self.gRPCThread.setUSRPFrequency)
        self.settingsWidget.usrpTab.signal_autoscan_state.connect(self.gRPCThread.setAutoscanState)
        self.gRPCThread.signal_alinx_soft_ver.connect(self.settingsWidget.alinxTab.update_soft_ver)
        self.gRPCThread.signal_alinx_load_detect_state.connect(self.settingsWidget.alinxTab.update_load_detect_state)
        self.gRPCThread.signal_nn_info.connect(self.settingsWidget.nnTab.update_models_info)
        self.init_recognition_widgets()
        self.processor.init_sound_states(sound_states=self.sound_states)
        self.processor.init_sound_classes_states(sound_classes_states=self.sound_classes_states)
        self.processor.signal_play_sound.connect(self.soundThread.start_stop_sound_thread)
        self.processor.signal_channel_central_freq.connect(self.settingsWidget.usrpTab.update_channel_freq)
        self.processor.signal_channel_central_freq.connect(lambda freq_dict:
                                 self.settingsWidget.alinxTab.update_cb_central_freq(freq_dict['central_freq']))

        self.settingsWidget.mainTab.cb_spectrogram_resolution.currentTextChanged.connect(lambda a:
        self.set_spectrogram_resolution(self.settingsWidget.mainTab.cb_spectrogram_resolution.currentData()))

        self.telemetryWidget = TelemetryWidget(theme_type=self.theme_type)
        self.gRPCErrorTread.signal_telemetry.connect(self.telemetryWidget.udpate_widgets_states)

        self.show()
        self.move_window_to_center()

    def set_spectrogram_resolution(self, new_resolution: tuple[int, int]):
        for recogn_widget in self.recogn_widgets.values():
            recogn_widget.resize_img(new_resolution)

    def create_menu(self):
        self.act_settings = QAction('Settings', self)
        self.act_settings.setIcon(QIcon(f'assets/icons/{self.theme_type}/btn_settings.png'))
        self.act_settings.triggered.connect(self.open_settings)

        self.act_telemetry = QAction('Telemetry', self)
        self.act_telemetry.setIcon(QIcon(f'assets/icons/{self.theme_type}/telemetry.png'))
        self.act_telemetry.triggered.connect(self.open_telemetry)

    def create_actions(self):
        self.act_start = QAction()
        self.act_start.setIcon(QIcon(f'assets/icons/{self.theme_type}/btn_start.png'))
        self.act_start.setText('Start')
        self.act_start.setCheckable(True)
        self.act_start.triggered.connect(self.change_connection_state)

    def create_toolbar(self):
        self.toolBar = QToolBar('Toolbar')
        self.toolBar.addAction(self.act_start)
        self.toolBar.addAction(self.act_settings)
        self.toolBar.addAction(self.act_telemetry)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolBar)

    def init_recognition_widgets(self):
        current_zscale_settings_dict = self.gRPCThread.getCurrentZScaleRequest()
        recogn_settings = self.gRPCThread.gerCurrentRecognitionSettings()
        for channel_info in self.channels_info:
            if channel_info.name in self.enabled_channels:
                recogn_widget = RecognitionWidget(window_name=channel_info.name,
                                                  map_list=self.map_list,
                                                  img_show_status=self.show_img_status,
                                                  zscale_settings=current_zscale_settings_dict[channel_info.name],
                                                  recogn_options=recogn_settings[channel_info.name],
                                                  signal_settings=self.signal_settings[channel_info.name],
                                                  show_recogn_options=bool(self.config['settings_main']['show_recogn_options']),
                                                  show_freq=bool(self.config['settings_main']['show_frequencies']),
                                                  show_images=self.show_img_status,
                                                  show_histogram=self.show_histogram_status,
                                                  show_spectrum=self.show_spectrum_status,
                                                  channel_info=channel_info,
                                                  theme_type=self.theme_type,)
                recogn_widget.recognOptions.signal_zscale_changed.connect(self.gRPCThread.changeZScaleRequest)
                recogn_widget.recognOptions.signal_recogn_settings.connect(self.gRPCThread.sendRecognitionSettings)
                # recogn_widget.recognOptions.signal_freq_changed.connect(self.gRPCThread.setFrequency)
                # recogn_widget.recognOptions.signal_gain_changed.connect(self.gRPCThread.setGain)
                self.settingsWidget.mainTab.chb_show_recogn_options.stateChanged.connect(recogn_widget.add_remove_recogn_options)
                self.settingsWidget.mainTab.chb_show_frequencies.stateChanged.connect(recogn_widget.show_frequencies)
                recogn_widget.processOptions.signal_process_name.connect(self.gRPCThread.getProcessStatusRequest)
                recogn_widget.processOptions.signal_restart_process_name.connect(self.gRPCThread.restartProcess)
                self.gRPCThread.signal_process_status.connect(recogn_widget.processOptions.update_process_status)

                self.recogn_widgets[channel_info.name] = recogn_widget
        self.add_recogn_widgets(type_of_adding='left')
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

        elif type_of_adding == 'left':
            for widget in self.recogn_widgets.values():
                self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, widget)

    def change_connection_state(self, status: bool):
        if status:
            self.act_start.setIcon(QIcon(f'assets/icons/{self.theme_type}/btn_stop.png'))
            for channel in self.enabled_channels:
                try:
                    self.gRPCThread.startChannelRequest(channel_name=channel)
                except:
                    self.logger_.warning(f'Error with starting gRPC channel {channel} or channel is already started.')

            self.gRPCThread.start()
            self.gRPCErrorTread.start()
        else:
            self.act_start.setIcon(QIcon(f'assets/icons/{self.theme_type}/btn_start.png'))

            # Stop gRPC main thread
            if self.gRPCThread.isRunning():
                self.gRPCThread.requestInterruption()
                self.logger_.info('gRPCThread is requested to stop.')
            else:
                self.logger_.warning('gRPCThread is not running.')

            # Stop gRPC Error thread
            if self.gRPCErrorTread.isRunning():
                self.gRPCErrorTread.requestInterruption()
                self.logger_.info('gRPCErrorTread is requested to stop.')
            else:
                self.logger_.warning('gRPCErrorTread is not running.')

    def open_recognition_settings(self, channel):
        self.recogn_settings_widgets[channel].show()

    def open_settings(self):
        self.settingsWidget.show()

    def open_telemetry(self):
        self.telemetryWidget.show()

    def link_events(self):
        self.gRPCThread.signal_dataStream_response.connect(
            lambda info: self.processor.parsing_data(info,
                                                     self.settingsWidget.saveTab.chb_save_detected.isChecked(),
                                                     ))

    def change_save_status(self):
        if self.clear_img_status:
            self.clear_img_status = False
            self.gRPCThread.clear_img_status = False
            self.change_connection_state(status=False)      # restart gRPC stream
            self.gRPCThread.msleep(500)
            self.change_connection_state(status=True)
            self.logger_.info('Saving images stopped.')
        else:
            self.clear_img_status = True
            self.gRPCThread.clear_img_status = True
            self.change_connection_state(status=False)      # restart gRPC stream
            self.gRPCThread.msleep(500)
            self.change_connection_state(status=True)
            self.logger_.info('Saving images started.')

    def change_img_status(self):
        if self.show_img_status:
            self.show_img_status = False
            self.gRPCThread.show_img_status = False
            for recogn_widget in self.recogn_widgets.values():
                recogn_widget.disable_spectrogram()
            self.change_connection_state(status=False)      # restart gRPC stream
            self.gRPCThread.msleep(500)
            self.change_connection_state(status=True)
            self.logger_.info('Spectrogram showing stopped.')
        else:
            self.show_img_status = True
            self.gRPCThread.show_img_status = True
            for recogn_widget in self.recogn_widgets.values():
                recogn_widget.enable_spectrogram()
            self.change_connection_state(status=False)      # restart gRPC stream
            self.gRPCThread.msleep(500)
            self.change_connection_state(status=True)
            self.logger_.info('Spectrogram showing started.')

    def change_histogram_status(self):
        if self.show_histogram_status:
            self.show_histogram_status = False
            for recogn_widget in self.recogn_widgets.values():
                recogn_widget.disable_histogram()
            self.logger_.info('Histogram showing stopped.')
        else:
            self.show_histogram_status = True
            for recogn_widget in self.recogn_widgets.values():
                recogn_widget.enable_histogram()
            self.logger_.info('Histogram showing started.')

    def change_spectrum_status(self):
        if self.show_spectrum_status:
            self.show_spectrum_status = False
            self.gRPCThread.show_spectrum_status = False
            for recogn_widget in self.recogn_widgets.values():
                recogn_widget.disable_spectrum()
            self.change_connection_state(status=False)      # restart gRPC stream
            self.gRPCThread.msleep(500)
            self.change_connection_state(status=True)
            self.logger_.info('Spectrum showing stopped.')
        else:
            self.show_spectrum_status = True
            self.gRPCThread.show_spectrum_status = True
            for recogn_widget in self.recogn_widgets.values():
                recogn_widget.enable_spectrum()
            self.change_connection_state(status=False)      # restart gRPC stream
            self.gRPCThread.msleep(500)
            self.change_connection_state(status=True)
            self.logger_.info('Spectrum showing started.')

    def change_sound_states(self):
        pass

    def save_config(self):
        try:
            config = {}
            self.config.update(self.settingsWidget.mainTab.collect_config())
            self.config.update(self.settingsWidget.soundTab.collect_config())
            self.config.update(self.settingsWidget.alinxTab.collect_config())
            self.config.update(config)

            with open('client_conf.yaml', 'w') as f:
                yaml.dump(self.config, f, sort_keys=False)
            self.logger_.success(f'Client config saved successfully!')
        except Exception as e:
            self.logger_.error(f'Error with saving client config! \n{e}')

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

        # Перемещаем окно в центр экрана
        self.move(x, y)

    def apply_theme(self):
        theme_name = self.settingsWidget.mainTab.cb_themes.currentText()
        self.theme_type = self.settingsWidget.mainTab.cb_theme_type.currentText()
        qdarktheme.setup_theme(theme=self.theme_type,
                               custom_colors=self.themes[theme_name],
                               additional_qss="QToolTip { "
                                              "background-color: #ffff99;"
                                              "color: #000000;"
                                              "border: 1px solid #000000;"
                                              "padding: 2px;}")
        self.act_settings.setIcon(QIcon(f'./assets/icons/{self.theme_type}/btn_settings.png'))
        self.act_telemetry.setIcon(QIcon(f'assets/icons/{self.theme_type}/telemetry.png'))
        self.settingsWidget.soundTab.btn_play_sound.setIcon(QIcon(f'./assets/icons/{self.theme_type}/play_sound.png'))
        for widget in self.recogn_widgets.values():
            widget.theme_changed(theme=self.theme_type)
        if self.act_start.isChecked():
            self.act_start.setIcon(QIcon(f'./assets/icons/{self.theme_type}/btn_stop.png'))
        else:
            self.act_start.setIcon(QIcon(f'./assets/icons/{self.theme_type}/btn_start.png'))
        self.telemetryWidget.theme_changed(type=self.theme_type)
    # def closeEvent(self, a0):
    #     self.gRPCThread.gRPC_channel.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    qdarktheme.setup_theme(theme='dark')
    main_window = MainWindow()
    # main_window.welcomeWindow.show()
    sys.exit(app.exec())



