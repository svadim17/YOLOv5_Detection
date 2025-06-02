import os
import sys
import time
from PySide6.QtWidgets import QMainWindow, QApplication, QToolBar, QGridLayout, QTabWidget, QSizePolicy, QWidget, QLabel
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt, QTimer
from datetime import datetime
import qdarktheme
from client_submodules.gRPC_thread import gRPCThread, connect_to_gRPC_server, gRPCServerErrorThread
from client_submodules.welcome_window import WelcomeWindow
from client_submodules.connection_window import ConnectWindow
from client_submodules.recognition_widget import RecognitionWidget
from client_submodules.channels_widget import ChannelsWidget
from client_submodules.settings import SettingsWidget
from client_submodules.processing import Processor
from client_submodules.sound_thread import SoundThread
from client_submodules.telemetry_widget import TelemetryWidget
from client_submodules.map_widget import MapWidget
from client_submodules.map_widget import UAVObject
from client_submodules.remote_id_widget import RemoteIdWidget
from client_submodules.wifi_widget import WiFiWidget
from client_submodules.aeroscope_widget import AeroscopeWidget


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
        central_widget = QWidget(self)
        self.grid = QGridLayout(central_widget)
        self.setCentralWidget(central_widget)

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
        self.init_widgets_status()

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

    def init_widgets_status(self):
        self.channels_status = True
        self.map_status = bool(self.config['widgets']['map_status'])
        self.telemetry_status = bool(self.config['widgets']['telemetry_status'])
        self.aeroscope_status = bool(self.config['widgets']['aeroscope_status'])
        self.wifi_status = bool(self.config['widgets']['wifi_status'])
        self.remote_id_status = bool(self.config['widgets']['remote_id_status'])

        self.channels_show_status = True
        self.map_show_status = bool(self.config['widgets_show']['map_show_status']) if self.map_status else False
        self.telemetry_show_status = bool(self.config['widgets_show']['telemetry_show_status']) if self.telemetry_status else False
        self.aeroscope_show_status = bool(self.config['widgets_show']['aeroscope_show_status']) if self.aeroscope_status else False
        self.wifi_show_status = bool(self.config['widgets_show']['wifi_show_status']) if self.wifi_status else False
        self.remote_id_show_status = bool(self.config['widgets_show']['remote_id_show_status']) if self.remote_id_status else False

        self.map_ind, self.telemetry_ind, self.aeroscope_ind, self.wifi_ind, self.rid_ind = None, None, None, None, None

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
        if self.map_status:
            self.init_map_widget()
        if self.telemetry_status:
            self.init_telemetry_widget()
        if self.aeroscope_status:
            self.init_aeroscope_widget()
        if self.remote_id_status:
            self.init_remote_id_widget()
        if self.wifi_status:
            self.init_wifi_widget()
        self.add_widgets_to_grid()

        self.processor.init_sound_states(sound_states=self.sound_states)
        self.processor.init_sound_classes_states(sound_classes_states=self.sound_classes_states)
        self.processor.signal_play_sound.connect(self.soundThread.start_stop_sound_thread)
        self.processor.signal_channel_central_freq.connect(self.settingsWidget.usrpTab.update_channel_freq)
        self.processor.signal_channel_central_freq.connect(lambda freq_dict:
                                 self.settingsWidget.alinxTab.update_cb_central_freq(freq_dict['central_freq']))

        self.settingsWidget.mainTab.cb_spectrogram_resolution.currentTextChanged.connect(lambda a:
        self.set_spectrogram_resolution(self.settingsWidget.mainTab.cb_spectrogram_resolution.currentData()))


        self.show()
        self.move_window_to_center()

    def create_menu(self):
        self.act_start = QAction()
        self.act_start.setIcon(QIcon(f'assets/icons/{self.theme_type}/btn_start.png'))
        self.act_start.setText('Start')
        self.act_start.setCheckable(True)
        self.act_start.triggered.connect(self.change_connection_state)

        self.act_settings = QAction('Settings', self)
        self.act_settings.setIcon(QIcon(f'assets/icons/{self.theme_type}/btn_settings.png'))
        self.act_settings.triggered.connect(self.open_settings)

        self.act_channels = QAction('Channels', self)
        self.act_channels.setIcon(QIcon(f'assets/icons/{self.theme_type}/eye_on.png'))
        self.act_channels.triggered.connect(self.open_channels)

        self.act_screenshot = QAction('Make screenshot', self)
        self.act_screenshot.setIcon(QIcon(f'assets/icons/{self.theme_type}/screenshot.png'))
        self.act_screenshot.triggered.connect(self.make_screenshot)

        if self.telemetry_status:
            telemetry_icon_state = '_on' if self.telemetry_show_status else ''
            self.act_telemetry = QAction('Telemetry', self)
            self.act_telemetry.setIcon(QIcon(f'assets/icons/{self.theme_type}/telemetry{telemetry_icon_state}.png'))
            self.act_telemetry.triggered.connect(self.open_telemetry)

        if self.map_status:
            map_icon_state = '_on' if self.map_show_status else ''
            self.act_map = QAction('Map', self)
            self.act_map.setIcon(QIcon(f'assets/icons/{self.theme_type}/map{map_icon_state}.png'))
            self.act_map.triggered.connect(self.open_map)

        if self.aeroscope_status:
            aeroscope_icon_state = '_on' if self.aeroscope_show_status else ''
            self.act_aeroscope = QAction('Aeroscope', self)
            self.act_aeroscope.setIcon(QIcon(f'assets/icons/{self.theme_type}/aeroscope{aeroscope_icon_state}.png'))
            self.act_aeroscope.triggered.connect(self.open_aeroscope)

        if self.remote_id_status:
            remote_id_icon_state = '_on' if self.remote_id_show_status else ''
            self.act_remote_id = QAction('Remote ID', self)
            self.act_remote_id.setIcon(QIcon(f'assets/icons/{self.theme_type}/remote_id{remote_id_icon_state}.png'))
            self.act_remote_id.triggered.connect(self.open_remote_id)

        if self.wifi_status:
            wifi_icon_state = '_on' if self.wifi_show_status else ''
            self.act_wifi = QAction('WiFi', self)
            self.act_wifi.setIcon(QIcon(f'assets/icons/{self.theme_type}/wifi{wifi_icon_state}.png'))
            self.act_wifi.triggered.connect(self.open_wifi)

    def create_toolbar(self):
        self.toolBar = QToolBar('Toolbar')
        self.toolBar.addAction(self.act_start)
        self.toolBar.addAction(self.act_settings)
        self.toolBar.addAction(self.act_channels)
        if self.telemetry_status:
            self.toolBar.addAction(self.act_telemetry)
        if self.map_status:
            self.toolBar.addAction(self.act_map)
        if self.aeroscope_status:
            self.toolBar.addAction(self.act_aeroscope)
        if self.remote_id_status:
            self.toolBar.addAction(self.act_remote_id)
        if self.wifi_status:
            self.toolBar.addAction(self.act_wifi)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolBar)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolBar.addWidget(spacer)
        self.toolBar.addAction(self.act_screenshot)

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

        self.processor.init_recogn_widgets(recogn_widgets=self.recogn_widgets)

    def init_map_widget(self):
        self.mapWidget = MapWidget(map_settings=self.config['map'], theme_type=self.theme_type)

    def init_telemetry_widget(self):
        self.telemetryWidget = TelemetryWidget(theme_type=self.theme_type)
        self.gRPCErrorTread.signal_telemetry.connect(self.telemetryWidget.udpate_widgets_states)

    def init_aeroscope_widget(self):
        self.aeroscopeWidget = AeroscopeWidget(theme_type=self.theme_type, logger_=self.logger_)

    def init_remote_id_widget(self):
        self.remoteIdWidget = RemoteIdWidget(theme_type=self.theme_type, logger_=self.logger_)

    def init_wifi_widget(self):
        self.wifiWidget = WiFiWidget(theme_type=self.theme_type, logger_=self.logger_)

    def add_widgets_to_grid(self):
        self.tab_bottom = QTabWidget()
        self.tab_bottom.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        if self.map_status:
            self.map_ind = self.tab_bottom.addTab(self.mapWidget, 'Map')
        if self.telemetry_status:
            self.telemetry_ind = self.tab_bottom.addTab(self.telemetryWidget, "Server's Telemetry")
        if self.map_ind is not None and not self.map_show_status:
            self.tab_bottom.widget(self.map_ind).hide()
            self.tab_bottom.setTabText(self.map_ind, "")
        if self.telemetry_ind is not None and not self.telemetry_show_status:
            self.tab_bottom.widget(self.telemetry_ind).hide()
            self.tab_bottom.setTabText(self.telemetry_ind, "")
        self.grid.addWidget(self.tab_bottom, 1, 0, 1, len(self.enabled_channels))

        self.tab_right = QTabWidget()
        self.tab_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        if self.aeroscope_status:
            self.aeroscope_ind = self.tab_right.addTab(self.aeroscopeWidget, 'Aeroscope')
        if self.remote_id_status:
            self.rid_ind = self.tab_right.addTab(self.remoteIdWidget, 'Remote ID')
        if self.wifi_status:
            self.wifi_ind = self.tab_right.addTab(self.wifiWidget, 'WiFi')
        if self.aeroscope_ind is not None and not self.aeroscope_show_status:
            self.tab_right.widget(self.aeroscope_ind).hide()
            self.tab_right.setTabText(self.aeroscope_ind, "")
        if self.rid_ind is not None and not self.remote_id_show_status:
            self.tab_right.widget(self.rid_ind).hide()
            self.tab_right.setTabText(self.rid_ind, "")
        if self.wifi_ind is not None and not self.wifi_show_status:
            self.tab_right.widget(self.wifi_ind).hide()
            self.tab_right.setTabText(self.wifi_ind, "")
        self.grid.addWidget(self.tab_right, 0, len(self.enabled_channels), 2, 1)

        i = 0
        for widg in self.recogn_widgets.values():
            self.grid.addWidget(widg, 0, i)
            i += 1

        if not self.map_show_status and not self.telemetry_show_status:
            self.tab_bottom.hide()
        if not self.aeroscope_show_status and not self.wifi_show_status and not self.remote_id_show_status:
            self.tab_right.hide()

    def set_spectrogram_resolution(self, new_resolution: tuple[int, int]):
        for recogn_widget in self.recogn_widgets.values():
            recogn_widget.resize_img(new_resolution)

    def change_connection_state(self, status: bool):
        if status:
            self.act_start.setIcon(QIcon(f'assets/icons/{self.theme_type}/btn_stop.png'))
            self.mapWidget.map_emulation() if self.map_status else None
            self.aeroscope_emulation() if self.aeroscope_status else None
            self.remote_id_emulation() if self.remote_id_status else None
            self.wifi_emulation() if self.wifi_status else None

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

    def aeroscope_emulation(self):
        self.aeroscope_timer = QTimer(self)
        self.aeroscope_timer.timeout.connect(self.aeroscopeWidget.emulate)
        self.aeroscope_timer.start(3000)  # каждые 3 секунды

    def remote_id_emulation(self):
        self.rid_timer = QTimer(self)
        self.rid_timer.timeout.connect(self.remoteIdWidget.emulate)
        self.rid_timer.start(3000)  # каждые 3 секунды

    def wifi_emulation(self):
        self.wifi_timer = QTimer(self)
        self.wifi_timer.timeout.connect(self.wifiWidget.emulate)
        self.wifi_timer.start(3000)  # каждые 3 секунды

    def open_settings(self):
        self.settingsWidget.show()

    def open_channels(self):
        if self.channels_show_status:
            for widg in self.recogn_widgets.values():
                self.grid.removeWidget(widg)
                widg.setParent(None)  # Отключает от layout, но не удаляет объект
            self.act_channels.setIcon(QIcon(f'assets/icons/{self.theme_type}/eye.png'))
            self.channels_show_status = False
        else:
            i = 0
            for widg in self.recogn_widgets.values():
                self.grid.addWidget(widg, 0, i)
                i += 1
            self.act_channels.setIcon(QIcon(f'assets/icons/{self.theme_type}/eye_on.png'))
            self.channels_show_status = True

    def open_map(self):
        if self.mapWidget.isVisible():
            self.tab_bottom.widget(self.map_ind).hide()
            self.tab_bottom.setTabEnabled(self.map_ind, False)
            self.tab_bottom.setTabText(self.map_ind, "")  # Или "Скрыто"
            self.act_map.setIcon(QIcon(f'assets/icons/{self.theme_type}/map.png'))
            self.map_show_status = False
            if not self.map_show_status and not self.telemetry_show_status:
                self.tab_bottom.hide()
        else:
            self.tab_bottom.widget(self.map_ind).show()
            self.tab_bottom.setTabEnabled(self.map_ind, True)
            self.tab_bottom.setTabText(self.map_ind, "Map")
            self.act_map.setIcon(QIcon(f'assets/icons/{self.theme_type}/map_on.png'))
            self.map_show_status = True
            if self.tab_bottom.isHidden():
                self.tab_bottom.show()

    def open_telemetry(self):
        if self.telemetryWidget.isVisible():
            self.tab_bottom.widget(self.telemetry_ind).hide()
            self.tab_bottom.setTabEnabled(self.telemetry_ind, False)
            self.tab_bottom.setTabText(self.telemetry_ind, "")  # Или "Скрыто"
            self.act_telemetry.setIcon(QIcon(f'assets/icons/{self.theme_type}/telemetry.png'))
            self.telemetry_show_status = False
            if not self.map_show_status and not self.telemetry_show_status:
                self.tab_bottom.hide()
        else:
            self.tab_bottom.widget(self.telemetry_ind).show()
            self.tab_bottom.setTabEnabled(self.telemetry_ind, True)
            self.tab_bottom.setTabText(self.telemetry_ind, "Server's Telemetry")
            self.act_telemetry.setIcon(QIcon(f'assets/icons/{self.theme_type}/telemetry_on.png'))
            self.telemetry_show_status = True
            if self.tab_bottom.isHidden():
                self.tab_bottom.show()

    def open_aeroscope(self):
        if self.aeroscopeWidget.isVisible():
            self.tab_right.widget(self.aeroscope_ind).hide()
            self.tab_right.setTabEnabled(self.aeroscope_ind, False)
            self.tab_right.setTabText(self.aeroscope_ind, "")  # Или "Скрыто"
            self.act_aeroscope.setIcon(QIcon(f'assets/icons/{self.theme_type}/aeroscope.png'))
            self.aeroscope_show_status = False
            if not self.aeroscope_show_status and not self.remote_id_show_status and not self.wifi_show_status:
                self.tab_right.hide()
        else:
            self.tab_right.widget(self.aeroscope_ind).show()
            self.tab_right.setTabEnabled(self.aeroscope_ind, True)
            self.tab_right.setTabText(self.aeroscope_ind, "Aeroscope")
            self.act_aeroscope.setIcon(QIcon(f'assets/icons/{self.theme_type}/aeroscope_on.png'))
            self.aeroscope_show_status = True
            if self.tab_right.isHidden():
                self.tab_right.show()

    def open_remote_id(self):
        if self.remoteIdWidget.isVisible():
            # self.tab_right.removeTab(1)
            self.tab_right.widget(self.rid_ind).hide()
            self.tab_right.setTabEnabled(self.rid_ind, False)
            self.tab_right.setTabText(self.rid_ind, "")  # Или "Скрыто"
            self.act_remote_id.setIcon(QIcon(f'assets/icons/{self.theme_type}/remote_id.png'))
            self.remote_id_show_status = False
            if not self.aeroscope_show_status and not self.remote_id_show_status and not self.wifi_show_status:
                self.tab_right.hide()
        else:
            # self.tab_right.insertTab(1, self.remoteIdWidget, "Remote ID")
            self.tab_right.widget(self.rid_ind).show()
            self.tab_right.setTabEnabled(self.rid_ind, True)
            self.tab_right.setTabText(self.rid_ind, "Remote ID")
            self.act_remote_id.setIcon(QIcon(f'assets/icons/{self.theme_type}/remote_id_on.png'))
            self.remote_id_show_status = True
            if self.tab_right.isHidden():
                self.tab_right.show()

    def open_wifi(self):
        if self.wifiWidget.isVisible():
            # self.tab_right.removeTab(2)
            self.tab_right.widget(self.wifi_ind).hide()
            self.tab_right.setTabEnabled(self.wifi_ind, False)
            self.tab_right.setTabText(self.wifi_ind, "")  # Или "Скрыто"
            self.act_wifi.setIcon(QIcon(f'assets/icons/{self.theme_type}/wifi.png'))
            self.wifi_show_status = False
            if not self.aeroscope_show_status and not self.remote_id_show_status and not self.wifi_show_status:
                self.tab_right.hide()
        else:
            # self.tab_right.insertTab(2, self.wifiWidget, "WiFi")
            self.tab_right.widget(self.wifi_ind).show()
            self.tab_right.setTabEnabled(self.wifi_ind, True)
            self.tab_right.setTabText(self.wifi_ind, "WiFi")
            self.act_wifi.setIcon(QIcon(f'assets/icons/{self.theme_type}/wifi_on.png'))
            self.wifi_show_status = True
            if self.tab_right.isHidden():
                self.tab_right.show()

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

    def save_config(self):
        try:
            config = {}
            self.config.update(self.settingsWidget.mainTab.collect_config())
            self.config.update(self.settingsWidget.soundTab.collect_config())
            self.config.update(self.settingsWidget.alinxTab.collect_config())
            self.config.update(self.mapWidget.collect_config())
            self.config.update(self.collect_ui_config())
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

    def collect_ui_config(self):
        config = {'widgets_show': {'map_show_status': self.map_show_status,
                                   'telemetry_show_status': self.telemetry_show_status,
                                   'aeroscope_show_status': self.aeroscope_show_status,
                                   'wifi_show_status': self.wifi_show_status,
                                   'remote_id_show_status': self.remote_id_show_status,
                                   }
                  }
        return config

    def make_screenshot(self):
        if not os.path.isdir('Screenshots'):
            os.mkdir('Screenshots')
        try:
            pixmap = self.grab()
            name = f'{datetime.now().strftime("%H-%M-%S")}.png'
            filepath = os.path.join('Screenshots', name)
            pixmap.save(filepath, 'PNG')
            self.show_screenshot_notice()
            logger.info(f'Screenshot saved as {filepath}')
        except Exception as e:
            logger.warning(f"Error with saving screenshot!")

    def show_screenshot_notice(self):
        notice = QLabel("📸 Screenshot saved", self)
        notice.setStyleSheet("""
            background-color: rgba(0, 0, 0, 160);
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 16px;
        """)
        notice.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        notice.resize(190, 40)

        # locate on center position
        center_x = (self.width() - notice.width()) // 2
        center_y = (self.height() - notice.height()) // 2
        notice.move(center_x, center_y)
        notice.show()

        QTimer.singleShot(1000, notice.deleteLater)     # hide

    def move_window_to_center(self):
        # get screen size
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        window_width = self.width()             # get window width
        window_height = self.height()           # get window height

        x = (screen_width - window_width) // 2              # calculate x for screen center
        y = (screen_height - window_height) // 2            # calculate y for screen center

        self.move(x, y)     # move window to center

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

        channels_state = '_on' if self.channels_show_status else ''
        telemetry_state = '_on' if self.telemetry_show_status else ''
        map_state = '_on' if self.map_show_status else ''
        aeroscope_state = '_on' if self.aeroscope_show_status else ''
        rid_state = '_on' if self.remote_id_show_status else ''
        wifi_state = '_on' if self.wifi_show_status else ''

        self.act_settings.setIcon(QIcon(f'./assets/icons/{self.theme_type}/btn_settings.png'))
        self.act_channels.setIcon(QIcon(f'assets/icons/{self.theme_type}/eye{channels_state}.png'))
        self.act_telemetry.setIcon(QIcon(f'assets/icons/{self.theme_type}/telemetry{telemetry_state}.png'))
        self.act_map.setIcon(QIcon(f'assets/icons/{self.theme_type}/map{map_state}.png'))
        self.act_aeroscope.setIcon(QIcon(f'assets/icons/{self.theme_type}/aeroscope{aeroscope_state}.png'))
        self.act_remote_id.setIcon(QIcon(f'assets/icons/{self.theme_type}/remote_id{rid_state}.png'))
        self.act_wifi.setIcon(QIcon(f'assets/icons/{self.theme_type}/wifi{wifi_state}.png'))

        self.settingsWidget.soundTab.btn_play_sound.setIcon(QIcon(f'./assets/icons/{self.theme_type}/play_sound.png'))
        for widget in self.recogn_widgets.values():
            widget.theme_changed(theme=self.theme_type)
        if self.act_start.isChecked():
            self.act_start.setIcon(QIcon(f'./assets/icons/{self.theme_type}/btn_stop.png'))
        else:
            self.act_start.setIcon(QIcon(f'./assets/icons/{self.theme_type}/btn_start.png'))
        self.telemetryWidget.theme_changed(type=self.theme_type)
        self.remoteIdWidget.theme_changed(type=self.theme_type)
        self.wifiWidget.theme_changed(type=self.theme_type)
        self.mapWidget.theme_changed(type=self.theme_type)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    qdarktheme.setup_theme(theme='dark')
    main_window = MainWindow()
    # main_window.welcomeWindow.show()
    sys.exit(app.exec())



