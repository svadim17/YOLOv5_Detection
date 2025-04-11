import loguru
from PyQt6.QtWidgets import (QWidget, QListWidget, QApplication, QSpacerItem, QSizePolicy,
                             QStackedWidget, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QTabWidget, QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtCore import pyqtSlot, Qt, pyqtSignal
import qdarktheme
from os import walk
import yaml


class SettingsWidget(QWidget):
    def __init__(self, enabled_channels: list, config: dict, enabled_channels_info: list, logger_):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle('Settings')
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)
        self.enabled_channels_info = enabled_channels_info
        self.logger = logger_

        # Проверка, есть ли 'alinx' в hardware_type какого-либо объекта
        alinx_hardware_status = any(channel.hardware_type.lower() == 'alinx' or
                                    channel.hardware_type.lower() == 'Alinx' or
                                    channel.hardware_type.lower() == 'ALINX' for channel in self.enabled_channels_info)

        # Проверка, есть ли 'usrp' в hardware_type какого-либо объекта
        usrp_hardware_status = any(channel.hardware_type.lower() == 'usrp' or
                                   channel.hardware_type.lower() == 'Usrp' or
                                   channel.hardware_type.lower() == 'USRP' for channel in self.enabled_channels_info)

        self.mainTab = MainTab(config=config, logger_=self.logger)
        self.saveTab = SaveTab(enabled_channels=enabled_channels, map_list=config['map_list'])
        self.soundTab = SoundTab(enabled_channels=enabled_channels, config=config, logger_=self.logger)
        self.nnTab = NNTab(logger_=self.logger, enabled_channels=enabled_channels)
        self.alinxTab = AlinxTab(enabled_channels_info=self.enabled_channels_info,
                                 autoscan=bool(config['fcm']['autoscan_frequency']),
                                 logger_=self.logger)
        self.usrpTab = USRPTab(enabled_channels=enabled_channels, logger_=self.logger)

        self.tab = QTabWidget()
        self.tab.addTab(self.mainTab, 'Main')
        self.tab.addTab(self.saveTab, 'Images saving')
        self.tab.addTab(self.soundTab, 'Sound')
        self.tab.addTab(self.nnTab, 'NN')
        if alinx_hardware_status:
            self.tab.addTab(self.alinxTab, 'Alinx')
        if usrp_hardware_status:
            self.tab.addTab(self.usrpTab, 'USRP')

        self.btn_save_client_config = QPushButton('Save client config')
        self.btn_save_server_config = QPushButton('Save server config')

        self.add_widgets_to_layout()

    def add_widgets_to_layout(self):
        self.main_layout.addWidget(self.tab)

        btns_layout = QHBoxLayout()
        btns_layout.addWidget(self.btn_save_client_config)
        btns_layout.addWidget(self.btn_save_server_config)
        self.main_layout.addLayout(btns_layout)


class MainTab(QWidget):

    def __init__(self, config: dict, logger_):
        super().__init__()
        self.config = config
        self.logger = logger_
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)
        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.l_spectrogram_resolution = QLabel('Spectrogram resolution')
        self.cb_spectrogram_resolution = QComboBox()
        self.cb_spectrogram_resolution.setMinimumWidth(100)
        self.cb_spectrogram_resolution.addItem('240x240', (240, 240))
        self.cb_spectrogram_resolution.addItem('320x320', (320, 320))
        self.cb_spectrogram_resolution.addItem('400x400', (400, 400))
        self.cb_spectrogram_resolution.addItem('480x480', (480, 480))
        self.cb_spectrogram_resolution.addItem('560x560', (560, 560))
        self.cb_spectrogram_resolution.addItem('640x640', (640, 640))
        self.cb_spectrogram_resolution.setCurrentIndex(1)

        self.chb_show_recogn_options = QCheckBox('Show recognition options')
        self.chb_show_recogn_options.setChecked(True)

        self.chb_show_frequencies = QCheckBox('Show frequencies')
        self.chb_show_frequencies.setChecked(self.config['settings_main']['show_frequencies'])

        self.chb_accumulation = QCheckBox('Enable accumulation')
        self.chb_accumulation.setChecked(True)

        self.chb_watchdog = QCheckBox('Watchdog')
        self.chb_watchdog.setChecked(self.config['settings_main']['watchdog'])
        self.chb_watchdog.setDisabled(True)

        self.box_widgets_states = QGroupBox('Widgets states')

        self.chb_spectrogram_show = QCheckBox('Show spectrogram')
        self.chb_spectrogram_show.setChecked(self.config['settings_main']['show_spectrogram'])

        self.chb_histogram_show = QCheckBox('Show histogram')
        self.chb_histogram_show.setChecked(self.config['settings_main']['show_histogram'])

        self.chb_spectrum_show = QCheckBox('Show spectrum')
        self.chb_spectrum_show.setChecked(self.config['settings_main']['show_spectrum'])

        self.box_theme_settings = QGroupBox('Theme')

        self.cb_themes = QComboBox()
        all_themes = self.get_all_themes()
        input_theme_name = self.config['settings_main']['theme']['name']
        input_theme_type = self.config['settings_main']['theme']['type']

        for theme in all_themes:
            self.cb_themes.addItem(theme)
        if input_theme_name in all_themes:
            self.cb_themes.setCurrentText(input_theme_name)
        else:
            self.cb_themes.setCurrentText('default')
            self.logger.warning(f'Unknown theme name {input_theme_name}. Theme is set to default')

        self.cb_theme_type = QComboBox()
        self.cb_theme_type.addItem('dark')
        self.cb_theme_type.addItem('light')
        if input_theme_type == 'dark' or input_theme_type == 'light':
            self.cb_theme_type.setCurrentText(input_theme_type)
        else:
            self.cb_theme_type.setCurrentText('dark')
            self.logger.warning(f'Unknown theme type {input_theme_type}. Theme type is set to dark')

    def add_widgets_to_layout(self):
        spectr_layout = QVBoxLayout()
        spectr_layout.setSpacing(5)
        spectr_layout.addWidget(self.l_spectrogram_resolution, alignment=Qt.AlignmentFlag.AlignLeft)
        spectr_layout.addWidget(self.cb_spectrogram_resolution, alignment=Qt.AlignmentFlag.AlignLeft)

        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        controls_layout.addWidget(self.chb_show_recogn_options)
        controls_layout.addWidget(self.chb_show_frequencies)
        controls_layout.addWidget(self.chb_accumulation)
        controls_layout.addWidget(self.chb_watchdog)

        widgets_states_layout = QVBoxLayout()
        widgets_states_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        widgets_states_layout.addWidget(self.chb_spectrogram_show)
        widgets_states_layout.addWidget(self.chb_histogram_show)
        widgets_states_layout.addWidget(self.chb_spectrum_show)
        self.box_widgets_states.setLayout(widgets_states_layout)

        theme_layout = QVBoxLayout()
        theme_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        theme_layout.addWidget(self.cb_themes)
        theme_layout.addSpacing(10)
        theme_layout.addWidget(self.cb_theme_type)
        self.box_theme_settings.setLayout(theme_layout)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.box_widgets_states)
        left_layout.addWidget(self.box_theme_settings)

        central_layout = QHBoxLayout()
        central_layout.addLayout(left_layout)
        central_layout.addSpacing(20)
        central_layout.addLayout(controls_layout)

        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.main_layout.addLayout(spectr_layout)
        self.main_layout.addLayout(central_layout)
        self.main_layout.addSpacerItem(spacer)

    def get_all_themes(self):
        with open('app_themes.yaml', 'r', encoding='utf-8') as f:
            themes = yaml.safe_load(f)
        return themes.keys()

    def collect_config(self):
        config = {'settings_main': {'show_recogn_options': self.chb_show_recogn_options.isChecked(),
                                    'show_frequencies': self.chb_show_frequencies.isChecked(),
                                    'enable_accumulation': self.chb_accumulation.isChecked(),
                                    'watchdog': self.chb_watchdog.isChecked(),
                                    'show_spectrogram': self.chb_spectrogram_show.isChecked(),
                                    'show_histogram': self.chb_histogram_show.isChecked(),
                                    'show_spectrum': self.chb_spectrum_show.isChecked(),
                                    'theme': {
                                        'name': self.cb_themes.currentText(),
                                        'type': self.cb_theme_type.currentText()
                                             }
                                    }
                  }
        return config


class SaveTab(QWidget):

    def __init__(self, enabled_channels: list, map_list: list):
        super().__init__()
        self.enabled_channels = enabled_channels
        self.map_list = map_list

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.channels_stacks = {}

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.chb_save_detected = QCheckBox('Save detected')

        self.box_channels = QGroupBox('Channels')

        self.channels_list = QListWidget()
        self.channels_list.setFixedHeight(90)
        self.channels_list.setMinimumWidth(80)
        self.stack = QStackedWidget()

        i = 0
        for channel in self.enabled_channels:
            self.channels_list.insertItem(i, channel)
            stack_widget = SaveStack(channel_name=channel, map_list=self.map_list)
            self.channels_stacks[channel] = stack_widget
            self.stack.addWidget(stack_widget)
            i += 1
        self.channels_list.currentRowChanged.connect(self.show_current_stack)
        # self.channels_list.setMinimumWidth(100)

        self.btn_save = QPushButton('Start saving')
        self.btn_save.setCheckable(True)
        self.btn_save.setStyleSheet("font-size: 14px;")
        self.btn_save.clicked.connect(self.btn_save_clicked)

    def add_widgets_to_layout(self):
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        save_detected_layout = QHBoxLayout()
        save_detected_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        save_detected_layout.addWidget(self.chb_save_detected)

        box_layout = QVBoxLayout()
        box_layout.addWidget(self.channels_list, alignment=Qt.AlignmentFlag.AlignTop)
        self.box_channels.setLayout(box_layout)

        central_layout = QHBoxLayout()
        central_layout.addWidget(self.box_channels, alignment=Qt.AlignmentFlag.AlignTop)
        central_layout.addWidget(self.stack, alignment=Qt.AlignmentFlag.AlignTop)

        self.main_layout.addLayout(save_detected_layout)
        self.main_layout.addSpacing(20)
        self.main_layout.addLayout(central_layout)
        self.main_layout.addWidget(self.btn_save)

    def show_current_stack(self, i):
        self.stack.setCurrentIndex(i)

    def btn_save_clicked(self):
        if not self.btn_save.isChecked():
            self.btn_save.setText('Start saving')
        else:
            self.btn_save.setText('Stop saving')


class SaveStack(QWidget):
    def __init__(self, channel_name: str, map_list: list):
        super().__init__()
        self.name = channel_name
        self.map_list = map_list

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.checkboxes = {}

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.box_classes = QGroupBox(f'Classes to save ({self.name})')
        self.box_classes.setMinimumWidth(170)
        checkbox_all_classes = QCheckBox('All images')
        checkbox_all_classes.stateChanged.connect(self.chb_all_classes_clicked)
        self.checkboxes['All'] = checkbox_all_classes
        for clas_name in self.map_list:
            checkbox = QCheckBox(clas_name)
            self.checkboxes[clas_name] = checkbox
        checkbox_clear = QCheckBox('Clear')
        self.checkboxes['Clear'] = checkbox_clear

    def add_widgets_to_layout(self):
        classes_layout = QVBoxLayout()
        classes_layout.setSpacing(0)
        for chb in self.checkboxes.values():
            chb_layout = QHBoxLayout()
            chb_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            chb_layout.addWidget(chb)
            classes_layout.addLayout(chb_layout)

        central_layout = QHBoxLayout()
        central_layout.addLayout(classes_layout)
        self.box_classes.setLayout(central_layout)
        self.main_layout.addWidget(self.box_classes, alignment=Qt.AlignmentFlag.AlignLeft)
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addSpacerItem(spacer)

    def chb_all_classes_clicked(self, state: int):
        if bool(state):
            for chb in self.checkboxes.values():
                chb.setChecked(True)
        else:
            for chb in self.checkboxes.values():
                chb.setChecked(False)


class SoundTab(QWidget):
    signal_sound_states = pyqtSignal(dict)
    signal_sound_classes_states = pyqtSignal(dict)

    def __init__(self, enabled_channels: list, config: dict, logger_):
        super().__init__()
        self.enabled_channels = enabled_channels
        self.config = config
        self.sound_status = self.config['settings_sound']['sound_status']
        self.input_sound_name = self.config['settings_sound']['sound_name']
        self.map_list = self.config['map_list']

        self.logger = logger_
        self.sound_path = 'assets/sounds'
        self.sound_states = {}
        self.sound_classes_states = {}

        for channel in self.enabled_channels:
            self.sound_states[channel] = self.sound_status
        for name in self.map_list:
            self.sound_classes_states[name] = self.config['settings_sound']['classes_sound'][name]

        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.chb_sound = QCheckBox('Enable sound')
        self.chb_sound.setCheckable(True)
        self.chb_sound.setChecked(self.sound_status)
        self.chb_sound.stateChanged.connect(self.chb_sound_clicked)

        self.box_type_of_sound = QGroupBox('Type of sound')
        self.cb_sound = QComboBox()
        all_sounds = self.get_all_sounds()
        for sound_name in all_sounds:
            self.cb_sound.addItem(sound_name)
        if self.input_sound_name in all_sounds:
            self.cb_sound.setCurrentText(self.input_sound_name)
        else:
            self.logger.warning(f'Unknown sound {self.input_sound_name}. {self.cb_sound.currentText()} is set to default')
        self.btn_play_sound = QPushButton()
        self.btn_play_sound.setIcon(QIcon('assets/icons/dark/play_sound.png'))

        self.box_channels_sound = QGroupBox('Channels sound')
        self.chb_channels_sound = {}
        for channel in self.enabled_channels:
            chb = QCheckBox(f'{channel} sound')
            chb.setChecked(self.sound_status)
            chb.stateChanged.connect(self.sound_states_changed)
            self.chb_channels_sound[channel] = chb

        self.box_classes_sound = QGroupBox('Classes sound')
        self.chb_classes_sound = {}
        for name in self.map_list:
            chb = QCheckBox(name)
            chb.setChecked(self.sound_classes_states[name])
            chb.stateChanged.connect(self.sound_classes_states_changed)
            self.chb_classes_sound[name] = chb

    def add_widgets_to_layout(self):
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        choose_sound_layout = QHBoxLayout()
        choose_sound_layout.addWidget(self.cb_sound, alignment=Qt.AlignmentFlag.AlignLeft)
        choose_sound_layout.addWidget(self.btn_play_sound, alignment=Qt.AlignmentFlag.AlignLeft)
        self.box_type_of_sound.setLayout(choose_sound_layout)

        chb_channels_sound_layout = QVBoxLayout()
        chb_channels_sound_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        chb_channels_sound_layout.setContentsMargins(10, 10, 40, 10)
        for chb in self.chb_channels_sound.values():
            chb_channels_sound_layout.addWidget(chb, alignment=Qt.AlignmentFlag.AlignLeft)
        self.box_channels_sound.setLayout(chb_channels_sound_layout)

        chb_classes_sound_layout = QVBoxLayout()
        chb_classes_sound_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        chb_classes_sound_layout.setContentsMargins(10, 10, 40, 10)
        for chb in self.chb_classes_sound.values():
            chb_classes_sound_layout.addWidget(chb, alignment=Qt.AlignmentFlag.AlignLeft)
        self.box_classes_sound.setLayout(chb_classes_sound_layout)

        central_layout = QHBoxLayout()
        central_layout.addWidget(self.box_channels_sound)
        central_layout.addWidget(self.box_classes_sound)

        self.main_layout.addWidget(self.chb_sound, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addLayout(central_layout)
        self.main_layout.addWidget(self.box_type_of_sound, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addSpacerItem(spacer)

    def chb_sound_clicked(self, state: int):
        if bool(state):
            for chb in self.chb_channels_sound.values():
                chb.setChecked(True)
        else:
            for chb in self.chb_channels_sound.values():
                chb.setChecked(False)

    def get_all_sounds(self):
        files = []
        for (dirpath, dirnames, filenames) in walk(self.sound_path):
            files.extend(filenames)
        return files

    def sound_states_changed(self):
        for channel in self.enabled_channels:
            self.sound_states[channel] = (self.chb_channels_sound[channel].checkState() == Qt.CheckState.Checked)
        self.signal_sound_states.emit(self.sound_states)

    def sound_classes_states_changed(self):
        for name in self.map_list:
            self.sound_classes_states[name] = (self.chb_classes_sound[name].checkState() == Qt.CheckState.Checked)
        self.signal_sound_classes_states.emit(self.sound_classes_states)

    def collect_config(self):
        sound_conf = {}
        for name in self.map_list:
            sound_conf[name] = self.chb_classes_sound[name].isChecked()

        config = {'settings_sound': {'sound_status': self.chb_sound.isChecked(),
                                     'classes_sound': sound_conf,
                                     'sound_name': self.cb_sound.currentText()
                                     }
                  }
        return config


class NNTab(QWidget):

    def __init__(self, logger_, enabled_channels: list):
        super().__init__()
        self.logger = logger_
        self.enabled_channels = enabled_channels

        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.box_versions = QGroupBox('Models info')
        self.model_ver_dict = {}
        for channel in self.enabled_channels:
            self.model_ver_dict[channel] = QLabel(f'{channel}:\tUnknown')

        self.btn_get_nn_info = QPushButton('Get NN info')

    def add_widgets_to_layout(self):
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        box_layout = QVBoxLayout()
        box_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        box_layout.setContentsMargins(15, 15, 15, 10)
        for label in self.model_ver_dict.values():
            box_layout.addWidget(label)
        box_layout.addSpacing(10)
        box_layout.addWidget(self.btn_get_nn_info)

        self.box_versions.setLayout(box_layout)
        self.main_layout.addWidget(self.box_versions, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addSpacerItem(spacer)

    def update_models_info(self, nn_info: dict):
        for channel in nn_info.keys():
            self.model_ver_dict[channel].setText(f"{channel}:\t{nn_info[channel]['name']} ({nn_info[channel]['version']})")


class AlinxTab(QWidget):
    signal_autoscan_state = pyqtSignal(bool)
    signal_set_central_freq = pyqtSignal(float)

    def __init__(self, enabled_channels_info, autoscan, logger_):
        super().__init__()
        self.enabled_channels_info = enabled_channels_info
        self.autoscan_state = autoscan
        self.logger = logger_
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.box_soft = QGroupBox('Software')
        self.l_soft_ver = QLabel('Software version: ')
        self.l_curr_soft_ver = QLabel('Unknown')
        self.btn_get_soft_ver = QPushButton('Get software version')

        self.box_loadDetect = QGroupBox('Load Detect')
        self.l_loadDetect = QLabel('Load Detect state: ')
        self.l_curr_loadDetect_ver = QLabel('Unknown')
        self.btn_get_load_detect = QPushButton('Get Load Detect state')

        self.box_rx_settings = QGroupBox('RX Settings')
        self.l_central_freq = QLabel('FCM Central frequency')
        self.chb_autoscan = QCheckBox('Autoscan frequency')
        self.chb_autoscan.setChecked(self.autoscan_state)
        self.chb_autoscan.stateChanged.connect(self.chb_autoscan_clicked)
        self.cb_central_freq = QComboBox()
        self.cb_central_freq.setDisabled(False)
        self.cb_central_freq.currentTextChanged.connect(self.cb_central_freq_changed)
        for channel in self.enabled_channels_info:
            if len(channel.central_freq) > 1:
                i = 0
                for freq in channel.central_freq:
                    self.cb_central_freq.addItem(f'{freq/1_000_000:.1f} MHz', freq)
                    if freq == 5_786_500_000:
                        self.cb_central_freq.setCurrentIndex(i)
                    i += 1
        self.l_attenuation_24 = QLabel('Gain 2G4')
        self.spb_gain_24 = QSpinBox()
        self.spb_gain_24.setRange(0, 31)
        self.spb_gain_24.setSingleStep(1)
        self.spb_gain_24.setValue(31)
        self.spb_gain_24.setSuffix(' dB')
        self.l_attenuation_58 = QLabel('Gain 5G8')
        self.spb_gain_58 = QSpinBox()
        self.spb_gain_58.setRange(0, 31)
        self.spb_gain_58.setSingleStep(1)
        self.spb_gain_58.setValue(31)
        self.spb_gain_58.setSuffix(' dB')

    def add_widgets_to_layout(self):
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        l_soft_layout = QHBoxLayout()
        l_soft_layout.addWidget(self.l_soft_ver, alignment=Qt.AlignmentFlag.AlignLeft)
        l_soft_layout.addWidget(self.l_curr_soft_ver, alignment=Qt.AlignmentFlag.AlignLeft)
        soft_layout = QVBoxLayout()
        soft_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        soft_layout.setContentsMargins(15, 15, 15, 10)
        soft_layout.addLayout(l_soft_layout)
        soft_layout.addSpacing(10)
        soft_layout.addWidget(self.btn_get_soft_ver)
        self.box_soft.setLayout(soft_layout)

        l_load_detect_layout = QHBoxLayout()
        l_load_detect_layout.addWidget(self.l_loadDetect, alignment=Qt.AlignmentFlag.AlignLeft)
        l_load_detect_layout.addWidget(self.l_curr_loadDetect_ver, alignment=Qt.AlignmentFlag.AlignLeft)
        load_detect_layout = QVBoxLayout()
        load_detect_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        load_detect_layout.setContentsMargins(15, 15, 15, 10)
        load_detect_layout.addLayout(l_load_detect_layout)
        load_detect_layout.addSpacing(10)
        load_detect_layout.addWidget(self.btn_get_load_detect)
        self.box_loadDetect.setLayout(load_detect_layout)

        l_central_freq_layout = QVBoxLayout()
        l_central_freq_layout.addWidget(self.l_central_freq)
        l_central_freq_layout.addWidget(self.cb_central_freq)
        l_attenuation_24_layout = QVBoxLayout()
        l_attenuation_24_layout.addWidget(self.l_attenuation_24)
        l_attenuation_24_layout.addWidget(self.spb_gain_24)
        l_attenuation_58_layout = QVBoxLayout()
        l_attenuation_58_layout.addWidget(self.l_attenuation_58)
        l_attenuation_58_layout.addWidget(self.spb_gain_58)
        rx_settings_layout = QVBoxLayout()
        rx_settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        rx_settings_layout.setContentsMargins(15, 15, 15, 10)
        rx_settings_layout.addWidget(self.chb_autoscan)
        rx_settings_layout.addLayout(l_central_freq_layout)
        rx_settings_layout.addLayout(l_attenuation_24_layout)
        rx_settings_layout.addLayout(l_attenuation_58_layout)
        self.box_rx_settings.setLayout(rx_settings_layout)

        first_line = QHBoxLayout()
        first_line.setAlignment(Qt.AlignmentFlag.AlignTop)
        first_line.addWidget(self.box_soft)
        first_line.addWidget(self.box_rx_settings)

        self.main_layout.addLayout(first_line)
        self.main_layout.addWidget(self.box_loadDetect, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addSpacerItem(spacer)

    def update_soft_ver(self, message: str):
        self.l_curr_soft_ver.setText(message)

    def update_load_detect_state(self, message: str):
        self.l_curr_loadDetect_ver.setText(message)

    def chb_autoscan_clicked(self, state: int):
        if bool(state):
            self.cb_central_freq.setDisabled(True)
            self.signal_autoscan_state.emit(True)
        else:
            self.cb_central_freq.setDisabled(False)
            self.signal_autoscan_state.emit(False)

    def update_cb_central_freq(self, freq: int):
        self.cb_central_freq.setCurrentText(f'{freq / 1_000_000:.1f} MHz')

    def cb_central_freq_changed(self):
        if not self.chb_autoscan.isChecked():
            freq = self.cb_central_freq.currentData()
            print(f'change on {freq}')
            self.signal_set_central_freq.emit(freq / 1_000_000)

    def collect_config(self):
        config = {'fcm': {'autoscan_frequency': self.chb_autoscan.isChecked()}}
        return config


class USRPTab(QWidget):

    signal_central_freq_changed = pyqtSignal(str, float)

    def __init__(self, enabled_channels: list, logger_):
        super().__init__()
        self.enabled_channels = enabled_channels
        self.logger = logger_

        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)

        self.channels_stacks = {}

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.chb_autoscan = QCheckBox('Auto rebuild frequency')
        self.chb_autoscan.stateChanged.connect(self.chb_autoscan_changed)

        self.box_channels = QGroupBox('Channels')

        self.channels_list = QListWidget()
        self.stack = QStackedWidget()

        i = 0
        for channel in self.enabled_channels:
            self.channels_list.insertItem(i, channel)
            stack_widget = FrequencyStack(channel_name=channel, central_freq=0)
            stack_widget.spb_freq.valueChanged.connect(lambda:
                                                       self.central_freq_changed(channel, stack_widget.spb_freq.value()))
            self.channels_stacks[channel] = stack_widget
            self.stack.addWidget(stack_widget)
            i += 1
        self.channels_list.currentRowChanged.connect(self.show_current_stack)

    def add_widgets_to_layout(self):
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        box_layout = QVBoxLayout()
        box_layout.addWidget(self.channels_list, alignment=Qt.AlignmentFlag.AlignTop)
        self.box_channels.setLayout(box_layout)

        central_layout = QHBoxLayout()
        central_layout.addWidget(self.box_channels, alignment=Qt.AlignmentFlag.AlignTop)
        central_layout.addWidget(self.stack, alignment=Qt.AlignmentFlag.AlignTop)

        self.main_layout.addWidget(self.chb_autoscan, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addSpacing(20)
        self.main_layout.addLayout(central_layout)

        self.main_layout.addItem(spacer)

    def chb_autoscan_changed(self, state: int):
        if bool(state):
            for stack in self.channels_stacks.values():
                stack.spb_freq.setDisabled(True)
        else:
            for stack in self.channels_stacks.values():
                stack.spb_freq.setDisabled(False)

    def show_current_stack(self, i):
        self.stack.setCurrentIndex(i)

    def update_channel_freq(self, cnannel_freq: dict):
        band = cnannel_freq['band_name']
        freq = cnannel_freq['central_freq']
        freq_mhz = freq / 1e6
        self.channels_stacks[band].spb_freq.setValue(freq_mhz)

    def central_freq_changed(self, channel: str, value: float):
        # freq = int(value * 1_000_000)
        print('central_freq_changed', channel, value)
        self.signal_central_freq_changed.emit(channel, value)       # send frequency in MHz


class FrequencyStack(QWidget):
    def __init__(self, channel_name: str, central_freq: int):
        super().__init__()
        self.name = channel_name
        self.central_freq = central_freq

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.box_freq = QGroupBox(f'Central frequency ({self.name})')
        self.box_freq.setMinimumWidth(170)

        self.spb_freq = QDoubleSpinBox()
        self.spb_freq.setDecimals(1)
        self.spb_freq.setRange(300, 6000)
        self.spb_freq.setSingleStep(10)
        self.spb_freq.setValue(self.central_freq)
        self.spb_freq.setSuffix(' MHz')

    def add_widgets_to_layout(self):
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        box_layout = QVBoxLayout()
        box_layout.addWidget(self.spb_freq)
        self.box_freq.setLayout(box_layout)
        self.main_layout.addWidget(self.box_freq, alignment=Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addSpacerItem(spacer)


if __name__ == '__main__':
    app = QApplication([])
    qdarktheme.setup_theme()
    window = SettingsWidget(enabled_channels=['2G4', '1G2', '5G8'],
                            config={'map_list': ['Autel', 'Fpv', 'Dji', 'WiFi'],
                                    'show_images': True,
                                    'show_histogram': False,
                                    'show_spectrum': False,
                                    'sound_status': False,},
                            logger_=loguru.logger)
    window.show()
    app.exec()
