from PyQt6.QtWidgets import (QWidget, QListWidget, QApplication, QSpacerItem, QSizePolicy,
                             QStackedWidget, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QTabWidget, QComboBox, QGroupBox)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtCore import pyqtSlot, Qt, pyqtSignal
import qdarktheme
from os import walk


class SettingsWidget(QWidget):
    def __init__(self, enabled_channels: list, config: dict, logger_):
        super().__init__()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle('Settings')
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.setLayout(self.main_layout)

        self.logger = logger_

        self.mainTab = MainTab(config=config)
        self.saveTab = SaveTab(enabled_channels=enabled_channels, map_list=config['map_list'])
        self.soundTab = SoundTab(enabled_channels=enabled_channels,
                                 sound_status=config['sound_status'],
                                 sound_name=config['sound_name'],
                                 map_list=config['map_list'],
                                 logger_=self.logger)

        self.tab = QTabWidget()
        self.tab.addTab(self.mainTab, 'Main')
        self.tab.addTab(self.saveTab, 'Images saving')
        self.tab.addTab(self.soundTab, 'Sound')

        self.main_layout.addWidget(self.tab)


class MainTab(QWidget):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
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

        self.chb_show_zscale = QCheckBox('Show ZScale settings')
        self.chb_show_zscale.setChecked(True)

        self.chb_show_frequencies = QCheckBox('Show frequencies')
        self.chb_show_frequencies.setChecked(True)

        self.chb_accumulation = QCheckBox('Enable accumulation')
        self.chb_accumulation.setChecked(True)

        self.chb_watchdog = QCheckBox('Watchdog')
        self.chb_watchdog.setChecked(self.config['watchdog'])

        self.btn_save_client_config = QPushButton('Save client config')

        self.btn_save_server_config = QPushButton('Save server config')

        self.box_widgets_states = QGroupBox('Widgets states')

        self.chb_img_show = QCheckBox('Show spectrogram')
        self.chb_img_show.setChecked(self.config['show_images'])

        self.chb_histogram_show = QCheckBox('Show histogram')
        self.chb_histogram_show.setChecked(self.config['show_histogram'])

        self.chb_spectrum_show = QCheckBox('Show spectrum')
        self.chb_spectrum_show.setChecked(self.config['show_spectrum'])

    def add_widgets_to_layout(self):
        spectr_layout = QVBoxLayout()
        spectr_layout.setSpacing(5)
        spectr_layout.addWidget(self.l_spectrogram_resolution, alignment=Qt.AlignmentFlag.AlignLeft)
        spectr_layout.addWidget(self.cb_spectrogram_resolution, alignment=Qt.AlignmentFlag.AlignLeft)

        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        controls_layout.addWidget(self.chb_show_zscale)
        controls_layout.addWidget(self.chb_show_frequencies)
        controls_layout.addWidget(self.chb_accumulation)
        controls_layout.addWidget(self.chb_watchdog)

        widgets_states_layout = QVBoxLayout()
        widgets_states_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        widgets_states_layout.addWidget(self.chb_img_show)
        widgets_states_layout.addWidget(self.chb_histogram_show)
        widgets_states_layout.addWidget(self.chb_spectrum_show)

        self.box_widgets_states.setLayout(widgets_states_layout)

        central_layout = QHBoxLayout()
        central_layout.addLayout(controls_layout)
        central_layout.addWidget(self.box_widgets_states)

        btns_layout = QHBoxLayout()
        btns_layout.addWidget(self.btn_save_client_config)
        btns_layout.addWidget(self.btn_save_server_config)

        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.main_layout.addLayout(spectr_layout)
        self.main_layout.addLayout(central_layout)
        self.main_layout.addSpacerItem(spacer)
        self.main_layout.addLayout(btns_layout)


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
        self.chb_save_detected = QCheckBox()
        self.l_save_detected = QLabel('Save detected')
        self.l_save_detected.setStyleSheet("font-size: 14px")

        self.channels_list = QListWidget()
        self.channels_list.setMinimumWidth(200)
        self.stack = QStackedWidget()

        i = 0
        for channel in self.enabled_channels:
            self.channels_list.insertItem(i, channel)
            stack_widget = SaveStack(channel_name=channel, map_list=self.map_list)
            self.channels_stacks[channel] = stack_widget
            self.stack.addWidget(stack_widget)
            i += 1
        self.channels_list.currentRowChanged.connect(self.show_current_stack)

        self.btn_save = QPushButton('Start saving')
        self.btn_save.setCheckable(True)
        self.btn_save.setStyleSheet("font-size: 14px;")
        self.btn_save.clicked.connect(self.btn_save_clicked)

    def add_widgets_to_layout(self):
        save_detected_layout = QHBoxLayout()
        save_detected_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        save_detected_layout.addWidget(self.chb_save_detected)
        save_detected_layout.addWidget(self.l_save_detected)

        central_layout = QHBoxLayout()
        central_layout.addWidget(self.channels_list)
        central_layout.addWidget(self.stack)

        self.main_layout.addLayout(save_detected_layout)
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
        self.l_stack = QLabel(self.name)
        self.l_stack.setStyleSheet("font-size: 16px;")

        self.l_classes = QLabel('Classes to save')
        self.l_classes.setStyleSheet("font-size: 14px")

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
        central_layout.addWidget(self.l_classes, alignment=Qt.AlignmentFlag.AlignTop)
        central_layout.addSpacing(50)
        central_layout.addLayout(classes_layout)

        self.main_layout.addWidget(self.l_stack, alignment=Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
        self.main_layout.addSpacing(20)
        self.main_layout.addLayout(central_layout)
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

    def __init__(self, enabled_channels: list, sound_status: bool, sound_name: str, map_list: list, logger_):
        super().__init__()
        self.enabled_channels = enabled_channels
        self.sound_status = sound_status
        self.input_sound_name = sound_name
        self.map_list = map_list
        self.logger = logger_
        self.sound_path = 'assets/sounds'
        self.sound_states = {}
        self.sound_classes_states = {}

        for channel in self.enabled_channels:
            self.sound_states[channel] = self.sound_status
        for name in self.map_list:
            self.sound_classes_states[name] = True

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
        self.btn_play_sound.setIcon(QIcon('assets/icons/play_sound.png'))

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
            chb.setChecked(True)
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


if __name__ == '__main__':
    app = QApplication([])
    qdarktheme.setup_theme()
    window = SettingsWidget(enabled_channels=['2G4', '1G2', '5G8'],
                            config={'map_list': ['Autel', 'Fpv', 'Dji', 'WiFi'],
                                    'show_images': True,
                                    'show_histogram': False,
                                    'show_spectrum': False,
                                    'sound_status': False})
    window.show()
    app.exec()
