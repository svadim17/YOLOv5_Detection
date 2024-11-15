from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal


class Processor(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.recogn_widgets = {}

    def init_recogn_widgets(self, recogn_widgets: dict):
        self.recogn_widgets = recogn_widgets

    def parsing_data(self, info: dict):
        """
        info:  {band_name: value
                    drones: [{name: value, state: value, freq: value}, {-//-}, {-//-}, {-//-}]
                    img: value}
        """
        band_name = info['band_name']
        # del info['band_name']           # удаление ненужного ключа со значением в словаре
        self.recogn_widgets[band_name].update_state(info)     # обновление кнопок в конкретном канале
