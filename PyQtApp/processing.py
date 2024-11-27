from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal
import os
import datetime
import cv2
from loguru import logger
import numpy as np


class Processor(QtCore.QObject):
    def __init__(self, logger_):
        super().__init__()
        self.logger_ = logger_
        self.recogn_widgets = {}

        if not os.path.isdir('saved_images'):
            os.mkdir('saved_images')

    def init_recogn_widgets(self, recogn_widgets: dict):
        self.recogn_widgets = recogn_widgets

    def parsing_data(self, info: dict, save_detected: bool):
        """         info:  {band_name: value
                            drones: [{name: value, state: value, freq: value}, {-//-}, {-//-}, {-//-}]
                            detected_img: value
                            clear_img: value}           """

        band_name = info['band_name']
        self.recogn_widgets[band_name].update_state(info)     # обновление кнопок в конкретном канале

        if 'clear_img' in info:
            if save_detected:
                self.save_images(img_bytes=info['clear_img'], img_detected_bytes=info['detected_img'])
            else:
                self.save_images(img_bytes=info['clear_img'])

    def save_images(self, img_bytes, img_detected_bytes=None):
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape((640, 640, 3))
        filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f_clear')
        if not cv2.imwrite(filename='saved_images' + '\\' + filename + '.jpg', img=img_arr):
            self.logger_.error('Error with saving images!')

        if img_detected_bytes is not None:
            img_detected_arr = np.frombuffer(img_detected_bytes, dtype=np.uint8).reshape((640, 640, 3))
            filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f_detected')
            if not cv2.imwrite(filename='saved_images' + '\\' + filename + '.jpg', img=img_detected_arr):
                self.logger_.error('Error with saving detected images!')
