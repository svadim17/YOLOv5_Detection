from collections import deque
import queue
import os
import sys
import socket
import time
import numpy as np
from multiprocessing import Process, Queue, Pipe, Event
from loguru import logger

from PyQtApp import custom_utils
from PyQtApp.server_submodules.tcp_control_alinx import Task_


try:
    from nn_processing import NNProcessing
except ImportError:
    # Получаем путь на два уровня вверх
    sys.path.append(os.path.abspath('../'))
    from nn_processing import NNProcessing


class Client(Process):
    def __init__(self,
                 name: str,
                 address: tuple,
                 hardware_type: str,
                 receive_freq_status: bool,
                 central_freq: int,
                 freq_list: list,
                 model_version: str,
                 weights_path: str,
                 project_path: str,
                 sample_rate: int,
                 signal_width: int,
                 signal_height: int,
                 img_size: tuple,
                 all_classes: tuple,
                 map_list: tuple,
                 z_min: int,
                 z_max: int,
                 threshold: float,
                 accumulation_size: int,
                 exceedance: float,
                 data_queue=None,
                 error_queue=None,
                 FCM_control_queue=None,
                 task_done_event=None,
                 logger_=None):
        super().__init__()
        self.nn = None
        self.name = name
        self.address = address
        self.hardware = hardware_type
        self.receive_freq_status = receive_freq_status
        self.central_freq = central_freq
        self.freq_list = freq_list
        self.model_version = model_version
        self.weights_path = weights_path
        self.project_path = project_path
        self.sample_rate = sample_rate
        self.signal_width = signal_width
        self.signal_height = signal_height
        self.img_size = img_size
        self.all_classes = all_classes
        self.map_list = map_list
        self.start_time = time.time()
        self.z_min = z_min
        self.z_max = z_max
        self.threshold = threshold
        self.accumulation_size = accumulation_size
        self.current_accum_index = 0
        self.exceedance = exceedance

        self.accum_status = True
        self.global_threshold = self.threshold * self.accumulation_size * self.exceedance
        self.pipe_control_child, self.pipe_control_parent = Pipe()
        self.control_q = Queue()
        self.config_q = Queue()
        self.data_q = data_queue
        self.error_queue = error_queue
        self.FCM_control_queue = FCM_control_queue
        self.task_done_event = task_done_event
        self.img_save_path = '\\saved_images'
        if not os.path.isdir(self.img_save_path):
            os.mkdir(self.img_save_path)

        if logger_ is None:
            self.logger = logger.bind(logger_name=self.name)
        else:
            self.logger = logger_

        if self.hardware == 'USRP' or self.hardware == 'Usrp' or self.hardware == 'usrp':
            self.msg_len = self.signal_width * self.signal_height
            if self.receive_freq_status:
                self.freq_len = 8
        elif self.hardware == 'ALINX' or self.hardware == 'Alinx' or self.hardware == 'alinx':
            self.msg_len = self.signal_width * self.signal_height * 2 + 16
        else:
            self.logger.error(f'Unknown Device Type: {self.hardware} !')

        self.accum_deques = {freq: {i: deque(maxlen=self.accumulation_size) for i in self.map_list} for freq in
                             self.freq_list}
        self.record_images_status = False

    def set_queue(self, data_queue: Queue):
        self.data_q = data_queue

    def change_zscale(self, z_min: int, z_max: int):
        self.z_min = z_min
        self.z_max = z_max
        try:
            self.nn.set_z_min(value=z_min)
            self.nn.set_z_max(value=z_max)
            self.logger.debug(f'New z: {z_min} {z_max}')
        except Exception as e:
            self.logger.error(e)

    def get_current_settings(self):
        self.config_q.put({self.name: {'neural_network_settings': {'z_min': self.z_min, 'z_max': self.z_max},
                                       'detection_settings': {'accumulation_size': self.accumulation_size,
                                                              'threshold': self.threshold,
                                                              'exceedance': self.exceedance}
                                       }
                           })

    def change_recognition_settings(self, accumulation_size: int, threshold: float, exceedance: float):
        self.accumulation_size = accumulation_size
        self.threshold = threshold
        self.exceedance = exceedance
        self.accum_deques = {freq: {i: deque(maxlen=self.accumulation_size) for i in self.map_list} for freq in
                                                                                                        self.freq_list}
        self.global_threshold = self.threshold * self.accumulation_size * self.exceedance

    def accumulate_and_make_decision(self, result_dict: dict, central_freq: int):
        accumed_results = []
        frequencies = []

        for key, key_dict in result_dict.items():
            if self.accum_status:
                self.accum_deques[central_freq][key].appendleft(key_dict['confidence'])
                accum = sum(self.accum_deques[central_freq][key])
                state = bool(accum >= self.global_threshold)
            else:
                state = bool(key_dict['confidence'] > self.threshold)
            accumed_results.append(state)
            if state:
                try:
                    freq_shift = self.calculate_frequency(ymin=key_dict['ymin'], ymax=key_dict['ymax'])
                    freq = int((freq_shift + self.central_freq) / 1_000_000)
                except Exception as e:
                    freq = 0
                    self.logger.trace(f"Can\'t calculate frequency for {self.name}! \n{e}")
                frequencies.append(freq)
            else:
                frequencies.append(0)
        return accumed_results, frequencies

    def calculate_frequency(self, ymin, ymax):
        """ Функция считает частоту сигнала относительно центральной частоты (нулевой) и возвращает смещение """
        freq_min = (self.sample_rate / self.img_size[0]) * ymin
        freq_max = (self.sample_rate / self.img_size[0]) * ymax
        freq_width = freq_max - freq_min
        f = freq_width / 2 + freq_min
        f_center = (self.sample_rate / 2 - f) * (-1)
        return f_center

    def change_record_images_status(self, status: bool):
        self.record_images_status = status
        self.logger.info(f'Record images status was changed on {status}.')

    def change_accum_status(self, accum_status: bool):
        self.accum_status = accum_status
        self.logger.info(f'Accum status = {self.accum_status}')

    def average_spectrum(self, arr: np.ndarray):
        spectrum = np.mean(arr, axis=0).astype(np.float16)  # среднее по столбцам (axis=0)
        return spectrum

    def send_error(self, error_txt: str):
        try:
            if self.error_queue.full():
                self.logger.debug(f'error_queue is full.')
            else:
                self.error_queue.put({'status': True, 'msg': error_txt}, timeout=0.01)
        except queue.Full as e:
            self.logger.error(f'error_queue is full. {e}')

    def receive_from_Alinx(self, sock):
        self.change_frequency_Alinx()
        sock.send(b'\x30')
        arr = sock.recv(self.msg_len)
        self.logger.trace(f'Received {self.msg_len} bytes.')
        i = 0
        while self.msg_len > len(arr) and i < 500:
            i += 1
            time.sleep(0.005)
            arr += sock.recv(self.msg_len - len(arr))
            # self.logger.warning(f'Packet {i} missed.')
        if len(arr) == self.msg_len:
            np_arr = np.frombuffer(arr[16:], dtype=np.float16)
            np_arr = np_arr.astype(np.float32)
            arr_resh = np_arr.reshape(self.signal_height, self.signal_width)

            spectrum = self.average_spectrum(arr=arr_resh)
            img_arr = self.nn.normalization(np.fft.fftshift(arr_resh, axes=(1,)))
            return img_arr, np.fft.fftshift(spectrum)
        else:
            self.logger.trace(f'Packet size = {len(arr)} missed.')
            raise custom_utils.AlinxException()

    def receive_from_USRP(self, sock):
        sock.send(self.central_freq.to_bytes(length=8, byteorder='big'))
        if self.receive_freq_status:           # receive central freq of channel if status is true
            freq = sock.recv(self.freq_len)
            self.central_freq = int.from_bytes(freq, byteorder='big')

        arr = sock.recv(self.msg_len)
        self.logger.trace(f'Received {self.msg_len} bytes.')
        i = 0
        while self.msg_len > len(arr) and i < 500:
            i += 1
            time.sleep(0.005)
            arr += sock.recv(self.msg_len - len(arr))
            # self.logger.warning(f'Packet {i} missed.')
        np_arr = np.frombuffer(arr, dtype=np.int8)
        arr_resh = np_arr.reshape(self.signal_height, self.signal_width)

        spectrum = self.average_spectrum(arr=arr_resh)

        if np_arr.size == self.msg_len:
            img_arr = self.nn.normalization(arr_resh)
            return img_arr, spectrum
        else:
            self.logger.trace(f'Packet size = {np_arr.size} missed.')
            return None

    def change_frequency_Alinx(self):

        if len(self.freq_list) > 1 and self.current_accum_index % self.accumulation_size == 0:
            index = self.freq_list.index(self.central_freq) + 1
            freq = self.freq_list[index % len(self.freq_list)]
            self.FCM_control_queue.put(Task_(self.name, 'set_frequency', freq))
            self.logger.debug(f'Send to set {freq} Hz')
            if self.task_done_event.wait(11):
                self.central_freq = freq
                self.logger.debug(f'Freq. {self.central_freq} Hz was set')
                time.sleep(0.15)
            else:
                self.logger.error(f'Frequency has not been set')
                self.send_error(f'Frequency has not been set')

    #@logger.catch
    def run(self):
        try:
            self.nn = NNProcessing(name=self.name,
                                   all_classes=self.all_classes,
                                   model_version=self.model_version,
                                   model_weights=self.weights_path,
                                   sample_rate=self.sample_rate,
                                   width=self.signal_width,
                                   height=self.signal_height,
                                   project_path=self.project_path,
                                   map_list=self.map_list,
                                   img_size=self.img_size,
                                   msg_len=self.msg_len,
                                   z_min=self.z_min,
                                   z_max=self.z_max,
                                   colormap='inferno',
                                   img_save_path=self.img_save_path)
            self.logger.debug(f'NNProcessing started by {self.nn.device}')

        except Exception as e:
            self.logger.error(f'Cant initialize NNProcessing! {e}')
            self.send_error(f'Cant initialize NNProcessing! {e}')
            return

        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.settimeout(10)
                    s.connect(self.address)
                    self.logger.success(f'Connected to {self.address}!')

                    if self.hardware.lower() == 'usrp':
                        nn_type = s.recv(2)     # nn type
                        self.logger.debug(f'NN type: {nn_type}')
                        receive = self.receive_from_USRP

                    elif self.hardware.lower() == 'alinx':
                        receive = self.receive_from_Alinx

                    else:
                        self.logger.critical(f'Unknown Device Type: {self.hardware}')
                        self.send_error('Unknown Device Type!')
                        return

                    while True:
                        img_arr, spectrum = receive(sock=s)
                        if img_arr is not None:
                            clear_img, df_result, detected_img = self.nn.processing_for_grpc(img_arr)
                            self.current_accum_index += 1
                            if self.data_q is not None:
                                res, freq = self.accumulate_and_make_decision(
                                    self.nn.grpc_convert_result(df_result, return_data_type='dict_with_freq'),
                                    central_freq=self.central_freq)

                                try:
                                    if self.data_q.full():
                                        self.data_q.get()
                                    self.data_q.put({'name': self.name,
                                                     'results': res,
                                                     'frequencies': freq,
                                                     'predict_df': df_result,
                                                     'detected_img': detected_img,
                                                     'clear_img': clear_img,
                                                     'spectrum': spectrum,
                                                     'channel_freq': self.central_freq},
                                                    timeout=10)
                                except queue.Full as e:
                                    self.logger.error(f'data_queue is full. {e}')

                            if not self.control_q.empty():
                                cmd_dict = self.control_q.get()
                                args = cmd_dict['args']
                                func = getattr(self, cmd_dict['func'])
                                func(*args)

                except socket.error as e:
                    s.close()
                    self.logger.error(f'Connection error! {e}')
                    self.data_q.put(None)
                    self.send_error(f'Connection error! {e}')
                    time.sleep(2)
                except Exception as e:
                    s.close()
                    self.logger.error(f'Unknown error! {e}')
                    self.send_error(f'Unknown error! {e}')
                    break
        self.logger.debug(f'Channel {self.name} ({self.address}) finished work')

