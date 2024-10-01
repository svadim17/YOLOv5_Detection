import datetime
import socket
import time
from multiprocessing import Process
import pandas
import torch.nn.functional as F
import cv2
import torch
import numpy as np
from loguru import logger
import yaml
import sys


logger.remove(0)
log_level = "TRACE"
log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {extra} | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
# logger.add(sys.stderr, format=log_format, colorize=True, backtrace=True, diagnose=True)
logger.add("file_{time}.log",
           level=log_level,
           format=log_format,
           colorize=False,
           backtrace=True,
           diagnose=True,
           rotation='200 kB',
           enqueue=True,
           )


class Client(Process):
    def __init__(self, address, config, logger_):
        super().__init__()
        self.logger_ = logger_
        self.address = address
        self.start_time = time.time()
        self.fft_size = 1024
        self.numb_of_slices = 2048
        self.save_img_freq = []
        self.config = config

    def start_nn(self):
        try:
            from nn_processing import NNProcessing
            self.logger_.debug('default import')
        except ImportError:
            import os
            sys.path.append(os.path.abspath('../'))
            from nn_processing import NNProcessing
            self.logger_.debug('sys import')

        try:
            self.project_path = self.config['project_path']
            self.weights_path = self.config['weights_path']
            self.map_list = self.config['map_list']
            self.z_min = self.config['z_min']
            self.z_max = self.config['z_max']

            self.nn = NNProcessing(name=str(self.address),
                                   weights=self.weights_path,
                                   project_path=self.project_path,
                                   map_list=tuple(self.map_list),
                                   z_min=self.z_min,
                                   z_max=self.z_max,)
            # self.nn.show_detected_img_status = True
            self.nn.img_save_path = self.config['save_img_path']
            self.logger_.success(f'NNProcessing initialized!')
        except Exception as e:
            self.logger_.error(f'NNProcessing did not initialize! \n{e}')

    def set_save_img_status(self, data):
        status = bool(data[0])
        central_freq = int.from_bytes(data[1:9], byteorder='big')
        if central_freq in self.save_img_freq:
            if status:
                pass
            else:
                self.save_img_freq.remove(central_freq)
        else:
            if status:
                self.save_img_freq.append(central_freq)
            else:
                pass
            self.logger_.debug(f'Save images status is {status}')

    def set_zscale(self, data):
        z_min = int.from_bytes(data[:2], byteorder='big', signed=True)
        z_max = int.from_bytes(data[2:4], byteorder='big', signed=True)
        self.nn.z_min = z_min
        self.nn.z_max = z_max
        self.logger_.debug(f'New ZScale: [{z_min}, {z_max}]')

    def set_imgshow_status(self, data):
        status = bool(data[0])
        self.nn.show_detected_img_status = status
        self.logger_.debug(f'Image show status is {status}')

    def set_spectrogram_size(self, data):
        fft_size = int.from_bytes(data[:2], byteorder='big')
        numb_of_slices = int.from_bytes(data[2:4], byteorder='big')
        self.fft_size = fft_size
        self.numb_of_slices = numb_of_slices
        self.logger_.debug(f'New spectrogram size is {fft_size} x {numb_of_slices}')

    def get_current_zscale(self):
        zmin = self.nn.z_min.to_bytes(length=2, byteorder='big', signed=True)
        zmax = self.nn.z_max.to_bytes(length=2, byteorder='big', signed=True)
        self.socket.send(zmin + zmax)
        self.logger_.debug(f'Current ZScale is [{self.nn.z_min}, {self.nn.z_max}]')

    def calculate_frequency(self, ymin, ymax):
        """ Функция считает частоту сигнала относительно центральной частоты (нулевой) и возвращает смещение """
        freq_min = (self.sample_rate / self.img_size[0]) * ymin
        freq_max = (self.sample_rate / self.img_size[0]) * ymax
        freq_width = freq_max - freq_min
        f = freq_width / 2 + freq_min
        f_center = (self.sample_rate / 2 - f) * (-1)
        return f_center

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.socket:
            self.start_nn()
            for _ in range(4):
                try:
                    self.socket.connect(self.address)
                    self.logger_.success(f'Connected to {self.address}!')
                except socket.error as msg:
                    self.logger_.error(msg)
                    time.sleep(1)
                    continue
                while True:
                    header = self.socket.recv(5)
                    command_type = header[0]
                    msg_len = int.from_bytes(header[1:], byteorder='big')
                    data = self.socket.recv(msg_len)
                    i = 0
                    while msg_len > len(data) and i < 50:
                        time.sleep(0.01)
                        data += self.socket.recv(msg_len - len(data))
                        i += 1
                    if i:
                        self.logger_.trace(f'Packet {i} missed.')
                    if msg_len > len(data) and i == 40:
                        self.logger_.warning(f'TCP timeout error: could not wait for the end of the parcel.')

                    match command_type:
                        case 1:
                            self.set_save_img_status(data)
                        case 2:
                            self.set_zscale(data)
                        case 3:
                            self.set_imgshow_status(data)
                        case 4:
                            self.set_spectrogram_size(data)
                        case 5:
                            self.get_current_zscale()
                        case 69:
                            try:
                                np_arr = np.frombuffer(data, dtype=np.int8)
                                if np_arr.size == msg_len:
                                    freq = np_arr[0:8]
                                    freq_int = int.from_bytes(freq, byteorder='big')
                                    if freq_int in self.save_img_freq:
                                        save_img = True
                                    else:
                                        save_img = False
                                    spectrogram = np_arr[8:].reshape(self.numb_of_slices, self.fft_size)
                                    norm_spectrogram = self.nn.normalization(spectrogram)
                                    result = self.nn.processing(norm_data=norm_spectrogram,
                                                                save_images=save_img)

                                    df_result = result.pandas().xyxy[0]

                                    res_list = self.nn.convert_result(df_result, return_data_type='list')
                                    # res_dict = self.nn.convert_result(df_result, return_data_type='dict')
                                    self.socket.send(freq.tobytes() + res_list.tobytes())
                                    self.logger_.trace(f'Freq {freq_int} Result: {res_list}')
                            except socket.error as e:
                                self.logger_.error(f'TCP Error: {e}')
                                break
                            except Exception as e:
                                self.logger_.critical(e)
                        case _:
                            self.socket.send(b'\x01')
                            self.logger_.warning(f'Unknown command {command_type}!')
            self.logger_.info(f'Port №{self.address} finished work')


def load_config(config_path):
    with open(config_path, encoding='utf-8') as f:
        try:
            config = dict(yaml.load(f, Loader=yaml.SafeLoader))
            logger.success(f'Config loaded successfully!')
            return config
        except Exception as e:
            logger.error(f'Error of reading config {config_path}\n {e}')


def main(ip, PORTS):
    config = load_config(config_path='config.yaml')
    processes = []
    # PORTS = [6345, 6346]
    for i in PORTS:
        try:
            cl = Client((ip, int(i)), config[int(i)], logger.bind(port=i))
        except KeyError:
            cl = Client((ip, int(i)), config['default'], logger.bind(port=i))
            logger.warning(f'Unknown port {i}! Loaded default configuration.')
        cl.start()
        processes.append(cl)
    for i in processes:
        i.join()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2:])
    else:
        logger.error('Error, enter ports for sockets as arguments')
