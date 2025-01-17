import socket
import time
from multiprocessing import Process
import numpy as np
from loguru import logger
import yaml
from collections.abc import Mapping


logger.remove(0)
log_level = "DEBUG"
log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |"
              " <level>{level: <8}</level> | {extra} | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
# logger.add(sys.stderr, format=log_format, colorize=True, backtrace=True, diagnose=True)
logger.add("logs/file_{time}.log",
           level=log_level,
           format=log_format,
           colorize=False,
           backtrace=True,
           diagnose=True,
           rotation='10 MB',
           enqueue=True,
           )


class Client(Process):
    def __init__(self, address, config, logger_):
        super().__init__()
        self.logger = logger_
        self.address = address
        self.port = self.address[1]
        self.start_time = time.time()
        self.save_img_freq = []
        self.config_full = config
        try:
            self.config = config[self.port]
        except KeyError:
            self.config = config['default']
            logger.warning(f'Unknown port {self.port}! Loaded default configuration.')
        self.fft_size = self.config['width']
        self.numb_of_slices = self.config['height']

    def start_nn(self):
        try:
            from nn_processing import NNProcessing
            self.logger.debug('default import')
        except ImportError:
            import os, sys
            sys.path.append(os.path.abspath('../'))
            from nn_processing import NNProcessing
            self.logger.debug('sys import')

        try:
            self.project_path = self.config['project_path']
            self.weights_path = self.config['weights_path']
            self.map_list = self.config['map_list']
            self.z_min = self.config['z_min']
            self.z_max = self.config['z_max']
            self.colormap = self.config['colormap']
            self.width = self.config['width']
            self.height = self.config['height']

            self.nn = NNProcessing(name=str(self.address),
                                   weights=self.weights_path,
                                   project_path=self.project_path,
                                   map_list=tuple(self.map_list),
                                   z_min=self.z_min,
                                   z_max=self.z_max,
                                   colormap=self.colormap,
                                   width=self.width,
                                   height=self.height)
            self.nn.img_save_path = self.config['save_img_path']
            self.logger.success(f'NNProcessing initialized! Use device {self.nn.device}')
        except Exception as e:
            self.logger.error(f'NNProcessing did not initialize! \n{e}')

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
                self.logger.debug(f'Save images status is {status}')

    def set_zscale(self, data):
        z_min = int.from_bytes(data[:2], byteorder='big', signed=True)
        z_max = int.from_bytes(data[2:4], byteorder='big', signed=True)
        self.nn.z_min, self.z_min = z_min, z_min
        self.nn.z_max, self.z_max = z_max, z_max

        self.logger.debug(f'New ZScale: [{z_min}, {z_max}]')

    def set_imgshow_status(self, data):
        status = bool(data[0])
        self.nn.show_detected_img_status = status
        self.logger.debug(f'Image show status is {status}')

    def set_spectrogram_size(self, data):
        fft_size = int.from_bytes(data[:2], byteorder='big')
        numb_of_slices = int.from_bytes(data[2:4], byteorder='big')
        self.fft_size = fft_size
        self.numb_of_slices = numb_of_slices
        self.logger.debug(f'New spectrogram size is {fft_size} x {numb_of_slices}')

    def save_config(self):
        full_config = load_config(config_path='config.yaml')

        info = {self.port: {'project_path': self.project_path,
                            'weights_path': self.nn.weights,
                            'save_img_path': self.nn.img_save_path,
                            'map_list': list(self.nn.map_list),
                            'z_min': self.z_min,
                            'z_max': self.z_max,
                            'colormap': self.nn.colormap,
                            'width': self.nn.width,
                            'height': self.nn.height}}
        self.logger.debug(f'Data for saving in config: {info}')
        self.config = deep_update(full_config, info)
        dump_conf('config.yaml', self.config)
        self.logger.debug('Config saved!')

    def get_current_zscale(self):
        zmin = self.nn.z_min.to_bytes(length=2, byteorder='big', signed=True)
        zmax = self.nn.z_max.to_bytes(length=2, byteorder='big', signed=True)
        self.socket.send(zmin + zmax)
        self.logger.debug(f'Current ZScale is [{self.nn.z_min}, {self.nn.z_max}]')

    def calculate_frequency(self, ymin, ymax):
        """ Функция считает частоту сигнала относительно центральной частоты (нулевой) и возвращает смещение """
        freq_min = (self.sample_rate / self.img_size[0]) * ymin
        freq_max = (self.sample_rate / self.img_size[0]) * ymax
        freq_width = freq_max - freq_min
        f = freq_width / 2 + freq_min
        f_center = (self.sample_rate / 2 - f) * (-1)
        return f_center

    def recognition(self, data, msg_len):
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
            freq_bytes = freq.tobytes()
            res_list_bytes = res_list.tobytes()
            info_len = len(freq_bytes + res_list_bytes).to_bytes(4, 'big')

            self.socket.send(b'\x45' + info_len + freq_bytes + res_list_bytes)
            self.logger.trace(f'Freq {freq_int} Result: {res_list}')

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.socket:
            # self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, b"eth0\0")
            self.start_nn()
            for _ in range(4):
                try:
                    self.socket.connect(self.address)
                    self.logger.success(f'Connected to {self.address}!')
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
                            self.logger.trace(f'Packet {i} missed.')
                        if msg_len > len(data) and i == 40:
                            self.logger.warning(f'TCP timeout error: could not wait for the end of the parcel.')

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
                            case 6:
                                self.save_config()
                            case 69:
                                self.recognition(data, msg_len)
                            case _:
                                error_text = f'Unknown command {command_type}!'
                                self.socket.send(b'\xff' +
                                                 len(error_text).to_bytes(4, 'big') +
                                                 str.encode(error_text)
                                                 )
                                self.logger.warning(error_text)

                except socket.error as e:
                    self.logger.error(f'TCP Error: {e}')
                    self.socket.close()
                    time.sleep(1)

                except Exception as e:
                    str_error_bytes = str.encode(str(e))
                    self.socket.send(b'\xff' +                                                       # шифр
                                     len(str_error_bytes).to_bytes(4, 'big') +       # error text len
                                     str_error_bytes)                                                # error tex
                    self.logger.critical(e)

            self.logger.info(f'Port №{self.address} finished work')


def load_config(config_path):
    with open(config_path, encoding='utf-8') as f:
        try:
            config = dict(yaml.load(f, Loader=yaml.SafeLoader))
            logger.success(f'Config loaded successfully!')
            return config
        except Exception as e:
            logger.error(f'Error of reading config {config_path}\n {e}')


def deep_update(source: dict, overrides: dict):
    for key, value in overrides.items():
        if isinstance(value, Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def dump_conf(config_path: str, config: dict):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)


def main(ip, ports):
    config = load_config(config_path='config.yaml')
    processes = []
    # PORTS = [6345, 6346]
    for port in ports:
        try:
            cl = Client((ip, int(port)), config, logger.bind(port=port))
        except Exception as e:
            logger.error(f'Error with initializing Client! \n{e}')
        cl.start()
        processes.append(cl)
    for process in processes:
        process.join()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2:])
    else:
        logger.error('Error, enter ports for sockets as arguments')
