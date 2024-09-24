import grpc
from concurrent import futures
import neuro_pb2_grpc as API_pb2_grpc
import neuro_pb2 as API_pb2
import socket
import time
from multiprocessing import Process, Queue
import cv2
import numpy as np
from loguru import logger
import copy
from collections import deque
from yolov5.nn_processing import NNProcessing
import yaml

# from nn_processing import NNProcessing


def load_conf(config_path: str):
    try:
        with open(config_path, encoding='utf-8') as f:
            config_str = str(yaml.load(f, Loader=yaml.SafeLoader))
            config = dict(yaml.load(config_str, Loader=yaml.SafeLoader))
            logger.success(f'Config loaded successfully!')
            return config
    except Exception as e:
        logger.error(f'Error with loading config: {e}')


def dump_conf(config_path:str, config: dict):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)


class Client(Process):
    def __init__(self,
                 name: str,
                 address: tuple,
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
                 threshold: int,
                 accumulation_size: int,
                 data_queue=None,
                 img_queue=None):
        super().__init__()
        self.q_control = Queue()
        self.nn = None
        self.name = name
        self.address = address
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
        self.data_q = data_queue
        self.img_q = img_queue
        self.msg_len = self.signal_width * self.signal_height
        self.accum_deques = {i: deque(maxlen=self.accumulation_size) for i in self.map_list}




    def set_queue(self, data_queue: Queue, img_queue: Queue):
        self.data_q = data_queue
        self.img_q = img_queue

    def change_zscale(self, z_min: int, z_max: int):
        self.z_min = z_min
        self.z_max = z_max
        logger.info(f'New z: {z_min} {z_max}')
        try:
            self.nn.z_min_value_changed(value=z_min)
            self.nn.z_max_value_changed(value=z_max)
        except Exception as e:
            logger.error(e)

    def accumulate_and_make_decision(self, result_dict: dict):
        accumed_results = []
        frequencies = []
        for key, key_dict in result_dict.items():
            self.accum_deques[key].appendleft(key_dict['confidence'])
            accum = sum(self.accum_deques[key])
            state = bool(accum >= self.threshold)
            accumed_results.append(state)
            if state:
                freq_shift = self.calculate_frequency(ymin=key_dict['ymin'], ymax=key_dict['ymax'])
                if self.name == '2G4':
                    freq = int(freq_shift/1000000 + 2437)
                elif self.name == '5G8':
                    freq = int(freq_shift / 1000000 + 5786.5)
                else:
                    freq = 0
                    logger.warning(f'Unknown connection name {self.name}!')
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

    def run(self):
        tcp_connection_status = False
        while not tcp_connection_status:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                for _ in range(4):
                    try:
                        s.connect(self.address)
                        tcp_connection_status = True
                        logger.success(f'Connected to {self.address}!')
                        nn_type = s.recv(2)
                        logger.info(f'NN type: {nn_type}')
                        self.nn = NNProcessing(name=self.name,
                                               weights=self.weights_path,
                                               sample_rate=self.sample_rate,
                                               width=self.signal_width,
                                               height=self.signal_height,
                                               project_path=self.project_path,
                                               map_list=self.map_list,
                                               img_size=self.img_size,
                                               msg_len=self.msg_len,
                                               z_min=self.z_min,
                                               z_max=self.z_max)
                        res_1 = np.arange(len(self.map_list) * 4)  # array of bytes to send (4 bytes for one class (float32)

                        while True:
                            s.send(res_1.tobytes())
                            arr = s.recv(self.msg_len)
                            i = 0
                            while self.msg_len > len(arr) and i < 200:
                                i += 1
                                time.sleep(0.005)
                                arr += s.recv(self.msg_len - len(arr))
                                logger.warning(f'Packet {i} missed.')
                            np_arr = np.frombuffer(arr, dtype=np.int8)
                            if np_arr.size == self.msg_len:
                                img_arr = self.nn.normalization4(np_arr.reshape(self.signal_height, self.signal_width))
                                result = self.nn.processing(img_arr)
                                df_result = result.pandas().xyxy[0]

                            if self.data_q is not None:
                                res, freq = self.accumulate_and_make_decision(
                                    self.nn.grpc_convert_result(df_result, return_data_type='dict_with_freq'))
                                self.data_q.put({'name': self.name,
                                                 'results': res,
                                                 'frequencies': freq,
                                                 'predict_df': df_result})
                            if self.img_q is not None:
                               self.img_q.put({'image': copy.deepcopy(result.render()[0]),
                                                'img_size': self.img_size,
                                                'name': self.name})

                            if not self.q_control.empty():
                                cmd_dict = self.q_control.get()
                                args = cmd_dict['args']
                                func = getattr(self, cmd_dict['func'])
                                func(*args)

                    except Exception as e:
                        # print(f'Connection failed\n{e}')
                        logger.error(e)
                        s.close()
                        time.sleep(1)
                logger.info(f'Port №{self.address} finished work')


class DataProcessingService(API_pb2_grpc.DataProcessingServiceServicer):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = load_conf(config_path=self.config_path)
        self.data_store = {}
        self.data_q = Queue(maxsize=9)
        self.img_q = Queue(maxsize=5)
        self.processes = {}
        self.connections = self.config['connections']

    def ProceedDataStream(self, request, context):
        logger.info(f'Start data stream')
        while True:
            if not self.data_q.empty():
                res = self.data_q.get()
                band_name = res['name']
                Uavs = [API_pb2.UavObject(type=API_pb2.DroneType.Autel, state=res['results'][0], freq=res['frequencies'][0]),
                          API_pb2.UavObject(type=API_pb2.DroneType.Fpv, state=res['results'][1], freq=res['frequencies'][1]),
                          API_pb2.UavObject(type=API_pb2.DroneType.Dji, state=res['results'][2], freq=res['frequencies'][2]),
                          API_pb2.UavObject(type=API_pb2.DroneType.Wifi, state=res['results'][3], freq=res['frequencies'][3])
                        ]
                # logger.info(f'Data was send from band {band_name}')
                yield API_pb2.DataResponse(band_name=band_name, uavs=Uavs)

    def SpectrogramImageStream(self, request, context):
        logger.info(f'Start image stream request: {request.band_name}')
        while True:
            if not self.img_q.empty():
                res = self.img_q.get()
                band_name = res['name']
                yield API_pb2.ImageResponse(band_name=band_name,
                                            data=res['image'].tobytes(),
                                            height=res['img_size'][0],
                                            width=res['img_size'][1])

    def GetAvailableChannels(self, request, context):
        available_channels = tuple(self.connections.keys())
        return API_pb2.ChannelsResponse(channels=available_channels)

    def StartChannel(self, request, context):
        if request.connection_name in self.connections:
            conn_name = request.connection_name
            connection = self.connections[conn_name]
            try:
                cl = Client(name=conn_name,
                            address=(str(connection['ip']), int(connection['port'])),
                            weights_path=connection['neural_network_settings']['weights_path'],
                            project_path=connection['neural_network_settings']['project_path'],
                            sample_rate=int(connection['signal_settings']['sample_rate']),
                            signal_width=int(connection['signal_settings']['signal_width']),
                            signal_height=int(connection['signal_settings']['signal_height']),
                            img_size=tuple(connection['neural_network_settings']['img_size']),
                            all_classes=tuple(connection['detection_settings']['all_classes']),
                            map_list=tuple(connection['detection_settings']['map_list']),
                            z_min=int(connection['neural_network_settings']['z_min']),
                            z_max=int(connection['neural_network_settings']['z_max']),
                            threshold=int(connection['detection_settings']['threshold']),
                            accumulation_size=int(connection['detection_settings']['accumulation_size']),
                            data_queue=self.data_q,
                            img_queue=self.img_q)

                cl.start()
                self.processes[conn_name] = cl
                logger.success(f'Connected successfully to {str(connection["ip"])}:{str(connection["port"])}')
                return API_pb2.StartChannelResponse(
                    connection_status=f'Connected successfully to {str(connection["ip"])}:{str(connection["port"])}')
            except Exception as e:
                logger.error(f'Connection error: {e}')
                return API_pb2.StartChannelResponse(
                    connection_status=f'Error with connecting to {str(connection["ip"])}:{str(connection["port"])}')
        else:
            logger.error(f'Unknown name {request.connection_name}')
            return API_pb2.StartChannelResponse(connection_status=f'Unknown channel: {request.connection_name}!')

    def ZScaleChanging(self, request, context):
        name = request.band_name
        if name in self.processes:
            self.processes[name].q_control.put({'func': 'change_zscale', 'args': (request.z_min, request.z_max)})
            #self.processes[name].change_zscale(z_min=request.z_min, z_max=request.z_max)
            logger.info(f'Z scale in channel {name} was changed on [{request.z_min}, {request.z_max}]')
            return API_pb2.ZScaleResponse(status=f'Z scale in channel {name} was changed on [{request.z_min}, {request.z_max}]')
        else:
            logger.error(f'Unknown channel {name}')
            return API_pb2.ZScaleResponse(status=f'Unknown channel: {name}!')

    def LoadConfig(self, request, context):
        original_hash = b'7L\x08\x8d\xe77d\xc4Z$/\x16\xe9\x96w\x9f\x15Uh\xd8\x9e\x81\xa2)\x13[\x95\xbb\xb5\xa5\x16\xa9'
        if request.password_hash == original_hash:
            try:
                new_config = dict(yaml.load(request.config, Loader=yaml.SafeLoader))
                dump_conf(config_path=self.config_path, config=new_config)
            except Exception as e:
                logger.error('Config did not load! Error: {e}')
                return API_pb2.LoadConfigResponse(status=f'Config did not load! Error: {e}')
            logger.success('Config loaded successfully!')
            return API_pb2.LoadConfigResponse(status='Config loaded successfully!')
        else:
            logger.warning('Incorrect password!')
            return API_pb2.LoadConfigResponse(status='Incorrect password!')


def serve():
    gRPC_PORT = 51234
    path = r'C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\gRPC_stream\with_central_freq_and_img\config_twinrx_test.yaml'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    API_pb2_grpc.add_DataProcessingServiceServicer_to_server(DataProcessingService(config_path=path), server)
    server.add_insecure_port(f'localhost:{gRPC_PORT}')
    server.start()

    print(f"gRPC Server is running on port {gRPC_PORT}...")
    try:
        while True:
            time.sleep(86400)  # Удерживаем сервер в рабочем состоянии
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
