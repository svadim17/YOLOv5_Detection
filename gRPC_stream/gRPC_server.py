import grpc
from concurrent import futures

import pandas as pd

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
# from nn_processing import NNProcessing

TCP_HOST = "192.168.1.3"  # The server's hostname or IP address
gRPC_PORT = '50051'

w = 1024
h = w * 3

sample_rate = 122880000
msg_len = h * w * 2 + 16
img_size = (640, 640)

ALL_CLASSES = ['dji', 'wifi', 'autel_lite', 'autel_max_4n(t)', 'autel_tag', 'fpv', '3G/4G']
MAP_LIST = ['autel', 'fpv', 'dji', 'wifi']

PROJECT_PATH = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5"
WEIGHTS_PATH = PROJECT_PATH + r"\runs\train\yolov5m_6classes_aftertrain_2\weights\best.pt"

accumulation_size = 10
threshold = accumulation_size * 0.5 * 0.6


class Client(Process):
    def __init__(self, address: tuple, weights_path, q=None):
        super().__init__()
        self.address = address
        self.weights_path = weights_path
        self.start_time = time.time()
        self.q = q
        self.accum_deques = {i: deque(maxlen=accumulation_size) for i in MAP_LIST}

    def set_queue(self, q: Queue):
        self.q = q

    def accumulate_and_make_decision(self, result_dict: dict):
        accumed_results = []
        for key, value in result_dict.items():
            self.accum_deques[key].appendleft(value)
            accum = sum(self.accum_deques[key])
            accumed_results.append(bool(accum >= threshold))
        return accumed_results

    def accumulate_and_make_decision_2(self, result_dict: dict):
        accumed_results = []
        frequencies = []
        for key, key_dict in result_dict.items():
            self.accum_deques[key].appendleft(key_dict['confidence'])
            accum = sum(self.accum_deques[key])
            state = bool(accum >= threshold)
            accumed_results.append(state)
            if state:
                freq_shift = self.calculate_frequency(ymin=key_dict['ymin'], ymax=key_dict['ymax'])
                freq = int(freq_shift/1000000 + 5786.5)
                frequencies.append(freq)
            else:
                frequencies.append(0)
        return accumed_results, frequencies

    def calculate_frequency(self, ymin, ymax):
        """ Функция считает частоту сигнала относительно центральной частоты (нулевой) """
        freq_min = (sample_rate / img_size[0]) * ymin
        freq_max = (sample_rate / img_size[0]) * ymax
        freq_width = freq_max - freq_min
        f = freq_width / 2 + freq_min
        f_center = (sample_rate / 2 - f) * (-1)
        return f_center

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            for _ in range(4):
                try:
                    s.connect(self.address)
                    logger.info(f'Connected to {self.address}!')
                    self.nn = NNProcessing(name=str(self.address),
                                           project_path=PROJECT_PATH,
                                           weights=WEIGHTS_PATH,
                                           width=w,
                                           height=h,
                                           map_list=MAP_LIST,
                                           source_device='alinx',
                                           sample_rate=sample_rate,
                                           img_size=img_size)

                    while True:
                        s.send(b'\x30')     # send for start
                        arr = s.recv(msg_len)
                        i = 0
                        while msg_len > len(arr) and i < 200:
                            i += 1
                            time.sleep(0.01)
                            arr += s.recv(msg_len - len(arr))
                            # logger.warning(f'Packet {i} missed. len = {len(arr)}')

                        if len(arr) == msg_len:
                            logger.info(f'Header: {arr[:16].hex()}')
                            np_arr = np.frombuffer(arr[16:], dtype=np.float16)
                            log_mag = np_arr.astype(np.float64)
                            # mag = (np_arr * 1.900165802481979E-9)**2 * 20
                            # mag = (np_arr * 3.29272254144689E-14)**2 * 20
                            # with np.errstate(divide='ignore'):
                            #     log_mag = np.log10(mag) * 10
                            # img_arr = self.nn.normalization4(np.fft.fftshift(log_mag.reshape(h, w)))
                            # print(max(enumerate(log_mag), key=lambda _ : _ [1]))  # поиск позиции с макс значением
                            img_arr = self.nn.normalization4(np.fft.fftshift(log_mag.reshape(h, w), axes=(1,)))
                            result = self.nn.processing(img_arr)
                            df_result = result.pandas().xyxy[0]

                        if self.q is not None:
                            res, freq = self.accumulate_and_make_decision_2(
                                self.nn.grpc_convert_result(df_result, return_data_type='dict_with_freq'))
                            self.q.put({'port': self.address[1],
                                        'results': res,
                                        'frequencies': freq,
                                        'image': copy.deepcopy(result.render()[0]),
                                        'predict_df': df_result})
                        else:
                            cv2.imshow(f"{self.address}", result.render()[0])

                except Exception as e:
                    # print(f'Connection failed\n{e}')
                    logger.error(e)
                    s.close()
                    time.sleep(1)
            logger.info(f'Port №{self.address} finished work')


class DataProcessingService(API_pb2_grpc.DataProcessingServiceServicer):
    def __init__(self, ports):
        self.data_store = {}
        self.q = Queue()
        self.processes = []

        for i in ports:
            cl = Client(address=(TCP_HOST, int(i)),
                        weights_path=WEIGHTS_PATH,
                        q=self.q)
            cl.start()
            self.processes.append(cl)

    def ProceedDataStream(self, request, context):
        while True:
            if not self.q.empty():
                res = self.q.get()
                if res['port'] == 16024:
                    band = API_pb2.Band.Band2p4
                    res['frequencies'][1] = 0
                elif res['port'] == 16058:
                    band = API_pb2.Band.Band5p8
                else:
                    band = 255
                Uavs = [API_pb2.UavObject(type=API_pb2.DroneType.Autel, state=res['results'][0], freq=res['frequencies'][0]),
                          API_pb2.UavObject(type=API_pb2.DroneType.Fpv, state=res['results'][1], freq=res['frequencies'][1]),
                          API_pb2.UavObject(type=API_pb2.DroneType.Dji, state=res['results'][2], freq=res['frequencies'][2]),
                          API_pb2.UavObject(type=API_pb2.DroneType.Wifi, state=res['results'][3], freq=res['frequencies'][3])
                        ]
                yield API_pb2.DataResponse(band=band, uavs=Uavs)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    API_pb2_grpc.add_DataProcessingServiceServicer_to_server(DataProcessingService(ports=[16024, 16058]), server)
    server.add_insecure_port(f'[::]:{gRPC_PORT}')
    server.start()

    print(f"gRPC Server is running on port {gRPC_PORT}...")
    try:
        while True:
            time.sleep(86400)  # Удерживаем сервер в рабочем состоянии
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
