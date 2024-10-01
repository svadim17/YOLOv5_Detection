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
# from nn_processing import NNProcessing

TCP_HOST = "192.168.1.3"  # The server's hostname or IP address
gRPC_PORT = '50051'

w = 1024
h = w * 3       # 3072

SAMPLE_RATE = 122880000
IMG_SIZE = (640, 640)

ALL_CLASSES = ['dji', 'wifi', 'autel_lite', 'autel_max_4n(t)', 'autel_tag', 'fpv', '3G/4G']
MAP_LIST = ['autel', 'fpv', 'dji', 'wifi']

PROJECT_PATH = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5"
WEIGHTS_PATH = PROJECT_PATH + r"\runs\train\yolov5m_6classes_AUGMENTATED_3\weights\best.pt"

ACCUMULATION_SIZE = 10
THRESHOLD = ACCUMULATION_SIZE * 0.5 * 0.6

CALCULATE_LOG = False
if CALCULATE_LOG:
    MSG_LEN = h * w * 4 + 16
else:
    MSG_LEN = h * w * 2 + 16

Z_MIN_2G4 = -20
Z_MAX_2G4 = 75
Z_MIN_5G8 = -24
Z_MAX_5G8 = 77


class Client(Process):
    def __init__(self, address: tuple, weights_path: str, z_min: int, z_max: int, data_queue=None, img_queue=None):
        super().__init__()
        self.address = address
        self.weights_path = weights_path
        self.start_time = time.time()
        self.data_q = data_queue
        self.img_q = img_queue
        self.z_min = z_min
        self.z_max = z_max
        self.accum_deques = {i: deque(maxlen=ACCUMULATION_SIZE) for i in MAP_LIST}

    def set_queue(self, data_queue: Queue, img_queue: Queue):
        self.data_q = data_queue
        self.img_q = img_queue

    def accumulate_and_make_decision(self, result_dict: dict):
        accumed_results = []
        frequencies = []
        for key, key_dict in result_dict.items():
            self.accum_deques[key].appendleft(key_dict['confidence'])
            accum = sum(self.accum_deques[key])
            state = bool(accum >= THRESHOLD)
            accumed_results.append(state)
            if state:
                freq_shift = self.calculate_frequency(ymin=key_dict['ymin'], ymax=key_dict['ymax'])
                if self.address[1] == 16024:
                    freq = int(freq_shift/1000000 + 2437)
                elif self.address[1] == 16058:
                    freq = int(freq_shift / 1000000 + 5786.5)
                else:
                    freq = 0
                    logger.warning(f'Unknown port {self.address[1]}!')
                frequencies.append(freq)
            else:
                frequencies.append(0)
        return accumed_results, frequencies

    def calculate_frequency(self, ymin, ymax):
        """ Функция считает частоту сигнала относительно центральной частоты (нулевой) и возвращает смещение """
        freq_min = (SAMPLE_RATE / IMG_SIZE[0]) * ymin
        freq_max = (SAMPLE_RATE / IMG_SIZE[0]) * ymax
        freq_width = freq_max - freq_min
        f = freq_width / 2 + freq_min
        f_center = (SAMPLE_RATE / 2 - f) * (-1)
        return f_center

    def run(self):
        tcp_connection_status = False
        while not tcp_connection_status:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                for _ in range(4):
                    try:
                        s.connect(self.address)
                        tcp_connection_status = True
                        logger.info(f'Connected to {self.address}!')
                        self.nn = NNProcessing(name=str(self.address),
                                               weights=WEIGHTS_PATH,
                                               sample_rate=SAMPLE_RATE,
                                               width=w,
                                               height=h,
                                               project_path=PROJECT_PATH,
                                               map_list=MAP_LIST,
                                               img_size=IMG_SIZE,
                                               msg_len=MSG_LEN,
                                               z_min=self.z_min,
                                               z_max=self.z_max)

                        while True:
                            s.send(b'\x30')     # send for start
                            arr = s.recv(MSG_LEN)
                            i = 0
                            while MSG_LEN > len(arr) and i < 200:
                                i += 1
                                time.sleep(0.01)
                                arr += s.recv(MSG_LEN - len(arr))
                                # logger.warning(f'Packet {i} missed. len = {len(arr)}')
                            if len(arr) == MSG_LEN:
                                if CALCULATE_LOG:
                                    logger.info(f'Header: {arr[:16].hex()}')
                                    np_arr = np.frombuffer(arr[16:], dtype=np.int32)
                                    # mag = (np_arr * 1.900165802481979E-9)**2 * 20
                                    mag = (np_arr * 3.29272254144689E-14)**2 * 20
                                    with np.errstate(divide='ignore'):
                                        log_mag = np.log10(mag) * 10
                                    # img_arr = self.nn.normalization4(np.fft.fftshift(log_mag.reshape(h, w)))
                                    # print(max(enumerate(log_mag), key=lambda _ : _ [1]))  # поиск позиции с макс значением
                                else:
                                    logger.info(f'Header: {arr[:16].hex()}')
                                    log_mag = np.frombuffer(arr[16:], dtype=np.float16)
                                    log_mag = log_mag.astype(np.float64)

                                img_arr = self.nn.normalization(np.fft.fftshift(log_mag.reshape(h, w), axes=(1,)))
                                result = self.nn.processing(img_arr)
                                df_result = result.pandas().xyxy[0]

                            if self.data_q is not None:
                                res, freq = self.accumulate_and_make_decision(
                                    self.nn.grpc_convert_result(df_result, return_data_type='dict_with_freq'))
                                self.data_q.put({'port': self.address[1],
                                            'results': res,
                                            'frequencies': freq,
                                            'predict_df': df_result})
                            if self.img_q is not None:
                                self.img_q.put({'image': copy.deepcopy(result.render()[0]),
                                                'img_size': IMG_SIZE,
                                                'port': self.address[1]})

                    except Exception as e:
                        # print(f'Connection failed\n{e}')
                        logger.error(e)
                        s.close()
                        time.sleep(1)
                logger.info(f'Port №{self.address} finished work')


class DataProcessingService(API_pb2_grpc.DataProcessingServiceServicer):
    def __init__(self, ports):
        self.data_store = {}
        self.data_q = Queue()
        self.img_q = Queue()
        self.processes = []

        for port in ports:
            try:
                if port == 16024:
                    cl = Client(address=(TCP_HOST, int(port)),
                                weights_path=WEIGHTS_PATH,
                                z_min=Z_MIN_2G4,
                                z_max=Z_MAX_2G4,
                                data_queue=self.data_q,
                                img_queue=self.img_q)
                elif port == 16058:
                    cl = Client(address=(TCP_HOST, int(port)),
                                weights_path=WEIGHTS_PATH,
                                z_min=Z_MIN_5G8,
                                z_max=Z_MAX_5G8,
                                data_queue=self.data_q,
                                img_queue=self.img_q)
                else:
                    logger.error(f'Unknown port {port}')

                cl.start()
                self.processes.append(cl)
            except Exception as e:
                print(e)
                time.sleep(1)

    def ProceedDataStream(self, request, context):
        while True:
            if not self.data_q.empty():
                res = self.data_q.get()
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

    def SpectrogramImageStream(self, request, context):
        while True:
            if not self.img_q.empty():
                res = self.img_q.get()
                if res['port'] == 16024:
                    band = API_pb2.Band.Band2p4

                elif res['port'] == 16058:
                    band = API_pb2.Band.Band5p8
                else:
                    band = 255
                yield API_pb2.ImageResponse(band=band,
                                            data=res['image'].tobytes(),
                                            height=res['img_size'][0],
                                            width=res['img_size'][1])


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
