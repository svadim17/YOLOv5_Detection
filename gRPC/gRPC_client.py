import grpc
import define_API_pb2_grpc
import define_API_pb2
import datetime
import socket
import time
from multiprocessing import Process, Queue
import pandas
import torch.nn.functional as F
import cv2
import torch
import numpy as np
from loguru import logger
import copy
from collections import deque


h = 2048
w = 1024
msg_len = h*w*4+16

ALL_CLASSES = ['dji', 'wifi', 'autel_lite', 'autel_max_4n(t)', 'autel_tag', 'fpv', '3G/4G']
MAP_LIST = ['autel', 'fpv', 'dji', 'wifi']
TCP_HOST = "192.168.1.3"  # The server's hostname or IP address
gRPC_PORT = '50051'
PROJECT_PATH = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5"
WEIGHTS_PATH = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\runs\train\yolov5m_6classes_aftertrain_2\weights\best.pt"


class NNProcessing(object):

    def __init__(self, name: str):
        super().__init__()
        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")
        self.last_time = time.time()
        logger.info(f'Using device: {self.device}')
        self.f = np.arange(80000000 / (-2), 80000000 / 2, 80000000 / 1024)
        self.t = np.arange(0, 1024*2048/80000000, 1024/80000000)
        self.load_model(WEIGHTS_PATH)
        self.name = name
        self.accum_size = 10
        self.accum_deques = {i: deque(maxlen=self.accum_size) for i in MAP_LIST}
        self.threshold = self.accum_size * 0.5 * 0.5

    def load_model(self, weights):
        self.model = torch.hub.load(PROJECT_PATH, 'custom',
                                    path=weights,
                                    source='local')
        # self.model.iou = 0.1
        # self.model.conf = 0.1
        # self.model.augment = True

    def normalization(self, data):
        data = np.transpose(data)
        z_min = -75
        z_max = 20
        norm_data = 255 * (data - z_min) / (z_max - z_min)
        norm_data = norm_data.astype(np.uint8)
        return norm_data

    def processing(self, norm_data):
        # Use OpenCV to create a color image from the normalized data
        color_image = cv2.applyColorMap(norm_data, cv2.COLORMAP_RAINBOW)
        screen = cv2.resize(color_image, (640, 640))
        result = self.model(screen, size=640)       # set the model use the screen
        return result

    def convert_result(self, df: pandas.DataFrame):
        labels_to_combine = ['autel_lite', 'autel_max', 'autel_pro_v3', 'autel_tag', 'autel_max_4n(t)']

        group_res = df.groupby(['name'])['confidence'].max()

        # Получаем значения этих меток, если они существуют, иначе None
        values = [group_res.get(label) for label in labels_to_combine]
        values = [value for value in values if value is not None]

        if values:
            # Выбираем максимальное значение среди доступных
            max_value = max(values)
        else:
            max_value = 0  # Или какое-то другое значение по умолчанию

        # Создаем новый Series с учетом объединения
        new_data = group_res.drop(labels_to_combine, errors='ignore')
        new_data['autel'] = max_value

        result_dict = {}
        for name in MAP_LIST:
            try:
                result_dict[name] = new_data[name]
            except KeyError:
                result_dict[name] = 0

        return self.accumulate_and_make_decision(result_dict)

    def accumulate_and_make_decision(self, result_dict: dict):
        accumed_results = []
        for key, value in result_dict.items():
            self.accum_deques[key].appendleft(value)
            accum = sum(self.accum_deques[key])
            accumed_results.append(bool(accum >= self.threshold))
        return accumed_results


class Client(Process):
    def __init__(self, address: tuple, weights_path, q=None):
        super().__init__()
        self.address = address
        self.weights_path = weights_path
        self.start_time = time.time()
        self.q = q

    def set_queue(self, q: Queue):
        self.q = q

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            for _ in range(4):
                try:
                    s.connect(self.address)
                    logger.info(f'Connected to {self.address}!')
                    self.nn = NNProcessing(name=str(self.address))
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
                            np_arr = np.frombuffer(arr[16:], dtype=np.int32)
                            # mag = (np_arr * 1.900165802481979E-9)**2 * 20
                            mag = (np_arr * 3.29272254144689E-14)**2 * 20
                            with np.errstate(divide='ignore'):
                                log_mag = np.log10(mag) * 10
                            # img_arr = self.nn.normalization4(np.fft.fftshift(log_mag.reshape(h, w)))
                            # print(max(enumerate(log_mag), key=lambda _ : _ [1]))  # поиск позиции с макс значением
                            img_arr = self.nn.normalization(np.fft.fftshift(log_mag.reshape(h, w), axes=(1,)))

                            result = self.nn.processing(img_arr)
                            df_result = result.pandas().xyxy[0]

                        if self.q is not None:
                            self.q.put({'port': self.address[1],
                                        'results': self.nn.convert_result(df_result),
                                        'image': copy.deepcopy(result.render()[0]),
                                        'predict_df': df_result})
                        else:
                            cv2.imshow(f"{self.address}", result.render()[0])

                except Exception as e:
                    # print(f'Connection failed\n{e}')
                    logger.error(e)
                    s.close()
                    time.sleep(1)
            # print(f'Port №{self.address} finished work')
            logger.info(f'Port №{self.address} finished work')


def main(PORTS):
    grpc_channel = grpc.insecure_channel(f'localhost:{gRPC_PORT}')
    stub = define_API_pb2_grpc.DataProcessingServiceStub(grpc_channel)
    q = Queue()
    processes = []

    for i in PORTS:
        cl = Client((TCP_HOST, int(i)), weights_path=WEIGHTS_PATH, q=q)
        cl.start()
        processes.append(cl)

    while 1:
        if not q.empty():
            res = q.get()
            print(res['results'])
            result = [define_API_pb2.Result(type=define_API_pb2.DroneTypes.autel, state=res['results'][0]),
                      define_API_pb2.Result(type=define_API_pb2.DroneTypes.fpv, state=res['results'][1]),
                      define_API_pb2.Result(type=define_API_pb2.DroneTypes.dji, state=res['results'][2]),
                      define_API_pb2.Result(type=define_API_pb2.DroneTypes.wifi, state=res['results'][3])]

            request = define_API_pb2.DataRequest(band=res['port']-16000,
                                                 results=result)
            try:
                response = stub.SendData(request)
                logger.info(f'gRPC response: {response.message}')
            except grpc.RpcError as e:
                logger.error(f'Failed to send data to gRPC server: {e}')
        else:
            time.sleep(0.05)


if __name__ == '__main__':
    ports = [16024, 16058]
    main(PORTS=ports)
    # import sys
    # if len(sys.argv) > 1:
    #     main(sys.argv[1:])
    # else:
    #     print('Error, enter ports for sockets as arguments')

