import datetime
import socket
import os
import numpy as np
import time
from multiprocessing import Process

import pandas
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


h = 2048
w = 1024
msg_len = h*w
map_list = ['noise', 'autel', 'fpv', 'dji', 'wifi']


model_path = r"C:\Users\v.stecko\Desktop\NN GROZA S\860_940ver3(PC)\models\12_06_ResNet_80M_TwinRX_NO_NORM\sigmoid\Epoch12_accur_0.9826_trainLoss_0.6194_validLoss_0.6193.dict"
save_path = r"C:\Users\v.stecko\Documents\fpv"
save_result_path = None
RETURN_MODE = None          # None or "CUSTOM" or 'tcp'


class NNProcessing(object):

    def __init__(self, name: str):
        super().__init__()
        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")
        self.last_time = time.time()
        print(f'Using device: {self.device}')
        self.f = np.arange(80000000 / (-2), 80000000 / 2, 80000000 / 1024)
        self.t = np.arange(0, 1024*2048/80000000, 1024/80000000)
        self.load_model()
        self.name = name

    def load_model(self):
        self.model = torch.hub.load(r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5", 'custom',
                               path=r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\runs\train\exp10\weights\best.pt",
                               source='local')

    def normalization2(self, data):
        tensor_data = torch.tensor(data, dtype=torch.float32).to(self.device)
        norm_tensor = F.normalize(tensor_data.clone().detach(), p=1, dim=1)
        print(norm_tensor)
        norm_data = np.transpose(norm_tensor.cpu().detach().numpy() * 255).astype(np.uint8)
        return norm_data

    def normalization(self, data):
        norm_data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
        norm_data = np.transpose(norm_data.astype(np.uint8))
        return norm_data

    def normalization4(self, data):
        data = np.transpose(data + 122)
        z_min = -45
        z_max = 35
        norm_data = 255 * (data - z_min) / (z_max - z_min)
        norm_data = norm_data.astype(np.uint8)
        return norm_data

    def normalization3(self, data):
        mean = np.mean(data)
        std = np.std(data)
        norm_data_zscore = (data - mean) / std

        norm_data_minmax = 255 * (norm_data_zscore - np.min(norm_data_zscore)) / (
                np.max(norm_data_zscore) - np.min(norm_data_zscore))
        norm_data_minmax = np.transpose(norm_data_minmax.astype(np.uint8))
        return norm_data_minmax

    def processing(self, data):
        norm_data = self.normalization4(data)
        # Use OpenCV to create a color image from the normalized data
        color_image = cv2.applyColorMap(norm_data, cv2.COLORMAP_RAINBOW)
        # color_image = cv2.cvtColor(norm_data, cv2.COLOR_RGB2GRAY)

        screen = cv2.resize(color_image, (640, 640))
        if save_path is not None:
            filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            cv2.imwrite(filename=save_path + '\\' + filename + '.jpg', img=screen)

        # set the model use the screen
        result = self.model(screen, size=640)

        cv2.imshow(self.name, result.render()[0])

        if save_result_path is not None:
            filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            cv2.imwrite(filename=save_result_path + '\\' + filename + '.jpg', img=result.render()[0])

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        return result

    def convert_result(self, df: pandas.DataFrame):
        labels_to_combine = ['autel_lite', 'autel_max', 'autel_pro_v3', 'autel_tag']

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

        result_list = []
        for name in map_list:
            try:
                result_list.append(new_data[name])
            except KeyError:
                result_list.append(0)

        if [value == 0 for value in result_list]:
            print('fsdfsdfsdf')

        return np.array(result_list, dtype=np.float32)


class Client(Process):
    def __init__(self, address):
        super().__init__()
        self.address = address
        self.start_time = time.time()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            for _ in range(4):
                try:
                    s.connect(self.address)
                    print(f'Connected to {self.address}')
                    nn_type = s.recv(2)
                    self.nn = NNProcessing(name=str(self.address))
                    print(f'NN type: {nn_type}')
                    res_1 = np.arange(20)
                    while True:
                        s.send(res_1.tobytes())
                        arr = s.recv(msg_len)
                        i = 0
                        while msg_len > len(arr) and i < 10:
                            i += 1
                            time.sleep(0.005)
                            arr += s.recv(msg_len - len(arr))
                            print(f"skipped {i}")
                        np_arr = np.frombuffer(arr, dtype=np.int8)
                        if np_arr.size == msg_len:

                            result = self.nn.processing(np_arr.reshape(h, w))

                            df_result = result.pandas().xyxy[0]
                            # print(df_result)

                            if RETURN_MODE == 'csv':
                                df_result.to_csv(r'C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\return_example.csv')
                                print('Result saved to csv dataset')
                            elif RETURN_MODE == "tcp":
                                res_1 = self.nn.convert_result(df_result)
                                print(f"Recognition result: {res_1}")
                            else:
                                to_print = df_result[['name', 'confidence']]
                                print(f'--------------- Process {self.address} --------------\n'
                                      f'{to_print}\n'
                                      f'--------------- Time:{time.time() - self.start_time} ----------------\n')
                            self.start_time = time.time()

                except Exception as e:
                    print(f'Connection failed\n{e}')
                    time.sleep(1)
            print(f'Port №{self.address} finish work')


def main(PORTS):
    processes = []
    HOST = "127.0.0.1"  # The server's hostname or IP address
    #PORTS = [6345, 6346]
    for i in PORTS:
        cl = Client((HOST, int(i)))
        cl.start()
        processes.append(cl)
    for i in processes:
        i.join()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print('Error, enter ports for sockets as arguments')

