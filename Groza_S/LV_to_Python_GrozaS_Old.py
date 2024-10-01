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

map_list = ['noise', 'autel', 'fpv', 'dji', 'wifi']

HOST = "127.0.0.1"  # The server's hostname or IP address
project_path = r"C:\Users\Professional\Desktop\RFBasedDetection"
weights_path = r"C:\Users\Professional\Desktop\RFBasedDetection\runs\best_AUGMENTATED.pt"
save_path = r'C:\Users\User\Documents\photo'
save_result_path = None
RETURN_MODE = 'tcp'          # None or "CUSTOM" or 'tcp'


class NNProcessing(object):

    def __init__(self, name: str):
        super().__init__()
        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")
        self.last_time = time.time()
        logger.info(f'Using device: {self.device}')
        self.f = np.arange(80000000 / (-2), 80000000 / 2, 80000000 / 1024)
        self.t = np.arange(0, 1024*2048/80000000, 1024/80000000)
        self.load_model()
        self.name = name

    def load_model(self):
        self.model = torch.hub.load(project_path, 'custom',
                               path=weights_path,
                               source='local')

    def normalization(self, data, freq=2000000000):
        data = np.transpose(data)
        if freq < 3000000000:
            k = 117
        else:
            k = 133
        z_min = -30
        z_max = 50
        norm_data = 255 * (data + k - z_min) / (z_max - z_min)
        norm_data = norm_data.astype(np.uint8)
        return norm_data

    def processing(self, data, freq):

        norm_data = self.normalization(data, freq)
        # Use OpenCV to create a color image from the normalized data
        color_image = cv2.applyColorMap(norm_data, cv2.COLORMAP_RAINBOW)
        # color_image = cv2.cvtColor(norm_data, cv2.COLOR_RGB2GRAY)

        screen = cv2.resize(color_image, (640, 640))
        if save_path is not None:
            filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            cv2.imwrite(filename=save_path + '\\' + filename + '.jpg', img=screen)

        # set the model use the screen
        result = self.model(screen, size=640)

        # cv2.imshow(f'Freq={freq} || {self.name}', result.render()[0])

        if save_result_path is not None:
            filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            cv2.imwrite(filename=save_result_path + '\\' + filename + '.jpg', img=result.render()[0])

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
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

        result_list = []
        for name in map_list:
            try:
                result_list.append(new_data[name])
            except KeyError:
                result_list.append(0)

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
                    logger.success(f'Connected to {self.address}!')
                    nn_type = int.from_bytes(s.recv(2), byteorder = 'big')
                    self.nn = NNProcessing(name=str(self.address))
                    logger.info(f'NN type: {nn_type}')
                    if nn_type == 80:
                        h = 2048
                    elif nn_type == 160:
                        h = 4096
                    else:
                        h = 2048
                    w = 1024
                    msg_len = h*w + 8
                    res_1 = np.arange(len(map_list) * 4)
                    freq = np.arange(8)
                    while True:
                        s.send(freq.tobytes() + res_1.tobytes())
                        arr = s.recv(msg_len)
                        i = 0
                        while msg_len > len(arr) and i < 10:
                            i += 1
                            time.sleep(0.005)
                            arr += s.recv(msg_len - len(arr))
                            logger.warning(f'Packet {i} missed.')
                        np_arr = np.frombuffer(arr, dtype=np.int8)
                        if np_arr.size == msg_len:

                            data = np_arr[8:].reshape(h, w)
                            freq = np_arr[0:8]
                            result = self.nn.processing(data, int.from_bytes(arr[0:8], byteorder = 'big'))
                            df_result = result.pandas().xyxy[0]

                            if RETURN_MODE == 'csv':
                                df_result.to_csv(r'C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\return_example.csv')
                                logger.success('Result saved to csv dataset')
                            elif RETURN_MODE == "tcp":
                                res_1 = self.nn.convert_result(df_result)
                                # print(f"{self.nn.name} |Recogn res: {res_1}")

                            to_print = df_result[['name', 'confidence']]
                            logger.info(f'Process {self.address} __ data size: {data.shape} \n'
                                        f'{to_print}\n'
                                        f'--------------- Time:{time.time() - self.start_time} ----------------\n')
                            self.start_time = time.time()

                except Exception as e:
                    # print(f'Connection failed\n{e}')
                    logger.error(e)
                    s.close()
                    time.sleep(1)
            # print(f'Port №{self.address} finished work')
            logger.info(f'Port №{self.address} finished work')


def main(PORTS):
    processes = []
    # PORTS = [6345, 6346]
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
