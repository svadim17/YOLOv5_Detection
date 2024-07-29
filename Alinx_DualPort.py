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


h = 2048
w = 1024
msg_len = h*w*4+16

all_classes = ['dji', 'wifi', 'autel_lite', 'autel_max_4n(t)', 'autel_tag', 'fpv', '3G/4G']
map_list = ['noise', 'autel', 'fpv', 'dji', 'wifi', '3G/4G']
HOST = "192.168.1.3"  # The server's hostname or IP address
project_path = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5"
weights_path = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\runs\train\yolov5m_6classes_aftertrain_2\weights\best.pt"
save_path = None
save_result_path = None
RETURN_MODE = None         # None or "CUSTOM" or 'tcp'


class NNProcessing(object):

    def __init__(self, name: str, weights: str):
        super().__init__()
        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")
        self.last_time = time.time()
        logger.info(f'Using device: {self.device}')
        self.f = np.arange(80000000 / (-2), 80000000 / 2, 80000000 / 1024)
        self.t = np.arange(0, 1024*2048/80000000, 1024/80000000)
        self.load_model(weights)
        self.name = name

    def load_model(self, weights):
        self.model = torch.hub.load(project_path, 'custom',
                                    path=weights,
                                    source='local')
        # self.model.iou = 0.1
        # self.model.conf = 0.1
        # self.model.augment = True

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
        data = np.transpose(data)
        z_min = -75
        z_max = 20
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

    def processing(self, norm_data):
        # norm_data = self.normalization4(data)
        # Use OpenCV to create a color image from the normalized data
        color_image = cv2.applyColorMap(norm_data, cv2.COLORMAP_RAINBOW)
        # color_image = cv2.cvtColor(norm_data, cv2.COLOR_RGB2GRAY)

        screen = cv2.resize(color_image, (640, 640))
        if save_path is not None:
            filename = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            cv2.imwrite(filename=save_path + '\\' + filename + '.jpg', img=screen)

        # set the model use the screen
        result = self.model(screen, size=640)

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
    def __init__(self, address, weights_path):
        super().__init__()
        self.address = address
        self.weights_path = weights_path
        self.start_time = time.time()
        self.q = None

    def set_queue(self, q: Queue):
        self.q = q

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            for _ in range(4):
                try:
                    s.connect(self.address)
                    logger.info(f'Connected to {self.address}!')
                    self.nn = NNProcessing(name=str(self.address), weights=self.weights_path)
                    while True:
                        s.send(b'\x30')     # send for start
                        arr = s.recv(msg_len)
                        i = 0
                        while msg_len > len(arr) and i < 200:
                            i += 1
                            time.sleep(0.005)
                            arr += s.recv(msg_len - len(arr))
                            logger.warning(f'Packet {i} missed. len = {len(arr)}')

                        if len(arr) == msg_len:

                            logger.info(f'Header: {arr[:16].hex()}')
                            np_arr = np.frombuffer(arr[16:], dtype=np.int32)
                            # mag = (np_arr * 1.900165802481979E-9)**2 * 20
                            mag = (np_arr * 3.29272254144689E-14)**2 * 20
                            with np.errstate(divide='ignore'):
                                log_mag = np.log10(mag) * 10
                            # img_arr = self.nn.normalization4(np.fft.fftshift(log_mag.reshape(h, w)))
                            print(max(enumerate(log_mag), key=lambda _ : _ [1]))
                            img_arr = self.nn.normalization4(np.fft.fftshift(log_mag.reshape(h, w), axes=(1,)))

                            result = self.nn.processing(img_arr)
                            df_result = result.pandas().xyxy[0]

                        if self.q is not None:
                            self.q.put({'img_res': copy.deepcopy(result.render()[0]),
                                        'predict_res': self.nn.convert_result(df_result),
                                        'clear_image': copy.deepcopy(img_arr),
                                        'predict_df': df_result})
                        else:
                            cv2.imshow(f"{self.address}", result.render()[0])

                            if RETURN_MODE == 'csv':
                                df_result.to_csv(r'C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\return_example.csv')
                                logger.success('Result saved to csv dataset')
                            elif RETURN_MODE == "tcp":
                                res_1 = self.nn.convert_result(df_result)
                                # print(f"{self.nn.name} |Recogn res: {res_1}")

                            to_print = df_result[['name', 'confidence']]
                            logger.info(f'Process {self.address}\n'
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
        cl = Client((HOST, int(i)), weights_path=weights_path)
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

