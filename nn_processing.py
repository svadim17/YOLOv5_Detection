import datetime
import time
import pandas
import torch.nn.functional as F
import cv2
import torch
import numpy as np
from loguru import logger


class NNProcessing(object):

    def __init__(self, name: str,
                 weights: str,
                 sample_rate: int,
                 width: int,
                 height: int,
                 project_path: str,
                 map_list: list,
                 source_device='twinrx',
                 img_size=(640, 640)):
        super().__init__()
        self.name = name
        self.weights = weights
        self.project_path = project_path
        self.map_list = map_list
        self.source_device = source_device
        self.img_size = img_size
        msg_len = width * height * 4 + 16 if source_device == 'alinx' else width * height

        self.device = torch.device("cuda")  # = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_time = time.time()
        logger.info(f'Using device: {self.device}')
        self.f = np.arange(sample_rate / (-2), sample_rate / 2, sample_rate / width)
        self.t = np.arange(0, msg_len / sample_rate, width / sample_rate)
        self.load_model()

    def load_model(self):
        self.model = torch.hub.load(self.project_path,
                                    model='custom',
                                    path=self.weights,
                                    source='local')
        # self.model.iou = 0.1
        # self.model.conf = 0.1
        # self.model.augment = True

    def normalization(self, data):
        norm_data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
        norm_data = np.transpose(norm_data.astype(np.uint8))
        return norm_data

    def normalization4(self, data):
        if self.source_device == 'twinrx':
            data = np.transpose(data + 122)
            z_min = -40
            z_max = 40
        elif self.source_device == 'alinx':
            data = np.transpose(data)
            z_min = -75
            z_max = 20
        else:
            data = np.transpose(data)
            z_min = -75
            z_max = 20
        norm_data = 255 * (data - z_min) / (z_max - z_min)
        norm_data = norm_data.astype(np.uint8)
        return norm_data

    def processing(self, norm_data):
        # Use OpenCV to create a color image from the normalized data
        color_image = cv2.applyColorMap(norm_data, cv2.COLORMAP_RAINBOW)
        screen = cv2.resize(color_image, self.img_size)

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

        result_list = []
        for name in self.map_list:
            try:
                result_list.append(new_data[name])
            except KeyError:
                result_list.append(0)

        return np.array(result_list, dtype=np.float32)
