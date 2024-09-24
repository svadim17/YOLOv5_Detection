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
                 map_list: tuple,
                 source_device='twinrx',
                 img_size=(640, 640),
                 msg_len=0,
                 z_min=-20,
                 z_max=75):
        super().__init__()
        self.name = name
        self.weights = weights
        self.project_path = project_path
        self.map_list = map_list
        self.img_size = img_size
        self.z_min = z_min
        self.z_max = z_max
        torch.cuda.empty_cache()
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
        # data = np.transpose(data + 122)
        data = np.transpose(data)

        norm_data = 255 * (data - self.z_min) / (self.z_max - self.z_min)
        norm_data = norm_data.astype(np.uint8)
        return norm_data

    def z_min_value_changed(self, value):

        self.z_min = value

    def z_max_value_changed(self, value):
        self.z_max = value

    def processing(self, norm_data):
        # Use OpenCV to create a color image from the normalized data
        color_image = cv2.applyColorMap(norm_data, cv2.COLORMAP_RAINBOW)
        screen = cv2.resize(color_image, self.img_size)

        result = self.model(screen, size=self.img_size[0])       # set the model use the screen
        return result

    def convert_result(self, df: pandas.DataFrame, return_data_type='list'):
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

        if return_data_type == 'list':
            result_list = []
            for name in self.map_list:
                try:
                    result_list.append(new_data[name])
                except KeyError:
                    result_list.append(0)
            return np.array(result_list, dtype=np.float32)

        elif return_data_type == 'dict':
            result_dict = {}
            for name in self.map_list:
                try:
                    result_dict[name] = new_data[name]
                except KeyError:
                    result_dict[name] = 0
            return result_dict

    def grpc_convert_result(self, df: pandas.DataFrame, return_data_type='dict_with_freq'):
        """ Преобразование датафрейма в словарь с  координатами объекта на картинке """
        labels_to_combine = ['autel_lite', 'autel_max', 'autel_pro_v3', 'autel_tag', 'autel_max_4n(t)']

        # Группировка по имени и выбор строки с максимальным confidence
        idx = df.groupby(['name'])['confidence'].idxmax()
        group_res = df.loc[idx]

        # Получаем значения confidence для меток, которые нужно объединить
        confidence_values = [group_res[group_res['name'] == label]['confidence'].values[0]
                             for label in labels_to_combine if label in group_res['name'].values]

        if confidence_values:
            max_confidence = max(confidence_values)
            max_row = group_res[group_res['confidence'] == max_confidence].iloc[0]
            max_ymin = max_row['ymin']
            max_ymax = max_row['ymax']
        else:
            max_confidence = 0
            max_ymin = 0
            max_ymax = 0

        # Создаем новый DataFrame, исключая метки для объединения
        filtered_data = group_res[~group_res['name'].isin(labels_to_combine)]   # отбираем только те строки, у которых name НЕ входит в labels_to_combine
        new_data = filtered_data.copy()     # создаем копию отфильтрованных данных
        new_data = new_data.set_index('name')       # устанавливаем столбец 'name' в качестве индекса
        new_data.loc['autel'] = {'confidence': max_confidence, 'ymin': max_ymin, 'ymax': max_ymax}      # добавляем один скомбинированный autel

        if return_data_type == 'dict_with_freq':
            result_dict = {}
            for name in self.map_list:
                try:
                    result_dict[name] = {
                        'confidence': new_data.at[name, 'confidence'],
                        'ymin': new_data.at[name, 'ymin'],
                        'ymax': new_data.at[name, 'ymax']
                    }
                except KeyError:
                    result_dict[name] = {'confidence': 0, 'ymin': 0, 'ymax': 0}

            return result_dict

        # elif return_data_type == 'dict_with_freq':
        #     result_dict = {}
        #     for name in self.map_list:
        #         try:
        #             result_dict[name]['ymin'] = new_data2[name]['ymin']
        #             result_dict[name]['ymax'] = new_data2[name]['ymax']
        #             result_dict[name]['confidence'] = new_data2[name]
        #         except KeyError:
        #             result_dict[name]['ymin'] = 0
        #             result_dict[name]['ymax'] = 0
        #             result_dict[name]['confidence'] = 0
        #     print(result_dict)
        #     return result_dict
