import cv2
import torch
import numpy as np
import os
import time
from ultralytics import YOLO
import pandas


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

project_path = r'C:\Users\v.stecko\Desktop\YOLOv5 Project\server\yolov5'
weights_1 = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\server\yolov5\runs\yolov5m_6classes_BIG_AUGMENTATED_ver6.pt"
version_1 = 'v5'

img_folder_path = r"D:\YOLOv5 DATASET\STEP 8\IMAGES\dji_40M"
all_classes = ['dji', 'wifi',  'autel_lite', 'autel_max_4n(t)', 'autel_tag', 'fpv']


def load_model(version: str, project_path: str, weights: str):
    if version == 'v5':
        model = torch.hub.load(project_path, model='custom', path=weights, source='local')

        model.iou = 0.8
        model.conf = 0.4
        model.agnostic = True
    elif version == 'v10':
        model = YOLO(weights)
    else:
        raise Exception(f"Unknown model version {version}")
    return model


def convert_result_to_pandas(results):
    columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name']

    xyxy_data = []
    results_copy = results.copy()
    for r in results_copy:
        if not r.boxes:
            return pandas.DataFrame(columns=columns)
        else:
            for box in r.boxes:
                xyxy_data.append({
                    'name': all_classes[int(box.cls.item())],
                    'xmin': box.xyxy[0][0].item(),
                    'ymin': box.xyxy[0][1].item(),
                    'xmax': box.xyxy[0][2].item(),
                    'ymax': box.xyxy[0][3].item(),
                    'confidence': box.conf.item(),
                    'class': box.cls.item()
                })

    df_result = pandas.DataFrame(xyxy_data)
    return df_result


def processing(model, version, img):
    if version == 'v5':
        result = model(img, size=640)
        df_result = result.pandas().xyxy[0]
        detected_img = result.render()[0]
    elif version == 'v10':
        middle_result = model(img)
        df_result = convert_result_to_pandas(results=middle_result)
        detected_img = img.copy()
        for result in middle_result:
            for box in result.boxes:
                cv2.rectangle(detected_img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 255, 255), 2)
                cv2.putText(detected_img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    else:
        print('Error with processing !')
    return df_result


model_1 = load_model(version_1, project_path, weights_1)
start_time = time.time()
dji_counter = 0
dji_file_count = 0
for file in os.listdir(img_folder_path):
    if file.endswith('.jpg'):
        img_filepath = os.path.join(img_folder_path, file)  # get full path for signal
        print(f'\nOpening and processing {img_filepath}...')

        img = (cv2.imread(img_filepath, cv2.COLORMAP_INFERNO))
        img_resize = cv2.resize(img, (640, 640))
        norm_image = img.astype(np.uint8)

        df_result = processing(model=model_1, version=version_1, img=norm_image.copy())
        dji_count = (df_result['name'] == 'dji').sum()
        dji_counter += dji_count
        if dji_count > 0:
            dji_file_count += 1
print(f'Found {dji_counter} dji drones in {dji_file_count} files.')
print(f'Total time is {time.time() - start_time}')























