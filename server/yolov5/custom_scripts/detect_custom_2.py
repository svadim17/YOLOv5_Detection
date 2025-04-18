import cv2
import torch
import numpy as np
import os
import time


torch.cuda.empty_cache()
time.sleep(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

project_path = r'C:\Users\v.stecko\Desktop\YOLOv5 Project\server\yolov5'
weights = r'C:\Users\v.stecko\Desktop\YOLOv5 Project\server\yolov5\runs\yolov5m_6classes_BIG_AUGMENTATED_ver5.pt'

images_paths = [r"D:\YOLOv5 DATASET\STEP 5\IMAGES\autel_lite_high\2024-07-02-15-07-28-904310.jpg",
          r"D:\YOLOv5 DATASET\STEP 5\IMAGES\autel_lite_high\2024-07-02-15-05-58-369472.jpg",
          r"D:\YOLOv5 DATASET\STEP 5\IMAGES\fpv\2024-07-09-12-53-34-236272.jpg",
          # r"D:\YOLOv5 DATASET\STEP 4\IMAGES\dji\2024-06-28-11-00-46-363402.jpg",
          # r"D:\YOLOv5 DATASET\STEP 4\IMAGES\wifi\2024-06-25-14-49-06-836997.jpg",
          # r"D:\YOLOv5 DATASET\STEP 5\IMAGES\autel_lite_high\2024-07-02-15-07-28-904310.jpg",
          # r"D:\YOLOv5 DATASET\STEP 5\IMAGES\autel_lite_high\2024-07-02-15-05-58-369472.jpg",
          # r"D:\YOLOv5 DATASET\STEP 5\IMAGES\fpv\2024-07-09-12-53-34-236272.jpg",
          r"D:\YOLOv5 DATASET\STEP 4\IMAGES\dji\2024-06-28-11-00-46-363402.jpg",
          r"D:\YOLOv5 DATASET\STEP 4\IMAGES\wifi\2024-06-25-14-49-06-836997.jpg",
                ]

model = torch.hub.load(project_path, model='custom', path=weights, source='local')
model.iou = 0.8
model.conf = 0.4
model.agnostic = True

norm_images = []
start_time = time.time()
for filepath in images_paths:
    img = (cv2.imread(filepath, cv2.COLORMAP_INFERNO))
    img_resize = cv2.resize(img, (640, 640))
    norm_images.append(img.astype(np.uint8))
print(f'Time to read images: {time.time() - start_time}')

# # # PROCESSING ONE BY ONE # # #
t1 = []
for i in range(100):
    results = []
    start_time = time.time()
    for img in norm_images:
        result = model(img, size=640)
        # df_result = result.pandas().xyxy[0]
        # detected_img = result.render()[0]
        # results.append(df_result)
    t = time.time() - start_time
    t1.append(t)
    print(f'Time to detect images one by one: {t}')
t1_average = np.mean(t1[1:])
print(f'Average = {t1_average}')
print('\n\n')

torch.cuda.empty_cache()
time.sleep(7)

# # # PROCESSING BY BATCH # # #
t2 = []
for i in range(100):
    start_time = time.time()
    batch_resuls = model(norm_images, size=640)
    t = time.time() - start_time
    t2.append(t)
    print(f'Time to detect images by batch: {t}')
t2_average = np.mean(t2[1:])
print(f'Average = {t2_average}')

print(f'Batch faster on {t1_average - t2_average}')


