import torch
import cv2
from ultralytics.utils import yaml_load
from ultralytics.models.yolo import model


# "C:\Users\user\Desktop\yolov5\runs\train\yolov5m_6classes_BIG_AUGMENTATED_ver4\weights\best.pt"

if __name__ == '__main__':

    import detect

    print(detect.run(weights=r"C:\Users\user\Desktop\yolov5\runs\train\yolov5m_6classes_BIG_AUGMENTATED_ver4\weights\best.pt",
               source=r"C:\Users\user\Desktop\yolov5\data\new_img_for_test\vert_flip_chan_shuffle_%5Csaved_images%5C2023-11-21-16-18-46-647004.jpg",
               data=r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\data\dataset.yaml",
               imgsz=(640, 640),
               conf_thres=0.2,
               view_img=True,
               nosave=True,
               visualize=False,
               iou_thres=0.25))

    input()
