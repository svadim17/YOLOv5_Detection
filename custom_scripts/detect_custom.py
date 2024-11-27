import torch
import cv2
from yolov5.models import yolo
from ultralytics.utils import yaml_load
from ultralytics.models.yolo import model


def detect_image(image, model, conf_thres=0.4, iou_thres=0.4):
    """
        image - входное изображение
        model - натренированная модель YOLOv5
        conf_thres - порог точности (уверенности) для обнаружения
        iou_thres - порог IoU (Intersection over Union) для не максимального подавления (NMS). Значение iou_thres
                    представляет собой пороговое значение IoU, используемое для определения того, насколько сильно два
                    ограничивающих прямоугольника должны перекрываться, чтобы считаться "избыточными"
    """

    # Convert image to tensor
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).unsqueeze(0)        # add dimension
    image = image.to(model.device)

    # Output of model
    with torch.no_grad():
        predictions = model(image)[0]

    return predictions


if __name__ == '__main__':

    import detect

    print(detect.run(weights=r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\runs\train\yolov5m_6classes_BIG_AUGMENTATED_ver3\weights\best.pt",
               source=r"D:\2\11-21-12-57-22-731293.jpg",
               data=r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\data\dataset.yaml",
               imgsz=(640, 640),
               conf_thres=0.2,
               view_img=True,
               nosave=False,
               visualize=False,
               iou_thres=0.25))

    input()
