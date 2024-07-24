import os
import shutil
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


ALL_CLASSES = ['dji', 'wifi', 'autel_lite', 'autel_max_4n(t)', 'autel_tag', 'fpv']
BOX_COLOR = (255, 255, 255)     # white
TEXT_COLOR = (255, 0, 0)        # red
OBJ_PATH = r"D:\YOLOv5 DATASET\6 steps 6 classes\obj"
NEW_OBJ_PATH = r"D:\YOLOv5 DATASET\6 steps 6 classes\augmentated_obj"
BBOXES_STATUS = False


def visualize_bbox(img, bbox, class_name, thickness=2):
    """ This function creates bbox and text of this bbox """
    x_center = bbox[0] * 640
    y_center = bbox[1] * 640
    w = bbox[2] * 640
    h = bbox[3] * 640

    x_min, x_max = int(x_center - w / 2), int(x_center + w / 2)
    y_min, y_max = int(y_center - h / 2), int(y_center + h / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, classes):
    """ This function show image with all boxes on this image """
    img = image.copy()
    for bbox, clas in zip(bboxes, classes):
        class_name = ALL_CLASSES[clas]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


def sort_annotation_file(text: list):
    classes = []
    bboxes = []
    for line in text:
        elements = line.split(' ', 1)                       # разделение номера класса от координат
        classes.append(int(elements[0]))                    # номер класса в int
        bbox_temp = list(elements[1].split(' '))            # разделение чисел внутри координат
        bboxes.append([eval(numb) for numb in bbox_temp])   # перевод координат в int

    return classes, bboxes


def get_image_paths(directory):
    image_paths = []
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            img_filepath = os.path.join(directory, file)    # get full path for signal
            image_paths.append(img_filepath)
    return image_paths


def get_annotation_paths(directory):
    annotation_paths = []
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            annotation_filepath = os.path.join(directory, file)    # get full path for annotation
            annotation_paths.append(annotation_filepath)
    return annotation_paths


if __name__ == '__main__':
    images_paths = get_image_paths(directory=OBJ_PATH)
    annotations_paths = get_annotation_paths(directory=OBJ_PATH)

    for i in tqdm(range(len(images_paths)), desc=f'Processing files...'):
        head_path_img, tail_path_img = os.path.split(images_paths[i])
        head_path_txt, tail_path_txt = os.path.split(annotations_paths[i])

        image = cv2.imread(images_paths[i], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(annotations_paths[i], 'r') as f:
            lines = f.readlines()

        if BBOXES_STATUS:
            classes, bboxes = sort_annotation_file(lines)
            visualize(image, bboxes, classes)
            aug = A.Compose(
                [A.ChannelShuffle(p=1)],
                bbox_params=A.BboxParams(format='yolo', label_fields=['classes'])
            )
            aug_result = aug(image=image, bboxes=bboxes, classes=classes)
        else:
            aug = A.Compose(
                [A.ChannelShuffle(p=1)],
                )
            aug_result = aug(image=image)

            new_filename_img = NEW_OBJ_PATH + '\\' + 'augm_' + tail_path_img
            new_filename_txt = NEW_OBJ_PATH + '\\' + 'augm_' + tail_path_txt

            cv2.imwrite(filename=new_filename_img, img=aug_result['image'])
            shutil.copy(annotations_paths[i], new_filename_txt)


        # visualize(aug_result['image'],
        #           aug_result['bboxes'],
        #           aug_result['classes'])
        # plt.show()