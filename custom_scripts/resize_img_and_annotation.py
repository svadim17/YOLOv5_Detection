import cv2
import os
import numpy as np


def resize_image_and_annotations(image_path, annotations_path, target_size=(640, 640)):
    """
    Перемасштабирует изображение и обновляет его аннотации.

    Args:
        image_path (str): Путь к файлу изображения.
        annotations_path (str): Путь к файлу аннотаций (.txt).
        target_size (tuple): Желаемый размер выходного изображения (по умолчанию (640, 640)).

    Returns:
        None
    """

    # Загрузите изображение
    image = cv2.imread(image_path)

    print(image.shape)

    # Проверьте, является ли изображение 1024x1024
    if image.shape[0] != 1024 or image.shape[1] != 1024:
        raise ValueError("Изображение должно иметь размер 1024x1024.")

    # Загрузите аннотации
    with open(annotations_path, 'r') as f:
        annotations = []
        for line in f:
            class_num, x, y, w, h = line.strip().split()
            annotations.append({'x': float(x), 'y': float(y), 'width': float(w), 'height': float(h)})

    # Определите параметры перемасштабирования
    scale_factor = target_size[0] / image.shape[0]

    # Перемасштабируйте изображение
    resized_image = cv2.resize(image, target_size)

    # Обновите аннотации
    for annotation in annotations:
        annotation['x'] *= scale_factor
        annotation['y'] *= scale_factor
        annotation['width'] *= scale_factor
        annotation['height'] *= scale_factor

    # Сохраните перемасштабированное изображение
    resized_image_path = image_path.replace('.jpg', '_resized.jpg')
    cv2.imwrite(resized_image_path, resized_image)

    # Обновите аннотации в файле
    with open(annotations_path, 'w') as f:
        for annotation in annotations:
            f.write(f"{class_num} {annotation['x']} {annotation['y']} {annotation['width']} {annotation['height']}\n")


if __name__ == "__main__":
    target_size = (640, 640)  # resulting resolution

    images_path = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\data\obj\imgs"
    labels_path = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\data\obj\labels"

    all_images_filepaths = []
    all_labels_filepaths = []

    # Collect all images filepaths
    for img in os.listdir(images_path):
        img_filepath = os.path.join(images_path, img)  # get full path for image
        all_images_filepaths.append(img_filepath)

    # Collect all labels filepaths
    for label in os.listdir(labels_path):
        label_filepath = os.path.join(labels_path, label)  # get full path for label
        all_labels_filepaths.append(label_filepath)

    # Resize images and annotations
    if len(all_images_filepaths) == len(all_labels_filepaths):
        for i in range(len(all_images_filepaths)):
            resize_image_and_annotations(all_images_filepaths[i], all_labels_filepaths[i], target_size)
