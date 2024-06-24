import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


objects_path = r"/yolov5/data/obj"
labels_map = {'dji': '0', 'wifi': '1', 'autel_lite': '2', 'autel_max': '3', 'autel_pro_v3': '4', 'fpv': '5', 'autel_tag': '6'}


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
            img_filepath = os.path.join(directory, file)    # get full path for signal
            annotation_paths.append(img_filepath)
    return annotation_paths


def get_labels(annotations):
    """ This function creates a list with all labels """
    labels = []
    for annotation_file in annotations:
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                labels.append(line[0])
            f.close()
    return labels


def count_labels(labels, map):
    numb_of_classes = len(map)
    print(f'Number of unique classes: {numb_of_classes}')
    for key, value in map.items():
        numb = labels.count(value)      # считаем сколько раз значение словаря входит в список labels
        map[key] = numb
    return map


if __name__ == '__main__':
    annotations = get_annotation_paths(objects_path)

    all_labels = get_labels(annotations)
    print(f'Number of labels: {len(all_labels)}')

    labels_counts_map = count_labels(labels=all_labels, map=labels_map)
    print(labels_counts_map)

    # Plot histogram
    x_data, y_data = [], []
    for key, value in labels_counts_map.items():
        x_data.append(key)
        y_data.append(value)
    plt.bar(x_data, y_data)
    plt.grid(True)
    plt.title('Histogram of distribution labels')
    plt.show()






