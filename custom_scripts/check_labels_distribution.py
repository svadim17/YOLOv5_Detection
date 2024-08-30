import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


objects_paths = [r"D:\YOLOv5 DATASET\8 steps 6 classes\obj"]
labels_map = {'dji': '0',
              'wifi': '1',
              'autel_lite': '2',
              'autel_max_4n': '3',
              'autel_tag': '4',
              'fpv': '5',
              '3G/4G': '6'}


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


def plot_annotated_histogram(labels_counts_map):
    x_data, y_data = [], []
    for key, value in labels_counts_map.items():
        x_data.append(key)
        y_data.append(value)

    plt.bar(x_data, y_data)
    plt.grid(False)
    plt.title('Histogram of distribution labels')
    # Add object count annotations above bars
    for i, (v, txt) in enumerate(zip(y_data, x_data)):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=12)

    plt.xticks(rotation=45, ha='right')  # Rotate class names for better readability
    plt.tight_layout()      # для лучшего отображения
    plt.show()


if __name__ == '__main__':
    annotations = []
    for ojb_path in objects_paths:
        annotations += get_annotation_paths(ojb_path)

    all_labels = get_labels(annotations)
    print(f'Number of labels: {len(all_labels)}')

    labels_counts_map = count_labels(labels=all_labels, map=labels_map)
    print(labels_counts_map)

    plot_annotated_histogram(labels_counts_map)
