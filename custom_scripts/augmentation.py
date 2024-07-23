import albumentations as A
import cv2
import matplotlib.pyplot as plt

ALL_CLASSES = ['dji', 'wifi', 'autel_lite', 'autel_max_4n(t)', 'autel_tag', 'fpv']
BOX_COLOR = (255, 255, 255)     # white
TEXT_COLOR = (255, 0, 0)        # red


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


if __name__ == '__main__':
    image_path = r"D:\YOLOv5 DATASET\STEP 1\ImgLab MARKED\autel_evo_lite_with_wifi\images\640_autel_lite_with_wifi_4.jpg"
    annotation_path = r"D:\YOLOv5 DATASET\STEP 1\ImgLab MARKED\autel_evo_lite_with_wifi\labels\640_autel_lite_with_wifi_4.txt"

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    classes, bboxes = sort_annotation_file(lines)

    visualize(image, bboxes, classes)

    transform = A.Compose(
        [A.HorizontalFlip(p=0.5)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['classes'])
    )

    transformed = transform(image=image, bboxes=bboxes, classes=classes)

    visualize(transformed['image'],
              transformed['bboxes'],
              transformed['classes'])

    plt.show()