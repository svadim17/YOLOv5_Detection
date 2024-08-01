import os


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640


def open_all_files(path):
    for file in os.listdir(path):
        if file.endswith('.txt'):
            file_filepath = os.path.join(path, file)
            check_annotation(file_filepath)


def check_annotation(path):
    with open(path, 'r') as f:
        text = f.readlines()
        new_text = []
        error = 0

        for string in text:
            string_symbols = string.split()
            numb_class = string_symbols[0]
            coordinates = [float(c) for c in string_symbols[1:]]  # Преобразуем строки в числа

            for i in range(len(coordinates)):
                if coordinates[i] <= 0:
                    coordinates[i] = 0.000001
                    error += 1
                if coordinates[i] > 1:
                    coordinates[i] = 1
                    error += 1

            # Проверка абсолютных координат
            x_center, y_center, width, height = coordinates
            abs_x_center = x_center * IMAGE_WIDTH
            abs_y_center = y_center * IMAGE_HEIGHT
            abs_width = width * IMAGE_WIDTH
            abs_height = height * IMAGE_HEIGHT

            x_min = abs_x_center - abs_width / 2
            x_max = abs_x_center + abs_width / 2
            y_min = abs_y_center - abs_height / 2
            y_max = abs_y_center + abs_height / 2

            # Проверка, что координаты не выходят за границы изображения
            if x_min <= 0:
                x_min = 1
                error += 1
            if x_max > IMAGE_WIDTH:
                x_max = IMAGE_WIDTH
                error += 1
            if y_min <= 0:
                y_min = 1
                error += 1
            if y_max > IMAGE_HEIGHT:
                y_max = IMAGE_HEIGHT
                error += 1

            # Пересчет относительных координат
            abs_x_center = (x_min + x_max) / 2
            abs_y_center = (y_min + y_max) / 2
            abs_width = x_max - x_min
            abs_height = y_max - y_min

            coordinates[0] = abs_x_center / IMAGE_WIDTH
            coordinates[1] = abs_y_center / IMAGE_HEIGHT
            coordinates[2] = abs_width / IMAGE_WIDTH
            coordinates[3] = abs_height / IMAGE_HEIGHT

            s = ' '
            new_line = f"{numb_class} {s.join(map(str, coordinates))}\n"
            new_text.append(new_line)
        f.close()

    if error:
        with open(path, 'w') as f:
            f.writelines(new_text)
        print(f'File {path} with invalid range was changed.')


if __name__ == '__main__':
    annotations_path = r"D:\YOLOv5 DATASET\7 steps 6 classes\step7"
    open_all_files(annotations_path)
