import os


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
            coordinates = string_symbols[1:]

            for i in range(len(coordinates)):
                value = float(coordinates[i])
                if value < 0:
                    coordinates[i] = '0'
                    error += 1
                if value > 1:
                    error += 1
                    coordinates[i] = '1'
            s = ' '
            new_line = numb_class + ' ' + s.join(coordinates) + '\n'
            new_text.append(new_line)
        f.close()

    if error:
        with open(path, 'w') as f:
            f.writelines(new_text)
        print(f'File {path} with invalid range was changed.')


if __name__ == '__main__':
    annotations_path = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\data\obj"
    open_all_files(annotations_path)
