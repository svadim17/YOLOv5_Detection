import os

FOLDER_PATH = r"D:\YOLOv5 DATASET\STEP 9\IMAGES\dji_GrozaZ2"


def calculate_files_count(filepath):
    files_count = len(os.listdir(filepath))
    return files_count


def read_files(filepath):
    signals_names = []
    for signal in os.listdir(filepath):
        signals_names.append(signal)
    return signals_names


if __name__ == '__main__':
    numb_of_files = calculate_files_count(filepath=FOLDER_PATH)
    print(f'Number of files in {FOLDER_PATH} is {numb_of_files}')

    signals = read_files(FOLDER_PATH)

    for i in range(len(signals)):
        if 'detected' in signals[i]:
            os.remove(FOLDER_PATH + '\\' + signals[i])

