import os
import random
import shutil

MODE = 'MOVE'       # MOVE or COPY
FOLDER_PATH = r"D:\YOLOv5 DATASET\STEP 5\IMAGES\3G4G_part1"
NEW_FOLDER_PATH = r"D:\YOLOv5 DATASET\STEP 5\IMAGES\3G4G_part2"
NUMB_TO_MOVE = 300


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

    rand_list = random.sample(range(0, numb_of_files), NUMB_TO_MOVE)   # list of random signals

    signals = read_files(FOLDER_PATH)

    for index in rand_list:
        if MODE == 'COPY' or MODE == 'copy':
            shutil.copy(FOLDER_PATH + '\\' + signals[index], NEW_FOLDER_PATH + '\\' + signals[index])
        elif MODE == 'MOVE' or MODE == 'move':
            shutil.move(FOLDER_PATH + '\\' + signals[index], NEW_FOLDER_PATH + '\\' + signals[index])

    numb_of_files = calculate_files_count(filepath=FOLDER_PATH)
    numb_of_files_2 = calculate_files_count(filepath=NEW_FOLDER_PATH)
    print(f'After moving number of files in {FOLDER_PATH} is {numb_of_files}')
    print(f'After moving number of files in {NEW_FOLDER_PATH} is {numb_of_files_2}')
