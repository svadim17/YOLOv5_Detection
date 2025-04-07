import os
import shutil

FOLDER_PATH = r"C:\Users\v.stecko\Desktop\images2\clear"
NEW_FOLDER_PATH = r"C:\Users\v.stecko\Desktop\images2\detected"


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
            shutil.copy(FOLDER_PATH + '\\' + signals[i], NEW_FOLDER_PATH + '\\' + signals[i])
            # os.remove(FOLDER_PATH + '\\' + signals[i])
