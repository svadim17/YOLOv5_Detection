import os


fake_map_dict = {'0': 'autel', '1': 'wifi', '2': 'dji', '3': 'fpv'}
map_dict = {'wifi': '0', 'fpv': '1', 'dji': '2', 'autel': '3'}
filepath_annotations = r"C:\Users\v.stecko\Desktop\data part 2\labels"


def open_all_files(filepath):
    for file in os.listdir(filepath):
        file_filepath = os.path.join(filepath, file)  # get full path for signal
        print(f'\nOpening and processing {file_filepath}...')
        rename_class(file_filepath)
        # head_path, tail_path = os.path.split(file_filepath)
        # name, extension = os.path.splitext(tail_path)


def rename_class(filepath):
    with open(filepath, 'r') as f:
        new_correct_lines = []
        text = f.readlines()
        for string in text:
            fake_class = string[0]
            fake_value = fake_map_dict[fake_class]
            new_string = map_dict[fake_value] + string[1:]
            print(f'First string: {string}')
            print(f'Second string: {new_string}')
            new_correct_lines.append(new_string)        # append lines if there are more than one
        f.close()
    with open(filepath, 'w') as f:
        f.writelines(new_correct_lines)


if __name__ == '__main__':
    open_all_files(filepath_annotations)

