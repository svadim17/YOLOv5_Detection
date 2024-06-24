import os


# fake_map_dict = {'0': 'autel', '1': 'wifi', '2': 'dji', '3': 'fpv'}
map_dict = {'dji': '0', 'wifi': '1', 'autel_lite': '2', 'autel_max': '3', 'autel_pro_v3': '4', 'fpv': '5', 'autel_tag': '6'}
filepath_annotations = r"D:\YOLOv5 DATASET\STEP 3\ImgLab MARKED\autel_evo_pro_v3_alinx\labels"
created_class = 'autel_pro_v3'


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
            if fake_class == '0':
                new_string = map_dict[created_class] + string[1:]
                print(f'Old string: {string}')
                print(f'New string: {new_string}')
                new_correct_lines.append(new_string)        # append lines if there are more than one
            else:
                new_correct_lines.append(string)
        f.close()
    with open(filepath, 'w') as f:
        f.writelines(new_correct_lines)


if __name__ == '__main__':
    open_all_files(filepath_annotations)

