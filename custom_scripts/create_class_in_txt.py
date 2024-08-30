import os


# fake_map_dict = {'0': 'autel', '1': 'wifi', '2': 'dji', '3': 'fpv'}
map_dict = {'dji': '0',
            'wifi': '1',
            'autel_lite': '2',
            'autel_max_4n': '3',
            'autel_tag': '4',
            'fpv': '5',
            '3G/4G': '6'}

filepath_annotations = r"D:\YOLOv5 DATASET\STEP 8\ImgLab MARKED\autel_lite\labels"
# created_class = 'autel_lite'


def open_all_files(filepath):
    for file in os.listdir(filepath):
        if file.endswith('.txt'):
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
                new_string = map_dict['autel_lite'] + string[1:]
                print(f'Old string: {string}')
                print(f'New string: {new_string}')
                new_correct_lines.append(new_string)        # append lines if there are more than one
            # elif fake_class == '1':
            #     new_string = map_dict['autel_tag'] + string[1:]
            #     print(f'Old string: {string}')
            #     print(f'New string: {new_string}')
            #     new_correct_lines.append(new_string)        # append lines if there are more than one
            # elif fake_class == '2':
            #     new_string = map_dict['wifi'] + string[1:]
            #     print(f'Old string: {string}')
            #     print(f'New string: {new_string}')
            #     new_correct_lines.append(new_string)        # append lines if there are more than one
            else:
                new_correct_lines.append(string)
        f.close()
    with open(filepath, 'w') as f:
        f.writelines(new_correct_lines)


if __name__ == '__main__':
    open_all_files(filepath_annotations)

