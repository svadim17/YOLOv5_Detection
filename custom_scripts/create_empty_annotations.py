import os


images_path = r""
annotations_path_to_save = r""


def create_annotations(img_path, save_path):
    for file in os.listdir(img_path):
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            file_path = os.path.join(img_path, file)                # get full path for signal
            head_path, tail_path = os.path.split(file_path)
            name, extension = os.path.splitext(tail_path)

            annotation_name = name + '.txt'
            annotation_path = os.path.join(save_path, annotation_name)
            open(annotation_path, 'w').close()
            print(f'Empty annotation for {file} was created.')


if __name__ == "__main__":
    create_annotations(img_path=images_path, save_path=annotations_path_to_save)


