from PIL import Image
import os


images_directory = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\data\obj — копия"
new_images_directory = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\data\obj"


def get_image_paths(directory):
    image_paths = []
    names = []
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            img_filepath = os.path.join(directory, file)    # get full path for signal
            print(f'\nOpening and processing {img_filepath}...')
            head_path, tail_path = os.path.split(img_filepath)
            image_paths.append(img_filepath)
            names.append(tail_path)
    return image_paths, names


if __name__ == '__main__':
    image_paths, names = get_image_paths(directory=images_directory)

    for i in range(len(image_paths)):
        try:
            img = Image.open(image_paths[i])
            img.save(new_images_directory + f'/{names[i]}')
        except Exception as e:
            print(f'Error with file {image_paths[i]}: {e}')
