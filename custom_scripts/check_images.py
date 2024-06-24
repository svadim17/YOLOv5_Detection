from PIL import Image
import os


def check_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            file_path = os.path.join(directory, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Проверяет целостность изображения
            except (IOError, SyntaxError) as e:
                print(e)


check_images(r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\data\images\training")
