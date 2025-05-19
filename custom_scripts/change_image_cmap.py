import cv2
import os


img_folder_path = r"C:\Users\v.stecko\Documents\rainbow"
new_folder_path = r"C:\Users\v.stecko\Documents\jet"


for file in os.listdir(img_folder_path):
    if file.endswith('.jpg'):
        img_filepath = os.path.join(img_folder_path, file)  # get full path for signal
        print(f'\nOpening and processing {img_filepath}...')
        head_path, tail_path = os.path.split(img_filepath)
        name, extension = os.path.splitext(tail_path)

        image = cv2.imread(img_filepath)
        new_image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
        # new_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(filename=new_folder_path + '\\' + tail_path, img=new_image)

