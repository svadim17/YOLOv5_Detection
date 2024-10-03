import cv2
import os


def resize_image(image_path: str, target_size: tuple):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)
    resized_image_path = image_path.replace('.jpg', '_resized.jpg')
    cv2.imwrite(resized_image_path, resized_image)


if __name__ == "__main__":
    target_size = (640, 640)  # resulting resolution
    images_path = "D:\dji 40"

    all_images_filepaths = []

    # Collect all images filepaths
    for img in os.listdir(images_path):
        img_filepath = os.path.join(images_path, img)  # get full path for image
        all_images_filepaths.append(img_filepath)

    for img in all_images_filepaths:
        print(f'Processing file {img}...')
        resize_image(image_path=img, target_size=target_size)
