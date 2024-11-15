import base64
import hashlib
import numpy as np
import cv2
import os


def image_to_base64(image_path):
    if image_path is None:
        return None
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


def create_password_hash(password: str):
    salt = 'KURVA_bober'.encode()
    dk = hashlib.pbkdf2_hmac(hash_name='sha256', password=password.encode(), salt=salt, iterations=100000)
    return dk


def get_image_from_bytes(arr: bytes, size: tuple):
    img_arr = np.frombuffer(arr, dtype=np.uint8).reshape(size)

    # Convert image to base64
    _, buffer = cv2.imencode('.png', img_arr)
    img_base64 = base64.b64encode(buffer).decode()
    return img_base64


def count_files(directory):
    if os.path.isdir(directory):
        return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    else:
        return -1


if __name__ == '__main__':
    dk = create_password_hash(password='kgbradar')
    print(dk)

