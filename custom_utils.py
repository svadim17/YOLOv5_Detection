import base64
import hashlib
import numpy as np
import cv2
import os
from collections.abc import Mapping
from enum import Enum


def deep_update(source: dict, overrides: dict):
    for key, value in overrides.items():
        if isinstance(value, Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


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


def is_list_of_nparrays(x):
    return isinstance(x, list) and all(isinstance(i, np.ndarray) for i in x)


class AlinxException(Exception):

    def __init__(self, message='Error with receiving data from hardware!'):
        self.message = message

    def __str__(self):
        return self.message


class ErrorFlag(Enum):
    no_error = 0
    error = 1
    warning = 2



if __name__ == '__main__':
    dk = create_password_hash(password='kgbradar')
    print(dk)

