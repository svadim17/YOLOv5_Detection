import base64
import hashlib
import numpy as np
import cv2


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_password_hash(password: str):
    salt = 'KURVA_bober'.encode()
    dk = hashlib.pbkdf2_hmac(hash_name='sha256', password=password.encode(), salt=salt, iterations=100000)
    return dk


def get_image_from_bytes(arr: bytes, size: tuple):
    img_arr = np.frombuffer(arr, dtype=np.uint8).reshape(size)
    color_image = cv2.applyColorMap(img_arr, cv2.COLORMAP_RAINBOW)

    # Convert image to base64
    _, buffer = cv2.imencode('.png', color_image)
    img_base64 = base64.b64encode(buffer).decode()
    return img_base64

