import time
import grpc
import neuro_pb2_grpc as API_pb2_grpc
import neuro_pb2 as API_pb2
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

gRPC_PORT = '50051'
MAP_LIST = ['autel', 'fpv', 'dji', 'wifi']

def dataStream(channel):
    stub = API_pb2_grpc.DataProcessingServiceStub(channel)
    responses = stub.ProceedDataStream(API_pb2.VoidRequest())
    start_time = 0

    for response in responses:
        print('------------------')
        print(f'Processing time = {time.time() - start_time}')
        print('------------------')
        start_time = time.time()
        print(f"Band: {response.band}")
        for uav in response.uavs:
            drone_name = MAP_LIST[uav.type]
            print(f"UAV Type: {drone_name}, State: {uav.state}, Frequency: {uav.freq}")

def imageStream(channel):
    stub = API_pb2_grpc.DataProcessingServiceStub(channel)
    img_responses = stub.SpectrogramImageStream(API_pb2.ImageRequest())
    for img_response in img_responses:
        band = img_response.band
        size = (img_response.height, img_response.width, 3)  # (640, 640, 3)
        img_arr = np.frombuffer(img_response.data, dtype=np.uint8).reshape(size)
        color_image = cv2.applyColorMap(img_arr, cv2.COLORMAP_RAINBOW)
        resized_img = cv2.resize(color_image, (320, 320))
        cv2.imshow(f"{band}", resized_img)
        cv2.waitKey(1)  # Required to render the image properly


def main():
    with grpc.insecure_channel(f'localhost:{gRPC_PORT}') as grpc_channel:
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(dataStream, grpc_channel)
            executor.submit(imageStream, grpc_channel)


if __name__ == '__main__':
    main()
