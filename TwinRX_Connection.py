import socket
from multiprocessing import Process, Queue
import copy
from nn_processing import *


# HOST = "127.0.0.1"  # The server's hostname or IP address
# weights_path = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\runs\train\yolov5m_6classes_AUGMENTATED_3\weights\best.pt"
RETURN_MODE = 'tcp'     # or 'csv' or 'tcp'
# map_list = ['autel', 'fpv', 'dji', 'wifi']
# all_classes = ['dji', 'wifi', 'autel_lite', 'autel_max_4n(t)', 'autel_tag', 'fpv', '3G/4G']
# IMG_SIZE = (640, 640)


class Client(Process):
    def __init__(self, address, weights_path, map_list, z_min, z_max, z_values_queue):
        super().__init__()
        self.address = address
        self.weights_path = weights_path
        self.start_time = time.time()
        self.q = None
        self.map_list = map_list
        self.z_min = z_min
        self.z_max = z_max
        self.z_values_queue = z_values_queue
        self.w = 1024
        self.h = 2048
        self.msg_len = self.w * self.h
        self.sample_rate = 80000000
        self.img_size = (640, 640)

    def set_queue(self, q: Queue):
        self.q = q

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            for _ in range(4):
                # try:
                s.connect(self.address)
                logger.info(f'Connected to {self.address}!')
                nn_type = s.recv(2)

                self.nn = NNProcessing(name=str(self.address),
                                       weights=self.weights_path,
                                       sample_rate=self.sample_rate,
                                       width=self.w,
                                       height=self.h,
                                       project_path= r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5",
                                       map_list=self.map_list,
                                       source_device='twinrx',
                                       img_size=self.img_size,
                                       msg_len=self.msg_len,
                                       z_min=self.z_min,
                                       z_max=self.z_max)

                logger.info(f'NN type: {nn_type}')

                res_1 = np.arange(len(self.map_list) * 4)      # array of bytes to send (4 bytes for one class (float32)
                while True:

                    # Проверка обновлений в z_min и z_max
                    while not self.z_values_queue.empty():
                        key, value = self.z_values_queue.get()
                        if key == 'z_min':
                            self.nn.z_min = value
                        elif key == 'z_max':
                            self.nn.z_max = value

                    s.send(res_1.tobytes())
                    arr = s.recv(self.msg_len)
                    i = 0
                    while self.msg_len > len(arr) and i < 10:
                        i += 1
                        time.sleep(0.005)
                        arr += s.recv(self.msg_len - len(arr))
                        logger.warning(f'Packet {i} missed.')
                    np_arr = np.frombuffer(arr, dtype=np.int8)
                    if np_arr.size == self.msg_len:
                        img_arr = self.nn.normalization4(np_arr.reshape(self.h, self.w))
                        result = self.nn.processing(img_arr)
                        df_result = result.pandas().xyxy[0]

                        if self.q is not None:
                            self.q.put({'img_res': copy.deepcopy(result.render()[0]),
                                        'predict_res': self.nn.convert_result(df_result),
                                        'clear_image': copy.deepcopy(img_arr),
                                        'predict_df': df_result})
                        else:
                            cv2.imshow(str(self.address), result.render()[0])

                        if RETURN_MODE == 'csv':
                            df_result.to_csv(r'C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\return_example.csv')
                            logger.success('Result saved to csv dataset')
                        elif RETURN_MODE == "tcp":
                            res_1 = self.nn.convert_result(df_result)
                            # print(f"{self.nn.name} |Recogn res: {res_1}")

                        to_print = df_result[['name', 'confidence']]
                        # logger.info(f'Process {self.address}\n'
                        #             f'{to_print}\n'
                        #             f'--------------- Time:{time.time() - self.start_time} ----------------\n')
                        self.start_time = time.time()

                # except Exception as e:
                #     # print(f'Connection failed\n{e}')
                #     s.close()
                #     logger.error(e)
                #     time.sleep(1)
            # print(f'Port №{self.address} finished work')
            logger.info(f'Port №{self.address} finished work')


def main(PORTS):
    processes = []
    # PORTS = [6345, 6346]
    for i in PORTS:
        cl = Client(('127.0.0.1', int(i)),
                    weights_path="C:/Users/v.stecko/Desktop/YOLOv5 Project/yolov5/runs/train/yolov5m_6classes_AUGMENTATED_3/weights/best.pt")
        cl.start()
        processes.append(cl)
    for i in processes:
        i.join()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print('Error, enter ports for sockets as arguments')

