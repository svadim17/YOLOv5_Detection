import socket
from multiprocessing import Process, Queue
import copy
from nn_processing import *

# HOST = "192.168.1.3"  # The server's hostname or IP address
# weights_path = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\runs\train\yolov5m_6classes_AUGMENTATED_3\weights\best.pt"
RETURN_MODE = None     # or 'csv' or 'tcp'
# map_list = ['autel', 'fpv', 'dji', 'wifi']
# all_classes = ['dji', 'wifi', 'autel_lite', 'autel_max_4n(t)', 'autel_tag', 'fpv', '3G/4G']


class Client(Process):
    def __init__(self, address, weights_path, map_list):
        super().__init__()
        self.address = address
        self.weights_path = weights_path
        self.start_time = time.time()
        self.q = None
        self.map_list = map_list
        self.w = 1024
        self.h = 3072
        self.msg_len = self.w * self.h * 2 + 16
        self.sample_rate = 122880000
        self.img_size = (640, 640)

    def set_queue(self, q: Queue):
        self.q = q

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            for _ in range(4):
                try:
                    s.connect(self.address)
                    logger.info(f'Connected to  Alinx {self.address}!')
                    self.nn = NNProcessing(name=str(self.address),
                                           weights=self.weights_path,
                                           sample_rate=self.sample_rate,
                                           width=self.w,
                                           height=self.h,
                                           project_path= r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5",
                                           map_list=self.map_list,
                                           source_device='alinx',
                                           img_size=self.img_size)
                    while True:
                        s.send(b'\x30')     # send for start
                        arr = s.recv(self.msg_len)
                        i = 0
                        while self.msg_len > len(arr) and i < 500:
                            i += 1
                            time.sleep(0.01)
                            arr += s.recv(self.msg_len - len(arr))
                            logger.warning(f'Packet {i} missed. len = {len(arr)}')

                        if len(arr) == self.msg_len and (arr[:16].hex() == '31000000000060000000000000000000' or
                                                        arr[:16].hex() == '30000000000060000000000000000000'):

                            logger.info(f'Header: {arr[:16].hex()}')
                            log_mag = np.frombuffer(arr[16:], dtype=np.float16)
                            log_mag = log_mag.astype(np.float64)
                            # mag = (np_arr * 1.900165802481979E-9)**2 * 20
                            # mag = (np_arr * 3.29272254144689E-14)**2 * 20
                            # with np.errstate(divide='ignore'):
                            #     log_mag = np.log10(mag) * 10
                            # img_arr = self.nn.normalization4(np.fft.fftshift(log_mag.reshape(h, w)))
                            # print(max(enumerate(log_mag), key=lambda _ : _ [1]))

                            img_arr = self.nn.normalization4(np.fft.fftshift(log_mag.reshape(self.h, self.w), axes=(1,)))

                            result = self.nn.processing(img_arr)
                            df_result = result.pandas().xyxy[0]

                        if self.q is not None:
                            self.q.put({'img_res': copy.deepcopy(result.render()[0]),
                                        'predict_res': self.nn.convert_result(df_result),
                                        'clear_image': copy.deepcopy(img_arr),
                                        'predict_df': df_result})
                        else:
                            cv2.imshow(f"{self.address}", result.render()[0])

                            if RETURN_MODE == 'csv':
                                df_result.to_csv(r'C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\return_example.csv')
                                logger.success('Result saved to csv dataset')
                            elif RETURN_MODE == "tcp":
                                res_1 = self.nn.convert_result(df_result)
                                # print(f"{self.nn.name} |Recogn res: {res_1}")

                            to_print = df_result[['name', 'confidence']]
                            logger.info(f'Process {self.address}\n'
                                        f'{to_print}\n'
                                        f'--------------- Time:{time.time() - self.start_time} ----------------\n')
                            self.start_time = time.time()

                except Exception as e:
                    # print(f'Connection failed\n{e}')
                    logger.error(e)
                    s.close()
                    time.sleep(1)
            # print(f'Port №{self.address} finished work')
            logger.info(f'Port №{self.address} finished work')


def main(PORTS):
    processes = []
    # PORTS = [6345, 6346]
    for i in PORTS:
        cl = Client(('192.168.1.3', int(i)),
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

