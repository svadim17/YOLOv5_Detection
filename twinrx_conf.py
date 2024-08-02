h = 2048
w = 1024
msg_len = h*w

all_classes = ['dji', 'wifi', 'autel_lite', 'autel_max_4n(t)', 'autel_tag', 'fpv', '3G/4G']
map_list = ['noise', 'autel', 'fpv', 'dji', 'wifi', '3G/4G']
HOST = "127.0.0.1"  # The server's hostname or IP address
project_path = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5"
weights_path = r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\runs\train\yolov5m_7classes_aftertrain_2\weights\best.pt"
save_path = None
save_result_path = None
RETURN_MODE = 'tcp'          # None or "CUSTOM" or 'tcp'
IMG_SIZE = (640, 640)
FREQ_SMPLS = 80000000