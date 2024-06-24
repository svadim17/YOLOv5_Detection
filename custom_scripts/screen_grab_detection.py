import cv2
import torch
from mss import mss
import numpy as np
from PIL import Image


model = torch.hub.load(r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5", 'custom', path=r"C:\Users\v.stecko\Desktop\YOLOv5 Project\yolov5\runs\train\yolov5_STEP_1\weights\best.pt", source='local')

sct = mss()

while 1:
    top, left = 100, 100
    w, h = 640, 640
    monitor = {'top': top, 'left': left, 'width': w, 'height': h}
    img = Image.frombytes('RGB', (w, h), sct.grab(monitor).rgb)
    imr_arr = np.array(img)

    screen = cv2.cvtColor(imr_arr, cv2.COLOR_RGB2BGR)
    # set the model use the screen
    result = model(screen, size=640)
    cv2.imshow('Screen', result.render()[0])

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
